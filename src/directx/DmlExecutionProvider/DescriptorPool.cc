// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "DmlExecutionProvider/precomp.h"

#include <debugapi.h>

namespace dml_ep {

    // Amortization size for spill heaps that are NOT the shared shader-visible heap (e.g.
    // the FLAG_NONE staging descriptors used by FillBufferWithPattern). Sized so the heap
    // is reused across allocations without reallocating, but not inflated to the shared
    // heap's large reserve+scratch capacity, which those heaps don't need.
    static constexpr uint32_t kAuxiliaryHeapCapacity = 2048;

    DescriptorHeap::DescriptorHeap(ID3D12DescriptorHeap* heap) :
        m_heap(heap),
        m_capacity(heap->GetDesc().NumDescriptors),
        m_headCpuHandle(heap->GetCPUDescriptorHandleForHeapStart()),
        m_headGpuHandle(heap->GetGPUDescriptorHandleForHeapStart()),
        m_heapFlags(heap->GetDesc().Flags)
    {
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        ORT_THROW_IF_FAILED(heap->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));
        m_handleIncrementSize = device->GetDescriptorHandleIncrementSize(heap->GetDesc().Type);
        m_persistentTop = m_capacity;  // persistent reservations grow down from the top
    }

    std::optional<DescriptorRange> DescriptorHeap::TryAllocDescriptors(
        uint32_t numDescriptors, 
        GpuEvent completionEvent,
        D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags
        )
    {
        // Bail if the desired heap creation flags are incompatible with the existing heap.
        if (m_heapFlags != heapFlags)
        {
            return std::nullopt;
        }

        if ((m_completionEvent.fence != nullptr) && (m_completionEvent.IsSignaled()))
        {
            // Transient allocations grow up from 0; when the most recent completion event
            // is signaled, all previous transient allocations have completed and the whole
            // transient region [0, m_persistentTop) is reusable. Persistent reservations
            // live at the TOP and are untouched by this reset.
            m_size = 0;
            m_headCpuHandle = m_heap->GetCPUDescriptorHandleForHeapStart();
            m_headGpuHandle = m_heap->GetGPUDescriptorHandleForHeapStart();
        }

        // Transient grows up toward the persistent region at the top. Fail (-> new heap)
        // if this allocation would cross into the reserved persistent descriptors.
        if (m_size + numDescriptors > m_persistentTop)
        {
            return std::nullopt;
        }

        DescriptorRange range = { m_heap.Get(), m_headCpuHandle, m_headGpuHandle };

        m_size += numDescriptors;
        m_completionEvent = completionEvent;
        m_headCpuHandle.Offset(numDescriptors, m_handleIncrementSize);
        m_headGpuHandle.Offset(numDescriptors, m_handleIncrementSize);

        return range;
    }

    std::optional<DescriptorRange> DescriptorHeap::TryReservePersistent(
        uint32_t numDescriptors,
        D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags
        )
    {
        if (m_heapFlags != heapFlags)
        {
            return std::nullopt;
        }

        // If the last transient batch has completed on the GPU, its region is free; drop
        // the high-water mark to 0 so this reservation isn't rejected by a stale peak from
        // a prior inference (which would needlessly spill it to a new heap -> a switch).
        if ((m_completionEvent.fence != nullptr) && (m_completionEvent.IsSignaled()))
        {
            m_size = 0;
            m_headCpuHandle = m_heap->GetCPUDescriptorHandleForHeapStart();
            m_headGpuHandle = m_heap->GetGPUDescriptorHandleForHeapStart();
        }

        // Carve from the TOP, growing downward. Fail (-> new heap) if the reservation
        // would cross below the transient high-water mark. m_size is the current peak of
        // transient use; descriptors below it may be in flight, so the reserved region
        // must stay strictly above it. This is spatially independent of WHEN transient
        // allocations happen, so a reuse list can reserve mid-inference safely.
        if (numDescriptors > m_persistentTop || m_persistentTop - numDescriptors < m_size)
        {
            return std::nullopt;
        }

        m_persistentTop -= numDescriptors;

        CD3DX12_CPU_DESCRIPTOR_HANDLE cpu(m_heap->GetCPUDescriptorHandleForHeapStart());
        CD3DX12_GPU_DESCRIPTOR_HANDLE gpu(m_heap->GetGPUDescriptorHandleForHeapStart());
        cpu.Offset(m_persistentTop, m_handleIncrementSize);
        gpu.Offset(m_persistentTop, m_handleIncrementSize);
        return DescriptorRange{ m_heap.Get(), cpu, gpu };
    }

    DescriptorPool::DescriptorPool(ID3D12Device* device, uint32_t initialCapacity) :
        m_device(device),
        m_initialHeapCapacity(initialCapacity)
    {
        CreateHeap(initialCapacity, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
    }

    DescriptorRange DescriptorPool::AllocDescriptors(
        uint32_t numDescriptors, 
        GpuEvent completionEvent,
        D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags
        )
    {
        // Attempt to allocate from an existing heap.
        for (DescriptorHeap& heap : m_heaps)
        {
            auto descriptorRange = heap.TryAllocDescriptors(numDescriptors, completionEvent, heapFlags);
            if (descriptorRange.has_value())
            {
                return descriptorRange.value();
            }
        }

        // A new descriptor heap must be created, sized generously so it is reused across
        // future allocations (heaps recycle on fence signal) rather than reallocated.
        // m_initialHeapCapacity is sized for the SHARED shader-visible heap (transient
        // scratch + persistent reuse reservations in one heap, to avoid heap switches);
        // a heap with other flags (e.g. the FLAG_NONE staging descriptors used by
        // FillBufferWithPattern) gets no benefit from that large size, so it uses a
        // modest amortization default instead of committing a 65536-entry heap.
        uint32_t newHeapCapacity = (heapFlags == D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
            ? std::max(numDescriptors, m_initialHeapCapacity)
            : std::max(numDescriptors, kAuxiliaryHeapCapacity);
        CreateHeap(newHeapCapacity, heapFlags);
        auto descriptorRange = m_heaps.back().TryAllocDescriptors(numDescriptors, completionEvent, heapFlags);
        assert(descriptorRange.has_value());
        return descriptorRange.value();
    }

    DescriptorRange DescriptorPool::ReservePersistent(
        uint32_t numDescriptors,
        D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags
        )
    {
        // Prefer the first heap (the long-lived shared heap) so reuse descriptors and the
        // recorder's transient scratch share one heap -> no descriptor-heap switch.
        for (DescriptorHeap& heap : m_heaps)
        {
            auto range = heap.TryReservePersistent(numDescriptors, heapFlags);
            if (range.has_value())
            {
                return range.value();
            }
        }

        // No existing heap could host the reservation (shared heap's reserve+transient
        // budget is exhausted, or it's too small). Create a dedicated heap. This is the
        // budget-limit fallback: a reuse list in its OWN heap reintroduces the descriptor-
        // heap switch this feature avoids, so hitting this path means the shared heap
        // (m_initialHeapCapacity) is undersized for this graph's partition count × ring
        // depth. Note it so the perf cliff is observable rather than silent.
        OutputDebugStringA(
            "[DescriptorPool] ReservePersistent budget exceeded: reuse list spilled to a "
            "new heap (descriptor-heap switch reintroduced). Consider a larger pool.\n");
        uint32_t newHeapCapacity = (heapFlags == D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
            ? std::max(numDescriptors, m_initialHeapCapacity)
            : std::max(numDescriptors, kAuxiliaryHeapCapacity);
        CreateHeap(newHeapCapacity, heapFlags);
        auto range = m_heaps.back().TryReservePersistent(numDescriptors, heapFlags);
        assert(range.has_value());
        return range.value();
    }

    void DescriptorPool::Trim()
    {
        // Remove any heaps that are not pending execution, EXCEPT heaps holding a
        // persistent reservation: reuse lists bake those descriptor handles into their
        // recorded command lists, so the heap must outlive any Trim().
        auto it = std::remove_if(m_heaps.begin(), m_heaps.end(), [](const DescriptorHeap& heap) {
            if (heap.HasPersistentReservation())
            {
                return false;
            }
            auto completionEvent = heap.GetLastCompletionEvent();
            return !completionEvent.fence || completionEvent.IsSignaled();
        });

        m_heaps.erase(it, m_heaps.end());
    }

    void DescriptorPool::CreateHeap(uint32_t numDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags)
    {
        // This pool only manages CBV/SRV/UAV descriptors.
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Flags = heapFlags;
        desc.NumDescriptors = numDescriptors;
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
        ORT_THROW_IF_FAILED(m_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(heap.GetAddressOf())));

        m_heaps.push_back(DescriptorHeap{heap.Get()});
    }

    uint32_t DescriptorPool::GetTotalCapacity() const
    {
        uint32_t capacity = 0;

        for (auto& heap : m_heaps)
        {
            capacity += heap.GetCapacity();
        }

        return capacity;
    }

}  // namespace dml_ep
