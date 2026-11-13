// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlExecutionProvider/GpuEvent.h"
#include <directx/d3dx12.h>
#include <optional>

namespace dml_ep {

    // A contiguous range of descriptors.
    struct DescriptorRange
    {
        ID3D12DescriptorHeap* heap;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
        D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle;
    };

    // Wraps an ID3D12DescriptorHeap to allocate descriptor ranges.
    class DescriptorHeap
    {
    public:
        // Wraps an existing heap.
        explicit DescriptorHeap(ID3D12DescriptorHeap* heap);

        // Reserves descriptors from the end of the heap. Returns nullopt if there is
        // no space left in the heap.
        std::optional<DescriptorRange> TryAllocDescriptors(
            uint32_t numDescriptors,
            GpuEvent completionEvent,
            D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
            );

        // Reserves a PERSISTENT range from the TOP of the heap that is never recycled.
        // Used by command-list reuse: a replayed list bakes these descriptor handles into
        // its recorded command list, so they must stay valid for the heap's lifetime.
        // Transient allocations grow UP from 0 and reset to 0; persistent reservations grow
        // DOWN from capacity. The two are spatially independent, so a reservation can be
        // made at any time (even mid-inference, after transient scratch is in use) as long
        // as they have not met. Returns nullopt if the heap flags mismatch or the reserved
        // region would collide with the transient high-water mark.
        std::optional<DescriptorRange> TryReservePersistent(
            uint32_t numDescriptors,
            D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
            );

        GpuEvent GetLastCompletionEvent() const
        {
            return m_completionEvent;
        }

        uint32_t GetCapacity() const
        {
            return m_capacity;
        }

        // True if this heap holds any persistent reservation (must not be Trim()'d).
        bool HasPersistentReservation() const
        {
            return m_persistentTop < m_capacity;
        }

    private:
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_heap;
        uint32_t m_capacity = 0;
        uint32_t m_size = 0;
        uint32_t m_handleIncrementSize = 0;
        CD3DX12_CPU_DESCRIPTOR_HANDLE m_headCpuHandle;
        CD3DX12_GPU_DESCRIPTOR_HANDLE m_headGpuHandle;
        D3D12_DESCRIPTOR_HEAP_FLAGS m_heapFlags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

        // Persistent reservations (reuse lists) occupy [m_persistentTop, m_capacity) and
        // grow DOWNWARD. Transient allocations occupy [0, m_size) and grow upward; a
        // transient alloc fails (-> new heap) if m_size would exceed m_persistentTop.
        // m_persistentTop only decreases (reservations are never individually freed), but
        // it is bounded per session: the pool is owned by the per-session ExecutionContext
        // and destroyed at EP release, so reservations do not accumulate across models.
        uint32_t m_persistentTop = 0;  // initialized to m_capacity in the constructor

        // Most recent GPU completion event. Allocations are always done at the end,
        // so there is no fragmentation of the heap.
        GpuEvent m_completionEvent;
    };

    // Manages a pool of CBV/SRV/UAV descriptors.
    class DescriptorPool
    {
    public:
        DescriptorPool(ID3D12Device* device, uint32_t initialCapacity);

        // Reserves a contiguous range of descriptors from a single descriptor heap. The
        // lifetime of the referenced descriptor heap is managed by the DescriptorPool class.
        // The caller must supply a GpuEvent that informs the pool when the reserved descriptors
        // are no longer required.
        DescriptorRange AllocDescriptors(
            uint32_t numDescriptors,
            GpuEvent completionEvent,
            D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
            );

        // Reserves a PERSISTENT descriptor range from the shared heap, for command-list
        // reuse. The returned range lives in the SAME heap transient allocations use, so a
        // reuse list that binds this heap never forces a descriptor-heap switch against the
        // recorder's per-op work. The range is never recycled. Grows the pool (new heap)
        // only if the current heap can't fit the reservation alongside existing use —
        // which would reintroduce a switch, so the shared heap is sized generously.
        DescriptorRange ReservePersistent(
            uint32_t numDescriptors,
            D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
            );

        // Releases all descriptor heaps that contain only descriptors which have completed
        // their work on the GPU. Heaps holding a persistent reservation are never released.
        void Trim();

        // Returns the total capacity of all heaps.
        uint32_t GetTotalCapacity() const;

    private:
        Microsoft::WRL::ComPtr<ID3D12Device> m_device;
        std::vector<DescriptorHeap> m_heaps;
        const uint32_t m_initialHeapCapacity;

        void CreateHeap(uint32_t numDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags);
    };

}  // namespace dml_ep
