// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dml_readback_heap.h"
#include "dml_perf_timer.h"

namespace dml_ep {

    Microsoft::WRL::ComPtr<ID3D12Resource> PluginDmlReadbackHeap::CreateReadbackHeap(ID3D12Device* device, size_t size)
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> readbackHeap;
        auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size);

        THROW_IF_FAILED(device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(readbackHeap.ReleaseAndGetAddressOf())));

        return readbackHeap;
    }

    PluginDmlReadbackHeap::PluginDmlReadbackHeap(ID3D12Device* device, ExecutionContext* executionContext)
        : m_device(device)
        , m_executionContext(executionContext)
    {
    }

    static size_t ComputeNewCapacity(size_t existingCapacity, size_t desiredCapacity)
    {
        size_t newCapacity = existingCapacity;

        while (newCapacity < desiredCapacity)
        {
            if (newCapacity >= std::numeric_limits<size_t>::max() / 2)
            {
                // Overflow; there's no way we can satisfy this allocation request
                THROW_HR(E_OUTOFMEMORY);
            }

            newCapacity *= 2; // geometric growth
        }

        return newCapacity;
    }

    void PluginDmlReadbackHeap::EnsureReadbackHeap(size_t size)
    {
        if (!m_readbackHeap)
        {
            // Initialize the readback heap for the first time
            m_capacity = ComputeNewCapacity(c_initialCapacity, size);
            m_readbackHeap = CreateReadbackHeap(m_device.Get(), m_capacity);
        }
        else if (m_capacity < size)
        {
            // Ensure there's sufficient capacity
            m_capacity = ComputeNewCapacity(m_capacity, size);

            m_readbackHeap = nullptr;
            m_readbackHeap = CreateReadbackHeap(m_device.Get(), m_capacity);
        }
    }

    void PluginDmlReadbackHeap::ReadbackFromGpu(
        gsl::span<std::byte> dst,
        ID3D12Resource* src,
        uint64_t srcOffset,
        D3D12_RESOURCE_STATES srcState)
    {
        EnsureReadbackHeap(dst.size());

#ifdef DML_PERF_PROFILE
        uint64_t _rb_t0 = PerfNowUs();
        PERF_TIMER_LOG("[PERF] ReadbackFromGpu ENTER (single, ", dst.size(), " bytes): ", _rb_t0, " us\n");
#endif

        // Copy from the source resource into the readback heap
        m_executionContext->CopyBufferRegion(
            m_readbackHeap.Get(),
            0,
            D3D12_RESOURCE_STATE_COPY_DEST,
            src,
            srcOffset,
            srcState,
            dst.size());

#ifdef DML_PERF_PROFILE
        uint64_t _rb_t_copy = PerfNowUs();
        PERF_TIMER_LOG("[PERF] ReadbackFromGpu CopyBufferRegion: ", _rb_t_copy, " us (+", _rb_t_copy - _rb_t0, ")\n");
#endif

        // Wait for completion and map the result
        m_executionContext->Flush();

#ifdef DML_PERF_PROFILE
        uint64_t _rb_t_flush = PerfNowUs();
        PERF_TIMER_LOG("[PERF] ReadbackFromGpu flush: ", _rb_t_flush, " us (+", _rb_t_flush - _rb_t_copy, ")\n");
#endif

        m_executionContext->GetCurrentCompletionEvent().WaitForSignal(m_executionContext->CpuSyncSpinningEnabled());

#ifdef DML_PERF_PROFILE
        uint64_t _rb_t_wait = PerfNowUs();
        uint64_t _rb_wait_delta = _rb_t_wait - _rb_t_flush;
        PERF_TIMER_LOG("[PERF] ReadbackFromGpu wait: ", _rb_t_wait, " us (+", _rb_wait_delta, ")",
            (_rb_wait_delta > 1000 ? " *** STALL ***" : ""), "\n");
#endif

        m_executionContext->ReleaseCompletedReferences();

        // Map the readback heap and copy it into the destination
        void* readbackHeapData = nullptr;
        ORT_THROW_IF_FAILED(m_readbackHeap->Map(0, nullptr, &readbackHeapData));
        memcpy(dst.data(), readbackHeapData, dst.size());
        m_readbackHeap->Unmap(0, nullptr);

#ifdef DML_PERF_PROFILE
        { uint64_t _t = PerfNowUs(); PERF_TIMER_LOG("[PERF] ReadbackFromGpu EXIT: ", _t, " us (+", _t - _rb_t0, " total)\n"); }
#endif
    }

    void PluginDmlReadbackHeap::ReadbackFromGpu(
        gsl::span<void*> dst,
        gsl::span<const uint32_t > dstSizes,
        gsl::span<ID3D12Resource*> src,
        D3D12_RESOURCE_STATES srcState)
    {
        if (dst.empty())
        {
            return;
        }

        uint32_t totalSize = 0;
        for (auto size : dstSizes)
        {
            totalSize += size;
        }

        EnsureReadbackHeap(totalSize);

#ifdef DML_PERF_PROFILE
        uint64_t _rb_t0 = PerfNowUs();
        PERF_TIMER_LOG("[PERF] ReadbackFromGpu ENTER (batched, ", dst.size(), " tensors, ", totalSize, " bytes): ", _rb_t0, " us\n");
#endif

        // Copy from the source resource into the readback heap
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dst.size(); ++i)
        {
            m_executionContext->CopyBufferRegion(
                m_readbackHeap.Get(),
                offset,
                D3D12_RESOURCE_STATE_COPY_DEST,
                src[i],
                0,
                srcState,
                dstSizes[i]);

            offset += dstSizes[i];
        }

#ifdef DML_PERF_PROFILE
        uint64_t _rb_t_copy = PerfNowUs();
        PERF_TIMER_LOG("[PERF] ReadbackFromGpu CopyBufferRegion: ", _rb_t_copy, " us (+", _rb_t_copy - _rb_t0, ")\n");
#endif

        // Wait for completion and map the result
        m_executionContext->Flush();

#ifdef DML_PERF_PROFILE
        uint64_t _rb_t_flush = PerfNowUs();
        PERF_TIMER_LOG("[PERF] ReadbackFromGpu flush: ", _rb_t_flush, " us (+", _rb_t_flush - _rb_t_copy, ")\n");
#endif

        m_executionContext->GetCurrentCompletionEvent().WaitForSignal(m_executionContext->CpuSyncSpinningEnabled());

#ifdef DML_PERF_PROFILE
        uint64_t _rb_t_wait = PerfNowUs();
        uint64_t _rb_wait_delta = _rb_t_wait - _rb_t_flush;
        PERF_TIMER_LOG("[PERF] ReadbackFromGpu wait: ", _rb_t_wait, " us (+", _rb_wait_delta, ")",
            (_rb_wait_delta > 1000 ? " *** STALL ***" : ""), "\n");
#endif

        m_executionContext->ReleaseCompletedReferences();

        // Map the readback heap and copy it into the destination
        void* readbackHeapData = nullptr;
        THROW_IF_FAILED(m_readbackHeap->Map(0, nullptr, &readbackHeapData));

        // Copy from the source resource into the readback heap
        offset = 0;
        for (uint32_t i = 0; i < dst.size(); ++i)
        {
            memcpy(dst[i], static_cast<uint8_t*>(readbackHeapData) + offset, dstSizes[i]);
            offset += dstSizes[i];
        }

        m_readbackHeap->Unmap(0, nullptr);

#ifdef DML_PERF_PROFILE
        { uint64_t _t = PerfNowUs(); PERF_TIMER_LOG("[PERF] ReadbackFromGpu EXIT: ", _t, " us (+", _t - _rb_t0, " total)\n"); }
#endif
    }

}  // namespace dml_ep
