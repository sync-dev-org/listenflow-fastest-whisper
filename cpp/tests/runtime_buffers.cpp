#include "runtime_driver.hpp"
#include <cstdlib>
#include <stdexcept>

using namespace listenflow::fastest_whisper::runtime_detail;
static int live = 0;
static bool fail_allocation = false;
static bool fail_pinned = false, fail_event = false, fail_stream = false;
static bool fail_query = false, fail_sync = false;
static int streams = 0, events = 0, event_records = 0, event_queries = 0, synchronizations = 0;
static CUcontext current = nullptr;
// Only the external CUDA driver boundary is substituted. Allocation pools,
// reference counts, leases and cleanup are the production implementations.
extern "C" CUresult CUDAAPI cuCtxPushCurrent(CUcontext context) { current = context; return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuCtxPopCurrent(CUcontext *context) { *context = current; current = nullptr; return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuMemAlloc(CUdeviceptr *out, size_t bytes) {
    if (fail_allocation) return CUDA_ERROR_OUT_OF_MEMORY;
    *out = reinterpret_cast<CUdeviceptr>(std::malloc(bytes));
    ++live;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemFree(CUdeviceptr pointer) {
    std::free(reinterpret_cast<void *>(pointer)); --live; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemHostAlloc(void **out, size_t bytes, unsigned int) {
    if (fail_pinned) return CUDA_ERROR_OUT_OF_MEMORY;
    *out = std::malloc(bytes); ++live; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemFreeHost(void *pointer) { std::free(pointer); --live; return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuStreamCreate(CUstream *out, unsigned int flags) {
    if (fail_stream) return CUDA_ERROR_OUT_OF_MEMORY;
    if (flags != CU_STREAM_NON_BLOCKING) return CUDA_ERROR_INVALID_VALUE;
    *out = reinterpret_cast<CUstream>(1); ++streams; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamDestroy(CUstream) { --streams; return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuEventCreate(CUevent *out, unsigned int flags) {
    if (fail_event) return CUDA_ERROR_OUT_OF_MEMORY;
    if (flags != CU_EVENT_DISABLE_TIMING) return CUDA_ERROR_INVALID_VALUE;
    *out = reinterpret_cast<CUevent>(1); ++events; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventDestroy(CUevent) { --events; return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuEventRecord(CUevent, CUstream) { ++event_records; return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuEventQuery(CUevent) {
    ++event_queries; return fail_query ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamSynchronize(CUstream) {
    ++synchronizations; return fail_sync ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}
static void expect(bool value, const char *why) { if (!value) throw std::runtime_error(why); }
int main() {
    // SP-LFFASTESTWHISPER-001: retained device results must not be overwritten,
    // and releasing or closing their owner returns all native allocations.
    buffer_pool pool(reinterpret_cast<CUcontext>(1));
    auto first = pool.acquire(64);
    auto address = first->address;
    auto second = pool.acquire(64);
    expect(second->address != address && live == 2, "retained buffer was reused");
    first.reset();
    auto reused = pool.acquire(32);
    expect(reused->address == address && live == 2, "released buffer was not reused");
    fail_allocation = true;
    try { pool.acquire(128); return 1; }
    catch (const failure &error) { expect(error.status == LFFW_OUT_OF_MEMORY, "allocation status lost"); }
    expect(live == 2, "failed allocation leaked");
    pool.close();
    expect(live == 0 && reused->address == 0 && second->address == 0, "close retained GPU allocation");
    second.reset(); reused.reset();
    expect(live == 0, "late lease release freed an allocation twice");

    // SP-LFFASTESTWHISPER-001: pinned allocation failures leave no resource,
    // and an explicitly released host allocation is never freed twice.
    fail_allocation = false; fail_pinned = true;
    try { allocation host(reinterpret_cast<CUcontext>(1), 64, true); return 2; }
    catch (const failure &error) { expect(error.status == LFFW_OUT_OF_MEMORY, "pinned allocation status lost"); }
    expect(live == 0 && current == nullptr, "failed pinned allocation retained storage or context");
    fail_pinned = false;
    { allocation host(reinterpret_cast<CUcontext>(1), 64, true); host.close(); host.close(); }
    expect(live == 0, "pinned storage was freed twice");

    // SP-LFFASTESTWHISPER-017: failure to allocate a completion event returns
    // the already-created stream, including while the exception is retained.
    fail_event = true;
    try { gpu_queue queue; return 3; }
    catch (const failure &) { expect(streams == 0 && events == 0, "event creation leaked a stream"); }
    fail_event = false; fail_stream = true;
    try { gpu_queue queue; return 4; }
    catch (const failure &) { expect(streams == 0 && events == 0, "stream creation retained a queue"); }
    fail_stream = false;

    // Repeated successful exchanges reuse one event, do not synchronize the
    // stream, and never wait twice for the same completed work.
    {
        gpu_queue queue;
        queue.pending = true; queue.finish(); queue.finish();
        queue.pending = true; queue.finish();
        expect(streams == 1 && events == 1 && event_records == 2 && synchronizations == 0,
               "completion event was recreated or a completed exchange waited twice");
        queue.close(); queue.close();
        expect(streams == 0 && events == 0, "idempotent close retained or double-freed a queue");
    }

    // A failed completion drains before resources are released, retains the
    // original diagnostic if the drain fails too, and closes only once.
    {
        gpu_queue queue;
        queue.pending = true; fail_query = fail_sync = true;
        try { queue.close(); return 5; }
        catch (const std::exception &error) {
            expect(std::string(error.what()).find("cuEventQuery") != std::string::npos &&
                   std::string(error.what()).find("cuStreamSynchronize") != std::string::npos,
                   "cleanup replaced the original completion failure");
        }
        expect(streams == 0 && events == 0 && !queue.pending, "failed close retained queue ownership");
        auto queries = event_queries;
        queue.close();
        expect(event_queries == queries, "closed queue tried to wait on destroyed events");
    }
}
