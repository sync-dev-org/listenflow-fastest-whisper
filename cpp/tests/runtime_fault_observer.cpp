#include <cuda.h>
#include <tensoracle/builder.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <functional>
#include <string>
#include <vector>

// Substitute only external CUDA/tensoracle calls. The real inference C ABI,
// network wrappers, resource owners, graph cache and binding run unchanged.
// Device addresses point to zeroed host storage: no GPU is initialized here.
static int live = 0, total_allocations = 0, next_failure = 0, generation = 0, graph_captures = 0;
static int graph_submissions = 0;
static int event_ticks = 0;
static int timing_events = 0, query_delay = 0, queries_remaining = 0;
static bool product_tokens = false, capturing = false;
static float product_logprob = -0.01f, product_silence = 0.1f;
static std::function<void()> captured_body;
static std::map<CUgraph, std::function<void()>> graph_bodies;
static std::map<CUgraphExec, std::function<void()>> graph_executables;
static std::map<CUevent, unsigned> event_flags;
static unsigned cleanup_failures = 0;
static std::map<CUdeviceptr, size_t> storage;
static std::mutex completion_mutex;
static std::condition_variable completion_changed;
static bool hold_completion = false;
static std::atomic<bool> completion_entered{false};
static bool fail(int code) {
    if (cleanup_failures & (1u << code)) {
        cleanup_failures &= ~(1u << code);
        return true;
    }
    if (next_failure != code) return false;
    next_failure = 0; return true;
}
template <class T> static T handle() { ++live; return reinterpret_cast<T>(new int(0)); }
template <class T> static void release(T value) { delete reinterpret_cast<int *>(value); --live; }
extern "C" CUresult CUDAAPI cuInit(unsigned int) { return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuDeviceGetCount(int *out) { *out = 1; return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuDeviceGet(CUdevice *out, int ordinal) { *out = ordinal; return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuDeviceGetAttribute(int *out, CUdevice_attribute attr, CUdevice) {
    *out = attr == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? 12 : 0; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *out, CUdevice) {
    *out = reinterpret_cast<CUcontext>(1); ++live; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice) {
    if (fail(11)) return CUDA_ERROR_UNKNOWN;
    --live; return CUDA_SUCCESS;
}
// Count primary-context references without allocating a real driver context.
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext *out) { *out = reinterpret_cast<CUcontext>(1); return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuCtxPushCurrent(CUcontext) { return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuCtxPopCurrent(CUcontext *out) { *out = nullptr; return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuStreamCreate(CUstream *out, unsigned int) { *out = handle<CUstream>(); return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuStreamDestroy(CUstream value) {
    if (fail(10)) return CUDA_ERROR_UNKNOWN;
    release(value); return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventCreate(CUevent *out, unsigned int flags) {
    *out = handle<CUevent>(); event_flags[*out] = flags;
    if (!(flags & CU_EVENT_DISABLE_TIMING)) ++timing_events;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventDestroy(CUevent value) {
    if (fail(9)) return CUDA_ERROR_UNKNOWN;
    event_flags.erase(value); release(value); return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventRecord(CUevent event, CUstream) {
    *reinterpret_cast<int *>(event) = ++event_ticks;
    if (event_flags[event] & CU_EVENT_DISABLE_TIMING) queries_remaining = query_delay;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventElapsedTime(float *out, CUevent start, CUevent stop) {
    *out = float(*reinterpret_cast<int *>(stop) - *reinterpret_cast<int *>(start)) * 0.001f;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventQuery(CUevent) {
    std::unique_lock<std::mutex> lock(completion_mutex);
    if (hold_completion) {
        completion_entered = true;
        completion_changed.wait(lock, [] { return !hold_completion; });
    }
    if (fail(1)) return CUDA_ERROR_UNKNOWN;
    if (queries_remaining > 0) { --queries_remaining; return CUDA_ERROR_NOT_READY; }
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamSynchronize(CUstream) { return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuMemAlloc(CUdeviceptr *out, size_t bytes) {
    *out = reinterpret_cast<CUdeviceptr>(std::calloc(1, bytes));
    storage[*out] = bytes; ++live; ++total_allocations; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemFree(CUdeviceptr address) {
    if (fail(8)) return CUDA_ERROR_UNKNOWN;
    storage.erase(address); std::free(reinterpret_cast<void *>(address)); --live; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemHostAlloc(void **out, size_t bytes, unsigned int) {
    if (fail(2)) return CUDA_ERROR_OUT_OF_MEMORY;
    *out = std::calloc(1, bytes); ++live; ++total_allocations; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemFreeHost(void *address) {
    if (fail(6)) return CUDA_ERROR_UNKNOWN;
    std::free(address); --live; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr target, const void *source, size_t bytes, CUstream) {
    std::memcpy(reinterpret_cast<void *>(target), source, bytes); return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync(void *target, CUdeviceptr source, size_t bytes, CUstream) {
    if (fail(4)) return CUDA_ERROR_UNKNOWN;
    std::memcpy(target, reinterpret_cast<void *>(source), bytes); return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr target, CUdeviceptr source, size_t bytes, CUstream) {
    std::memmove(reinterpret_cast<void *>(target), reinterpret_cast<void *>(source), bytes); return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr target, unsigned char value, size_t bytes, CUstream) {
    std::memset(reinterpret_cast<void *>(target), value, bytes); return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *copy, CUstream) {
    for (size_t row = 0; row < copy->Height; ++row)
        std::memmove(reinterpret_cast<char *>(copy->dstDevice) + row * copy->dstPitch,
                     reinterpret_cast<const char *>(copy->srcDevice) + row * copy->srcPitch, copy->WidthInBytes);
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamBeginCapture(CUstream, CUstreamCaptureMode) {
    capturing = true; captured_body = {}; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamEndCapture(CUstream, CUgraph *out) {
    capturing = false; *out = handle<CUgraph>(); graph_bodies[*out] = captured_body; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *out, CUgraph graph, unsigned long long) {
    if (fail(5)) return CUDA_ERROR_UNKNOWN;
    ++graph_captures;
    *out = handle<CUgraphExec>(); graph_executables[*out] = graph_bodies[graph]; return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuGraphDestroy(CUgraph value) { graph_bodies.erase(value); release(value); return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec value) {
    if (fail(7)) return CUDA_ERROR_UNKNOWN;
    graph_executables.erase(value); release(value); return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuGraphLaunch(CUgraphExec graph, CUstream) {
    ++graph_submissions;
    if (product_tokens && graph_executables[graph]) graph_executables[graph]();
    return CUDA_SUCCESS;
}
struct tensoracle_engine {};
struct tensoracle_execution {};
extern "C" tensoracle_status tensoracle_open_engine(const tensoracle_builder_options *,
    const tensoracle_network_builder *, tensoracle_engine **out, tensoracle_error **error) noexcept {
    *out = new tensoracle_engine; ++live; *error = nullptr; return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_engine_path(const tensoracle_engine *, const char **out,
    tensoracle_error **error) noexcept { *out = "/test/no-engine-file"; *error = nullptr; return TENSORACLE_SUCCESS; }
extern "C" tensoracle_status tensoracle_engine_destroy(tensoracle_engine *value) noexcept {
    if (value) { delete value; --live; } return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_execution_create(const tensoracle_engine *, tensoracle_execution **out,
    tensoracle_error **error) noexcept { *out = new tensoracle_execution; ++live; *error = nullptr; return TENSORACLE_SUCCESS; }
extern "C" tensoracle_status tensoracle_engine_scratch_size(const tensoracle_engine *, size_t *out,
    tensoracle_error **error) noexcept { *out = 256; *error = nullptr; return TENSORACLE_SUCCESS; }
extern "C" tensoracle_status tensoracle_execution_create_with_scratch(const tensoracle_engine *,
    const tensoracle_scratch_memory *scratch, tensoracle_execution **out, tensoracle_error **error) noexcept {
    *error = nullptr;
    auto found = storage.find(reinterpret_cast<CUdeviceptr>(scratch->address));
    if (found == storage.end() || scratch->size < 256 || scratch->size > found->second)
        return TENSORACLE_INVALID_ARGUMENT;
    *out = new tensoracle_execution; ++live; return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_execution_destroy(tensoracle_execution *value) noexcept {
    if (value) { delete value; --live; } return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_execution_set_input_shapes(tensoracle_execution *,
    const tensoracle_input_shape *const *, size_t, tensoracle_error **error) noexcept {
    *error = nullptr; return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_execution_enqueue(tensoracle_execution *,
    const tensoracle_tensor_address *const *addresses, size_t count, tensoracle_cuda_stream,
    tensoracle_error **error) noexcept {
    *error = nullptr;
    if (fail(3)) return TENSORACLE_INTERNAL_ERROR;
    if (product_tokens) {
        std::vector<std::pair<std::string, void *>> values;
        for (size_t i = 0; i < count; ++i) values.emplace_back(addresses[i]->name, addresses[i]->address);
        auto perform = [values] {
            std::map<std::string, void *> tensors(values.begin(), values.end());
            if (!tensors.count("positions")) return;
            const auto position = *static_cast<int32_t *>(tensors["positions"]);
            const auto token = position < 4 ? 1 : 3;
            if (tensors.count("token")) *static_cast<int32_t *>(tensors["token"]) = token;
            if (tensors.count("greedy")) {
                auto *out = static_cast<float *>(tensors["greedy"]);
                out[0] = float(token); out[1] = product_logprob; out[2] = product_silence;
            }
            if (tensors.count("sampled")) {
                auto *out = static_cast<float *>(tensors["sampled"]);
                const auto rows = storage.at(reinterpret_cast<CUdeviceptr>(out)) / (3 * sizeof(float));
                for (size_t row = 0; row < rows; ++row) {
                    out[row * 3] = float(token); out[row * 3 + 1] = -0.01f; out[row * 3 + 2] = 0.1f;
                }
            }
            if (tensors.count("logits")) {
                auto *out = static_cast<float *>(tensors["logits"]);
                const auto elements = storage.at(reinterpret_cast<CUdeviceptr>(out)) / sizeof(float);
                std::fill_n(out, elements, -10.0f); out[token] = 10;
            }
        };
        if (capturing) captured_body = perform;
        else perform();
    }
    // Distinct external outputs make aliasing of host staging observable.
    ++generation;
    for (size_t i = 0; i < count; ++i) {
        if (std::strcmp(addresses[i]->name, "encoder") == 0) {
            auto bytes = storage.at(reinterpret_cast<CUdeviceptr>(addresses[i]->address));
            std::fill_n(static_cast<float *>(addresses[i]->address), bytes / sizeof(float), float(generation));
        }
    }
    return TENSORACLE_SUCCESS;
}
extern "C" int lffw_test_live_resources() { return live; }
extern "C" int lffw_test_timing_events() { return timing_events; }
extern "C" void lffw_test_query_delay(int count) { query_delay = count; }
extern "C" void lffw_test_enable_product_tokens(int enable) { product_tokens = enable; }
extern "C" void lffw_test_quality(float logprob, float silence) { product_logprob = logprob; product_silence = silence; }
extern "C" int lffw_test_total_allocations() { return total_allocations; }
extern "C" size_t lffw_test_device_bytes() {
    size_t bytes = 0;
    for (const auto &entry : storage) bytes += entry.second;
    return bytes;
}
extern "C" int lffw_test_graph_captures() { return graph_captures; }
extern "C" int lffw_test_graph_submissions() { return graph_submissions; }
extern "C" void lffw_test_fail_next(int code) { next_failure = code; }
extern "C" void lffw_test_fail_cleanup(unsigned mask) { cleanup_failures = mask; }
extern "C" void lffw_test_hold_completion(int value) {
    { std::lock_guard<std::mutex> lock(completion_mutex); hold_completion = bool(value); }
    completion_changed.notify_all();
}
extern "C" int lffw_test_completion_entered() { return completion_entered; }
