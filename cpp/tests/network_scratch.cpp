#include <cuda.h>
#include <listenflow/whisper_network.hpp>
#include <tensoracle/builder.h>
#include <tensoracle/engine.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

// Only the external tensoracle C ABI and CUDA driver are replaced. The test
// calls the production engine_build and network_execution implementations.
namespace fw = listenflow::fastest_whisper;
struct allocation {
    size_t bytes;
    CUcontext context;
    size_t borrowers = 0;
};
static std::map<CUdeviceptr, allocation> allocations;
static CUcontext current = reinterpret_cast<CUcontext>(1);
static std::vector<CUcontext> context_stack;
static CUdeviceptr next_address = 256;
static size_t required_bytes = 4096, legacy_creates = 0, shared_creates = 0, opens = 0;
static tensoracle_status next_creation = TENSORACLE_SUCCESS;
static bool allocation_failure = false, release_violation = false;
static bool free_failure = false;
struct tensoracle_engine { size_t bytes; };
struct tensoracle_execution { CUdeviceptr scratch; };

extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext *value) {
    *value = current;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxPushCurrent(CUcontext value) {
    context_stack.push_back(current);
    current = value;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxPopCurrent(CUcontext *value) {
    *value = current;
    current = context_stack.back();
    context_stack.pop_back();
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemAlloc(CUdeviceptr *value, size_t bytes) {
    if (allocation_failure)
        return CUDA_ERROR_OUT_OF_MEMORY;
    if (!bytes)
        return CUDA_ERROR_INVALID_VALUE;
    *value = next_address;
    next_address += ((bytes + 255) / 256) * 256;
    allocations.emplace(*value, allocation{bytes, current});
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemFree(CUdeviceptr value) {
    if (free_failure) {
        free_failure = false;
        return CUDA_ERROR_UNKNOWN;
    }
    auto found = allocations.find(value);
    if (found == allocations.end() || found->second.borrowers || found->second.context != current) {
        release_violation = true;
        return CUDA_ERROR_INVALID_VALUE;
    }
    allocations.erase(found);
    return CUDA_SUCCESS;
}
extern "C" tensoracle_status tensoracle_open_engine(const tensoracle_builder_options *,
    const tensoracle_network_builder *, tensoracle_engine **engine, tensoracle_error **) noexcept {
    ++opens;
    *engine = new tensoracle_engine{required_bytes};
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_engine_destroy(tensoracle_engine *engine) noexcept {
    delete engine;
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_engine_scratch_size(const tensoracle_engine *engine,
    size_t *bytes, tensoracle_error **) noexcept {
    *bytes = engine->bytes;
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_execution_create(const tensoracle_engine *,
    tensoracle_execution **execution, tensoracle_error **) noexcept {
    ++legacy_creates;
    *execution = new tensoracle_execution{0};
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_execution_create_with_scratch(const tensoracle_engine *engine,
    const tensoracle_scratch_memory *memory, tensoracle_execution **execution,
    tensoracle_error **) noexcept {
    auto result = next_creation;
    next_creation = TENSORACLE_SUCCESS;
    if (result != TENSORACLE_SUCCESS)
        return result;
    const auto pointer = reinterpret_cast<CUdeviceptr>(memory->address);
    auto found = allocations.find(pointer);
    if (!pointer || memory->size < engine->bytes || found == allocations.end() ||
        memory->size > found->second.bytes || found->second.context != current)
        return TENSORACLE_INVALID_ARGUMENT;
    ++shared_creates;
    ++found->second.borrowers;
    *execution = new tensoracle_execution{pointer};
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_execution_destroy(tensoracle_execution *execution) noexcept {
    if (execution->scratch) {
        auto found = allocations.find(execution->scratch);
        if (found == allocations.end() || !found->second.borrowers)
            release_violation = true;
        else
            --found->second.borrowers;
    }
    delete execution;
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_execution_set_input_shapes(tensoracle_execution *,
    const tensoracle_input_shape *const *, size_t, tensoracle_error **) noexcept {
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_execution_enqueue(tensoracle_execution *execution,
    const tensoracle_tensor_address *const *, size_t, tensoracle_cuda_stream,
    tensoracle_error **) noexcept {
    return allocations.count(execution->scratch) ? TENSORACLE_SUCCESS : TENSORACLE_INVALID_ARGUMENT;
}
static void expect(bool value, const char *why) {
    if (!value)
        throw std::runtime_error(why);
}
template <class Operation> static void rejects(Operation operation, tensoracle_status status) {
    try {
        operation();
    } catch (const fw::model_error &error) {
        expect(error.status() == status, "scratch failure lost its backend status");
        return;
    }
    throw std::runtime_error("scratch failure was not returned to the caller");
}
int main(int argc, char **argv) {
    try {
        expect(argc == 2, "pass the test's private directory");
        std::filesystem::create_directories(argv[1]);
        auto auxiliary = (std::filesystem::path(argv[1]) / "config").string();
        std::ofstream(auxiliary) << "scratch fixture";
        fw::model_source source;
        source.config = {128, 1500, 1280, 20, 5120, 32, 20, 5120, 32, 448, 51866};
        source.weights = {"fixture.safetensors"};
        source.config_file = source.tokenizer_file = auxiliary;
        source.fc1_library = "fixture-fc1.so";
        fw::engine_build build(source, fw::graph_kind::sampled_decoder, fw::precision::fp16, 5);

        auto engine = build.open();
        expect(!legacy_creates && shared_creates == 1,
               "cached-plugin probe must borrow caller-owned scratch");
        expect(allocations.size() == 1 && allocations.begin()->second.bytes >= required_bytes,
               "one engine must own one allocation of at least scratch_size bytes");
        const auto address = allocations.begin()->first;
        auto buckets = build.create_sampled_executions(engine);
        expect(buckets.size() == 16 && shared_creates == 17 && !legacy_creates,
               "all sixteen buckets must use caller-owned scratch");
        expect(allocations.size() == 1 && allocations.at(address).borrowers == 16,
               "bucket executions must share their engine's allocation");
        buckets.front().execution.close();
        auto replacement = build.create_execution(engine, 1);
        expect(allocations.size() == 1 && allocations.at(address).borrowers == 16,
               "replacement execution must reuse surviving scratch");
        next_creation = TENSORACLE_INVALID_ARGUMENT;
        rejects([&] { (void)build.create_execution(engine, 1); }, TENSORACLE_INVALID_ARGUMENT);
        expect(allocations.at(address).borrowers == 16,
               "failed execution creation must preserve existing scratch owners");

        auto independent = build.open();
        auto independent_execution = build.create_execution(independent, 1);
        expect(allocations.size() == 2, "different engines must own separate scratch");
        engine.close();
        buckets.clear();
        expect(allocations.at(address).borrowers == 1,
               "execution must retain scratch after the engine handle closes");
        replacement.enqueue({}, nullptr);
        auto moved = std::move(replacement);
        moved.enqueue({}, nullptr);
        // Releasing from another current context must free in the owning one
        // and restore the caller's context.
        current = reinterpret_cast<CUcontext>(2);
        free_failure = true;
        rejects([&] { moved.close(); }, TENSORACLE_INTERNAL_ERROR);
        expect(allocations.count(address) && current == reinterpret_cast<CUcontext>(2),
               "failed last-execution close must retain scratch and restore the caller's context");
        moved.close();
        expect(!allocations.count(address) && current == reinterpret_cast<CUcontext>(2),
               "last execution must release scratch in its owning CUDA context");
        current = reinterpret_cast<CUcontext>(1);
        independent_execution.close();
        expect(allocations.size() == 1,
               "closing an execution must keep its engine's scratch reusable");
        free_failure = true;
        rejects([&] { independent.close(); }, TENSORACLE_INTERNAL_ERROR);
        expect(allocations.size() == 1, "failed engine close must retain its allocation for retry");
        independent.close();
        expect(allocations.empty() && !release_violation, "scratch destruction order is unsafe");

        required_bytes = 0;
        auto zero = build.open();
        auto zero_execution = build.create_execution(zero, 1);
        expect(allocations.size() == 1 && allocations.begin()->first &&
                   allocations.begin()->second.bytes >= 1,
               "zero scratch requirement still needs a non-null allocation");
        zero.close();
        zero_execution.enqueue({}, nullptr);
        zero_execution.close();

        required_bytes = 4096;
        allocation_failure = true;
        const auto before = opens;
        rejects([&] { (void)build.open(); }, TENSORACLE_OUT_OF_MEMORY);
        allocation_failure = false;
        expect(opens == before + 1 && allocations.empty(), "scratch OOM must not rebuild or leak");
        next_creation = TENSORACLE_INTERNAL_ERROR;
        const auto before_retry = opens;
        auto recovered = build.open();
        expect(opens == before_retry + 2 && allocations.size() == 1,
               "failed cached-plugin probe must release scratch before its one rebuild");
        recovered.close();
        expect(allocations.empty() && !release_violation && context_stack.empty(),
               "scratch cleanup leaked or released memory while borrowed");
        std::cout << "caller-owned scratch, sixteen buckets, probe, lifetimes and failures verified\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
