#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Interpose the external CUDA boundary. Context retention either fails or
// returns a test pointer; these tests cannot allocate or execute on a GPU.
static std::atomic<int> initialized{0};
static int contexts = 0, depth = 0, streams = 0, events = 0;
static int engines = 0, executions = 0, opened = 0, created = 0;
static int allocations = 0;
static bool fails(const char *name) {
    auto value = std::getenv("LFFW_TEST_FAILURE");
    return value && std::strcmp(value, name) == 0;
}
extern "C" int cuInit(unsigned int) { ++initialized; return 0; }
extern "C" int cuDeviceGetCount(int *count) { *count = 1; return 0; }
extern "C" int cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }
extern "C" int cuDeviceGetAttribute(int *out, int attribute, int) {
    constexpr int major_attribute = 75, minor_attribute = 76;
    int sm = std::getenv("LFFW_TEST_SM") ? std::atoi(std::getenv("LFFW_TEST_SM")) : 86;
    if (attribute == major_attribute) *out = sm / 10;
    else if (attribute == minor_attribute) *out = sm % 10;
    else return 1;
    return 0;
}
extern "C" int cuDevicePrimaryCtxRetain(void **out, int) {
    if (!std::getenv("LFFW_TEST_FAILURE")) return 999;
    *out = reinterpret_cast<void *>(1); ++contexts; opened = created = 0; return 0;
}
extern "C" int cuDevicePrimaryCtxRelease_v2(int) { --contexts; return 0; }
extern "C" int cuCtxGetCurrent(void **out) { *out = reinterpret_cast<void *>(1); return 0; }
extern "C" int cuCtxPushCurrent_v2(void *) { ++depth; return 0; }
extern "C" int cuCtxPopCurrent_v2(void **out) { *out = reinterpret_cast<void *>(1); --depth; return 0; }
extern "C" int cuStreamCreate(void **out, unsigned int) {
    if (fails("stream")) return 2;
    *out = reinterpret_cast<void *>(1); ++streams; return 0;
}
extern "C" int cuStreamDestroy_v2(void *) { --streams; return 0; }
extern "C" int cuEventCreate(void **out, unsigned int) {
    if (fails("event")) return 2;
    *out = reinterpret_cast<void *>(1); ++events; return 0;
}
extern "C" int cuEventDestroy_v2(void *) { --events; return 0; }
extern "C" int cuMemAlloc_v2(uint64_t *out, size_t bytes) {
    *out = reinterpret_cast<uint64_t>(std::calloc(1, bytes)); ++allocations; return 0;
}
extern "C" int cuMemFree_v2(uint64_t address) {
    std::free(reinterpret_cast<void *>(address)); --allocations; return 0;
}

// Only the external tensoracle C ABI is replaced. Our model, network wrapper,
// executor construction and cascading cleanup execute unchanged. These modes
// fail before inference and never call a TensorRT builder.
struct tensoracle_builder_options;
struct tensoracle_network_builder;
struct tensoracle_error;
struct tensoracle_engine {};
struct tensoracle_execution {};
struct tensoracle_scratch_memory { size_t struct_size; void *address; size_t size; };
extern "C" int tensoracle_open_engine(const tensoracle_builder_options *, const tensoracle_network_builder *,
                                      tensoracle_engine **out, tensoracle_error **error) noexcept {
    *out = nullptr; *error = nullptr; ++opened;
    if ((opened == 2 && fails("open-encoder")) || (opened == 3 && fails("open-decoder"))) return 3;
    *out = new tensoracle_engine; ++engines; return 0;
}
extern "C" int tensoracle_engine_path(const tensoracle_engine *, const char **out, tensoracle_error **error) noexcept {
    *out = "/test/engine-is-never-built"; *error = nullptr; return 0;
}
extern "C" int tensoracle_engine_destroy(tensoracle_engine *value) noexcept {
    if (value) { delete value; --engines; } return 0;
}
extern "C" int tensoracle_execution_create(const tensoracle_engine *, tensoracle_execution **out,
                                           tensoracle_error **error) noexcept {
    *out = nullptr; *error = nullptr; ++created;
    if (created == 2 && fails("execution")) return 3;
    *out = new tensoracle_execution; ++executions; return 0;
}
extern "C" int tensoracle_engine_scratch_size(const tensoracle_engine *, size_t *out,
                                             tensoracle_error **error) noexcept {
    *out = 256; *error = nullptr; return 0;
}
extern "C" int tensoracle_execution_create_with_scratch(const tensoracle_engine *engine,
    const tensoracle_scratch_memory *scratch, tensoracle_execution **out,
    tensoracle_error **error) noexcept {
    if (!scratch->address || scratch->size < 256) { *out = nullptr; *error = nullptr; return 1; }
    return tensoracle_execution_create(engine, out, error);
}
extern "C" int tensoracle_execution_destroy(tensoracle_execution *value) noexcept {
    if (value) { delete value; --executions; } return 0;
}
extern "C" int lffw_test_init_calls() { return initialized.load(); }
extern "C" int lffw_test_live_resources() { return contexts + depth + streams + events + engines + executions + allocations; }
