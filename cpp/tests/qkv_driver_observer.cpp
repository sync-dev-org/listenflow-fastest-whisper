// LD_PRELOAD observer for real CUDA driver calls made by the plugin under test.
#include <cuda.h>
#include <dlfcn.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>

namespace {
std::mutex functions_mutex;
std::set<CUfunction> qkv_functions;
std::set<CUfunction> fc1_functions;

template <typename Function>
Function next(char const* name)
{
    auto symbol = reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
    if (symbol == nullptr) {
        static auto driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (driver != nullptr) symbol = reinterpret_cast<Function>(dlsym(driver, name));
    }
    if (symbol == nullptr) std::abort();
    return symbol;
}

void record(void* caller, char const* event, std::uintptr_t pointer,
    std::size_t bytes, CUresult status)
{
    auto const* path = std::getenv("LFFW_QKV_DRIVER_LOG");
    Dl_info origin{};
    if (path == nullptr || dladdr(caller, &origin) == 0 || origin.dli_fname == nullptr ||
        (std::strstr(origin.dli_fname, "liblffw_cubin_plugin.so") == nullptr &&
         std::strstr(origin.dli_fname, "liblffw_fc1_plugin.so") == nullptr)) return;
    CUcontext context{};
    static auto get_context = next<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent");
    get_context(&context);
    auto* file = std::fopen(path, "a");
    if (file == nullptr) std::abort();
    std::fprintf(file, "%s|%" PRIuPTR "|%" PRIuPTR "|%zu|%d\n", event,
        reinterpret_cast<std::uintptr_t>(context), pointer, bytes, static_cast<int>(status));
    std::fclose(file);
}
}  // namespace

extern "C" CUresult CUDAAPI cuMemAlloc(CUdeviceptr* pointer, std::size_t bytes)
{
    static auto call = next<decltype(&cuMemAlloc)>("cuMemAlloc_v2");
    auto status = call(pointer, bytes);
    record(__builtin_return_address(0), "alloc", status == CUDA_SUCCESS ? *pointer : 0, bytes, status);
    return status;
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr pointer, void const* source, std::size_t bytes)
{
    static auto call = next<decltype(&cuMemcpyHtoD)>("cuMemcpyHtoD_v2");
    auto status = call(pointer, source, bytes);
    record(__builtin_return_address(0), "upload", pointer, bytes, status);
    return status;
}

extern "C" CUresult CUDAAPI cuMemFree(CUdeviceptr pointer)
{
    static auto call = next<decltype(&cuMemFree)>("cuMemFree_v2");
    auto status = call(pointer);
    record(__builtin_return_address(0), "free", pointer, 0, status);
    return status;
}

extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule* module, void const* image)
{
    static auto call = next<decltype(&cuModuleLoadData)>("cuModuleLoadData");
    auto status = call(module, image);
    unsigned int flags{};
    std::memcpy(&flags, static_cast<unsigned char const*>(image) + 48, sizeof(flags));
    auto const sm = ((flags >> 8) & 0xff) == 120 ? 120 : (flags >> 16) & 0xff;
    record(__builtin_return_address(0), "load",
        status == CUDA_SUCCESS ? reinterpret_cast<std::uintptr_t>(*module) : 0, sm, status);
    return status;
}

extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule module)
{
    static auto call = next<decltype(&cuModuleUnload)>("cuModuleUnload");
    auto status = call(module);
    record(__builtin_return_address(0), "unload", reinterpret_cast<std::uintptr_t>(module), 0, status);
    return status;
}

extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* function, CUmodule module, char const* name)
{
    static auto call = next<decltype(&cuModuleGetFunction)>("cuModuleGetFunction");
    auto status = call(function, module, name);
    if (status == CUDA_SUCCESS && std::strstr(name, "qkv_fp16") != nullptr) {
        std::lock_guard<std::mutex> guard(functions_mutex);
        qkv_functions.insert(*function);
    }
    if (status == CUDA_SUCCESS && std::strstr(name, "fc1_fp16") != nullptr) {
        std::lock_guard<std::mutex> guard(functions_mutex);
        fc1_functions.insert(*function);
    }
    return status;
}

extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction function,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int block_x, unsigned int block_y, unsigned int block_z,
    unsigned int shared, CUstream stream, void** parameters, void** extra)
{
    static auto call = next<decltype(&cuLaunchKernel)>("cuLaunchKernel");
    auto status = call(function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
        shared, stream, parameters, extra);
    bool qkv{}, fc1{};
    {
        std::lock_guard<std::mutex> guard(functions_mutex);
        qkv = qkv_functions.count(function) != 0;
        fc1 = fc1_functions.count(function) != 0;
    }
    if (qkv && parameters != nullptr) {
        char const* names[] = {"weight", "q_bias", "v_bias"};
        for (std::size_t index = 0; index < 3; ++index) {
            CUdeviceptr pointer{};
            std::memcpy(&pointer, parameters[index + 1], sizeof(pointer));
            record(__builtin_return_address(0), names[index], pointer, 0, status);
        }
    }
    if (fc1 && parameters != nullptr) {
        record(__builtin_return_address(0), "fc1_launch",
            reinterpret_cast<std::uintptr_t>(function), grid_x, status);
        char const* names[] = {"fc1_weight", "fc1_bias"};
        for (std::size_t index = 0; index < 2; ++index) {
            CUdeviceptr pointer{};
            std::memcpy(&pointer, parameters[index + 1], sizeof(pointer));
            record(__builtin_return_address(0), names[index], pointer, 0, status);
        }
    }
    return status;
}
