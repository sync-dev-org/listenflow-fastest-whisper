// GPU checks use an independent double/IEEE-half reference with the existing tolerances.
#include "kernel_reference.hpp"
#include <tensoracle/builder.hpp>
#include <NvInfer.h>
#include <cuda.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;
using namespace lffw_test;
namespace {
void require(bool value, std::string const& why) { if (!value) throw std::runtime_error(why); }
void cuda_check(CUresult result)
{
    if (result != CUDA_SUCCESS) {
        char const* message = nullptr;
        cuGetErrorString(result, &message);
        throw std::runtime_error(std::string("CUDA: ") + (message ? message : "unknown"));
    }
}
struct Context {
    CUdevice device{};
    CUcontext value{};
    Context() {
        cuda_check(cuInit(0)); cuda_check(cuDeviceGet(&device, 0));
        int major{}, minor{};
        cuda_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        cuda_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        require(major * 10 + minor == 120, "numerical and speed measurements require sm120");
        cuda_check(cuDevicePrimaryCtxRetain(&value, device));
        cuda_check(cuCtxPushCurrent(value));
    }
    ~Context() { CUcontext old{}; cuCtxPopCurrent(&old); cuDevicePrimaryCtxRelease(device); }
};
struct Buffer {
    CUdeviceptr pointer{};
    explicit Buffer(std::size_t bytes) { cuda_check(cuMemAlloc(&pointer, bytes)); }
    ~Buffer() { cuMemFree(pointer); }
    void upload(std::vector<std::uint16_t> const& values) {
        cuda_check(cuMemcpyHtoD(pointer, values.data(), values.size() * 2));
    }
};
struct Stream {
    CUstream value{};
    Stream() { cuda_check(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~Stream() { cuStreamSynchronize(value); cuStreamDestroy(value); }
};
Dims dims(std::initializer_list<std::int64_t> values)
{
    Dims result{}; result.nbDims = values.size();
    std::copy(values.begin(), values.end(), result.d); return result;
}
PluginTensorDesc desc(Dims shape)
{
    PluginTensorDesc result{}; result.dims = shape;
    result.type = DataType::kHALF; result.format = TensorFormat::kLINEAR; return result;
}
struct Plugin {
    void* library{};
    std::unique_ptr<IPluginV3> value;
    explicit Plugin(char const* path, char const* name) {
        library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        auto get = reinterpret_cast<IPluginCreatorInterface* const* (*)(std::int32_t&)>(dlsym(library, "getCreators"));
        auto* digest = static_cast<char const*>(dlsym(library, "tensoracle_bundle_sha256"));
        require(get && digest, "missing creator or bundle export");
        std::int32_t count{}; auto creators = get(count);
        std::uint16_t bias = half_bits(-.375);
        std::vector<PluginField> fields{{"bundle_sha256", digest, PluginFieldType::kCHAR, 64}};
        if (std::string(name) == "LffwFp16Bias") fields.emplace_back("bias", &bias, PluginFieldType::kFLOAT16, 1);
        PluginFieldCollection collection{static_cast<int>(fields.size()), fields.data()};
        for (int i = 0; i < count; ++i) {
            auto* creator = dynamic_cast<IPluginCreatorV3One*>(creators[i]);
            if (creator && std::string(creator->getPluginName()) == name)
                value.reset(creator->createPlugin("test", &collection, TensorRTPhase::kBUILD));
        }
        require(bool(value), "requested plugin creator rejected its valid fields");
    }
    ~Plugin() { value.reset(); if (library) dlclose(library); }
    IPluginV3OneRuntime* runtime() {
        return static_cast<IPluginV3OneRuntime*>(value->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    }
};
void compare(std::vector<std::uint16_t> const& actual, std::vector<double> const& expected,
    double absolute, bool exact)
{
    require(actual.size() == expected.size(), "output size differs");
    double maximum = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        double value = half_value(actual[i]);
        double error = std::abs(value - expected[i]); maximum = std::max(maximum, error);
        require(std::isfinite(value) && (exact ? value == expected[i]
            : error <= absolute + std::ldexp(1.0, -10) * std::abs(expected[i])),
            "numerical mismatch at " + std::to_string(i) + ": actual=" + std::to_string(value) +
                " expected=" + std::to_string(expected[i]));
    }
    std::cout << "max_abs_error=" << maximum << (exact ? " exact" : "") << '\n';
}
void bias_test(char const* library)
{
    Plugin plugin(library, "LffwFp16Bias");
    Stream stream;
    for (int count : {1, 127, 128, 129, 4097}) {
        std::vector<std::uint16_t> x(count), output(count + 128, half_bits(-123));
        std::vector<double> expected(count);
        for (int i = 0; i < count; ++i) {
            x[i] = half_bits(std::sin(i * .17) * 5);
            expected[i] = round_half(half_value(x[i]) - .375);
        }
        Buffer source(x.size() * 2), target(output.size() * 2);
        source.upload(x); target.upload(output);
        auto input = desc(dims({count})), result = input;
        void const* inputs[] = {reinterpret_cast<void*>(source.pointer)};
        void* outputs[] = {reinterpret_cast<void*>(target.pointer)};
        require(plugin.runtime()->enqueue(&input, &result, inputs, outputs, nullptr,
            reinterpret_cast<cudaStream_t>(stream.value)) == 0, "bias enqueue failed");
        cuda_check(cuStreamSynchronize(stream.value));
        cuda_check(cuMemcpyDtoH(output.data(), target.pointer, output.size() * 2));
        for (int i = count; i < count + 128; ++i) require(output[i] == half_bits(-123), "partial block overwrote output guard");
        output.resize(count); compare(output, expected, 0, true);
    }
}
void projection_test(char const* library, bool fc1, char const* real_weights = nullptr)
{
    Plugin plugin(library, fc1 ? "LffwFp16Fc1" : "LffwFp16Qkv");
    Projection data(fc1);
    if (real_weights) {
        auto source = tensoracle::open_weights(real_weights);
        auto weight = source.view().at("weight"), bias = source.view().at("bias");
        require(fc1 && weight.dtype == TENSORACLE_FLOAT16 && weight.rank == 2 &&
            weight.dimensions[0] == 5120 && weight.dimensions[1] == 1280 &&
            bias.dtype == TENSORACLE_FLOAT16 && bias.rank == 1 && bias.dimensions[0] == 5120,
            "real fc1 test requires half weight (5120,1280) and bias (5120)");
        std::memcpy(data.weight.data(), weight.data, data.weight.size() * 2);
        std::memcpy(data.bias.data(), bias.data, data.bias.size() * 2);
    }
    Buffer x(Projection::width * 2), w(data.weight.size() * 2), b(data.bias.size() * 2),
        vb(std::max<std::size_t>(1, data.v_bias.size()) * 2), y((data.rows + 128) * 2);
    Stream stream;
    auto infer = [&](bool exact) {
        auto expected = data.reference(fc1);
        x.upload(data.x); w.upload(data.weight); b.upload(data.bias);
        if (!fc1) vb.upload(data.v_bias);
        std::vector<std::uint16_t> actual(data.rows + 128, half_bits(-123)); y.upload(actual);
        std::vector<PluginTensorDesc> descriptors{desc(dims({1, 1, Projection::width})),
            desc(dims({data.rows, Projection::width})), desc(dims({static_cast<int>(data.bias.size())}))};
        if (!fc1) descriptors.push_back(desc(dims({Projection::width})));
        auto result = desc(dims({1, 1, data.rows}));
        std::vector<void const*> inputs{reinterpret_cast<void*>(x.pointer), reinterpret_cast<void*>(w.pointer),
            reinterpret_cast<void*>(b.pointer)};
        if (!fc1) inputs.push_back(reinterpret_cast<void*>(vb.pointer));
        void* outputs[] = {reinterpret_cast<void*>(y.pointer)};
        require(plugin.runtime()->enqueue(descriptors.data(), &result, inputs.data(), outputs, nullptr,
            reinterpret_cast<cudaStream_t>(stream.value)) == 0, "projection enqueue failed");
        cuda_check(cuStreamSynchronize(stream.value));
        cuda_check(cuMemcpyDtoH(actual.data(), y.pointer, actual.size() * 2));
        for (int i = data.rows; i < data.rows + 128; ++i) require(actual[i] == half_bits(-123), "projection overwrote output guard");
        actual.resize(data.rows); compare(actual, expected, std::ldexp(1.0, fc1 ? -11 : -12), exact);
    };
    infer(false);
    for (auto& v : data.x) v = half_bits(-half_value(v));
    infer(false);
    std::fill(data.x.begin(), data.x.end(), 0);
    infer(true); // Bias alone, including the Q/K/V order and unbiased K.
    data.x[731] = half_bits(1); infer(!fc1);
    if (real_weights) {
        double norm = 0;
        for (int column = 0; column < Projection::width; ++column) {
            double v = half_value(data.weight[17 * Projection::width + column]); norm += v * v;
        }
        require(norm > 0, "checkpoint row used for directed cancellation is zero");
        for (double target : {0.0, -.75, .001}) {
            for (int column = 0; column < Projection::width; ++column)
                data.x[column] = half_bits(half_value(data.weight[17 * Projection::width + column]) *
                    (target - half_value(data.bias[17])) / norm);
            infer(false);
        }
    }
    if (fc1) {
        // Existing directed cancellation sentinels: a single final half round is incorrect.
        double sentinels[] = {.0001, .0002, .0003, .0004, .0005, .0006, .001, .0015};
        std::fill(data.x.begin(), data.x.end(), 0); data.x[0] = data.x[1] = half_bits(1);
        std::fill(data.weight.begin(), data.weight.end(), 0);
        std::fill(data.bias.begin(), data.bias.end(), half_bits(-1));
        for (int row = 0; row < data.rows; ++row) {
            data.weight[row * Projection::width] = half_bits(1);
            data.weight[row * Projection::width + 1] = half_bits(sentinels[row % 8]);
        }
        infer(true);
    }
}
struct Module {
    CUmodule module{}; CUfunction function{};
    Module(char const* path) {
        cuda_check(cuModuleLoad(&module, path));
        cuda_check(cuModuleGetFunction(&function, module, "fc1_fp16"));
    }
    ~Module() { cuModuleUnload(module); }
};
void benchmark(char const* old_path, char const* new_path)
{
    Projection data(true);
    Buffer x(data.x.size() * 2), w(data.weight.size() * 2), b(data.bias.size() * 2), y(data.rows * 2);
    x.upload(data.x); w.upload(data.weight); b.upload(data.bias);
    Module old_kernel(old_path), new_kernel(new_path);
    Stream stream;
    CUdeviceptr scratch{};
    void* arguments[] = {&x.pointer, &w.pointer, &b.pointer, &y.pointer, &scratch, &scratch};
    auto launch = [&](bool old) {
        cuda_check(cuLaunchKernel(old ? old_kernel.function : new_kernel.function,
            1280, 1, 1, 128, 1, 1, old ? 64 : 0, stream.value, arguments, nullptr));
    };
    auto expected = data.reference(true);
    for (bool old : {true, false}) {
        launch(old); cuda_check(cuStreamSynchronize(stream.value));
        std::vector<std::uint16_t> output(data.rows);
        cuda_check(cuMemcpyDtoH(output.data(), y.pointer, output.size() * 2));
        compare(output, expected, std::ldexp(1.0, -11), false);
        auto function = old ? old_kernel.function : new_kernel.function;
        int registers{}, local{}, shared{};
        cuda_check(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function));
        cuda_check(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function));
        cuda_check(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
        std::cout << (old ? "old" : "new") << " registers=" << registers << " local_bytes=" << local
            << " static_shared_bytes=" << shared << " dynamic_shared_bytes=" << (old ? 64 : 0) << '\n';
    }
    for (int i = 0; i < 100; ++i) { launch(true); launch(false); }
    cuda_check(cuStreamSynchronize(stream.value));
    CUevent start{}, stop{};
    cuda_check(cuEventCreate(&start, CU_EVENT_DEFAULT)); cuda_check(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    std::array<std::vector<double>, 2> times;
    for (int trial = 0; trial < 15; ++trial) {
        for (int order = 0; order < 2; ++order) {
            bool old = (trial + order) % 2 == 0;
            cuda_check(cuEventRecord(start, stream.value));
            for (int i = 0; i < 200; ++i) launch(old);
            cuda_check(cuEventRecord(stop, stream.value)); cuda_check(cuEventSynchronize(stop));
            float milliseconds{}; cuda_check(cuEventElapsedTime(&milliseconds, start, stop));
            times[old ? 0 : 1].push_back(milliseconds * 1000 / 200);
        }
    }
    cuEventDestroy(start); cuEventDestroy(stop);
    for (auto& time : times) std::sort(time.begin(), time.end());
    std::cout << "grid=1280 block=128 trials=15 launches_per_trial=200 old_us=" << times[0][7]
        << " new_us=" << times[1][7] << " new_over_old=" << times[1][7] / times[0][7] << '\n';
}
} // namespace
int main(int argc, char** argv)
{
    try {
        require(argc >= 3, "usage: kernel_gpu bias|qkv|fc1 PLUGIN or benchmark OLD_CUBIN NEW_CUBIN");
        std::string mode = argv[1];
        require(mode == "bias" || mode == "qkv" || mode == "fc1" || mode == "benchmark", "unknown kernel test mode");
        require(std::filesystem::is_regular_file(argv[2]), "input file does not exist");
        if (mode == "benchmark") require(argc == 4 && std::filesystem::is_regular_file(argv[3]),
            "benchmark requires two existing cubin files");
        Context context;
        if (mode == "bias") bias_test(argv[2]);
        else if (mode == "qkv") projection_test(argv[2], false);
        else if (mode == "fc1") projection_test(argv[2], true, argc == 4 ? argv[3] : nullptr);
        else if (mode == "benchmark" && argc == 4) benchmark(argv[2], argv[3]);
        else throw std::invalid_argument("unknown kernel test mode");
        return 0;
    } catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
