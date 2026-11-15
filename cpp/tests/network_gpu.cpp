#include "fc1_graph_contract.hpp"
#include "network_comparison.hpp"
#include "network_gpu_policy.hpp"
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <listenflow/whisper_network.hpp>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>

// Observe only the external tensoracle C ABI. The production library is linked
// unchanged; this test records callback invocation and cache error categories.
struct cache_observation {
    int opens = 0, builds = 0, corrupt = 0, condition = 0, io = 0;
    size_t scratch_bytes = 0;
};
static cache_observation observation;
extern "C" tensoracle_status tensoracle_engine_scratch_size(const tensoracle_engine *engine,
    size_t *bytes, tensoracle_error **error) noexcept {
    using function = tensoracle_status (*)(const tensoracle_engine *, size_t *, tensoracle_error **);
    static auto actual = reinterpret_cast<function>(dlsym(RTLD_NEXT, "tensoracle_engine_scratch_size"));
    if (!actual)
        return TENSORACLE_INTERNAL_ERROR;
    const auto status = actual(engine, bytes, error);
    if (status == TENSORACLE_SUCCESS)
        observation.scratch_bytes = *bytes;
    return status;
}
namespace ft = listenflow::fastest_whisper::test;
static bool expect_fp8 = false;
static int expected_plugins = 0;
static std::vector<std::string> expected_plugin_names;
static std::set<std::string> expected_linears;
static std::string graph_failure;
static void inspect_precision(nvinfer1::INetworkDefinition &network) {
    using namespace nvinfer1;
    std::map<std::string, int> operands;
    int dequantized = 0, plugins = 0;
    for (int i = 0; i < network.getNbLayers(); ++i) {
        auto *layer = network.getLayer(i);
        if (layer->getType() == LayerType::kPLUGIN_V3) {
            if (size_t(plugins) >= expected_plugin_names.size())
                throw std::runtime_error("unexpected plugin in the Whisper graph");
            auto layout = [](ITensor *tensor) -> ft::fc1_tensor_layout {
                if (!tensor || tensor->getDimensions().nbDims < 0)
                    throw std::runtime_error("fc1 tensor has no concrete shape");
                const auto shape = tensor->getDimensions();
                return {{shape.d, shape.d + shape.nbDims}, tensor->getType() == DataType::kHALF};
            };
            std::vector<ft::fc1_tensor_layout> inputs, outputs;
            for (int input = 0; input < layer->getNbInputs(); ++input)
                inputs.push_back(layout(layer->getInput(input)));
            for (int output = 0; output < layer->getNbOutputs(); ++output)
                outputs.push_back(layout(layer->getOutput(output)));
            ft::verify_fc1_layer(layer->getName(), expected_plugin_names[size_t(plugins)], inputs,
                                 outputs);
            ++plugins;
        }
        if (layer->getType() == LayerType::kDEQUANTIZE) {
            ++dequantized;
            if (layer->getOutput(0)->getType() != DataType::kHALF)
                throw std::runtime_error("FP8 dequantization must restore fp16 operands");
        }
        if (layer->getType() != LayerType::kQUANTIZE)
            continue;
        if (!expect_fp8)
            throw std::runtime_error("non-FP8 graph contains quantization");
        const std::string name = layer->getName();
        const bool weight = name.size() >= 9 && name.compare(name.size() - 9, 9, ".weight.Q") == 0;
        const bool input = name.size() >= 8 && name.compare(name.size() - 8, 8, ".input.Q") == 0;
        if (!weight && !input)
            throw std::runtime_error("unexpected FP8 quantization site");
        const auto key = name.substr(0, name.size() - (weight ? 9 : 8));
        if (!expected_linears.count(key))
            throw std::runtime_error("FP8 quantization on a non-linear operand");
        const auto flag = weight ? 1 : 2;
        if (operands[key] & flag)
            throw std::runtime_error("duplicate FP8 quantization operand");
        operands[key] |= flag;
        auto *quantize = static_cast<IQuantizeLayer *>(layer);
        auto *scale = layer->getInput(1);
        if (layer->getOutput(0)->getType() != DataType::kFP8 ||
            scale->getType() != DataType::kFLOAT || quantize->getAxis() != (weight ? 2 : 0))
            throw std::runtime_error("FP8 operand dtype, scale dtype or axis differs");
        const auto dims = scale->getDimensions();
        if ((weight &&
             (dims.nbDims != 1 || dims.d[0] != layer->getInput(0)->getDimensions().d[2])) ||
            (input && dims.nbDims != 0))
            throw std::runtime_error(
                "FP8 weight scales must be per output and activation scales scalar");
    }
    if (plugins != expected_plugins)
        throw std::runtime_error("required fp16 fc1 plugin count differs");
    if (expect_fp8) {
        if (operands.size() != expected_linears.size() ||
            dequantized != 2 * int(expected_linears.size()))
            throw std::runtime_error("FP8 must quantize every linear weight and activation");
        for (const auto &item : operands)
            if (item.second != 3)
                throw std::runtime_error("FP8 linear operand is unquantized");
    } else if (dequantized)
        throw std::runtime_error("non-FP8 graph contains dequantization");
    std::cout << "graph precision verified: " << operands.size() << " FP8 linears, " << plugins
              << " fc1 plugins" << std::endl;
    std::cout << "fc1 input/output layouts verified: " << plugins << std::endl;
}

extern "C" tensoracle_status tensoracle_open_engine(const tensoracle_builder_options *options,
                                                    const tensoracle_network_builder *builder,
                                                    tensoracle_engine **output,
                                                    tensoracle_error **error) noexcept {
    using function = tensoracle_status (*)(const tensoracle_builder_options *,
                                           const tensoracle_network_builder *, tensoracle_engine **,
                                           tensoracle_error **);
    static auto actual = reinterpret_cast<function>(dlsym(RTLD_NEXT, "tensoracle_open_engine"));
    if (!actual)
        return TENSORACLE_INTERNAL_ERROR;
    ++observation.opens;
    tensoracle_network_builder wrapped{
        sizeof(wrapped), const_cast<tensoracle_network_builder *>(builder),
        [](void *context, void *network, const tensoracle_weights *weights,
           const char **message) noexcept {
            ++observation.builds;
            auto *original = static_cast<const tensoracle_network_builder *>(context);
            auto status = original->build(original->context, network, weights, message);
            if (status != TENSORACLE_SUCCESS)
                return status;
            try {
                inspect_precision(*static_cast<nvinfer1::INetworkDefinition *>(network));
            } catch (const std::exception &error) {
                graph_failure = error.what();
                *message = graph_failure.c_str();
                return tensoracle_status(TENSORACLE_INVALID_ARGUMENT);
            }
            return tensoracle_status(TENSORACLE_SUCCESS);
        }};
    const auto status = actual(options, &wrapped, output, error);
    if (status == TENSORACLE_CORRUPT_ENGINE)
        ++observation.corrupt;
    if (status == TENSORACLE_ENGINE_CONDITION_MISMATCH)
        ++observation.condition;
    if (status == TENSORACLE_ENGINE_IO_MISMATCH)
        ++observation.io;
    return status;
}
namespace fw = listenflow::fastest_whisper;
namespace fs = std::filesystem;
static void check(CUresult s) {
    if (s != CUDA_SUCCESS) {
        const char *text = nullptr;
        cuGetErrorString(s, &text);
        throw std::runtime_error(text ? text : "CUDA failure");
    }
}
struct context {
    CUdevice device{};
    CUcontext value{};
    context() {
        check(cuInit(0));
        check(cuDeviceGet(&device, 0));
        check(cuDevicePrimaryCtxRetain(&value, device));
        check(cuCtxPushCurrent(value));
    }
    ~context() {
        CUcontext previous{};
        cuCtxPopCurrent(&previous);
        cuDevicePrimaryCtxRelease(device);
    }
};
struct buffer {
    CUdeviceptr value{};
    explicit buffer(size_t size) { check(cuMemAlloc(&value, size)); }
    ~buffer() {
        if (value)
            cuMemFree(value);
    }
};
struct stream {
    CUstream value{};
    stream() { check(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~stream() {
        if (value)
            cuStreamDestroy(value);
    }
};
static fw::model_source source(const fs::path &root) {
    std::ifstream input(root / "source.txt");
    fw::model_source s;
    auto &c = s.config;
    input >> c.n_mels >> c.audio_positions >> c.audio_state >> c.audio_heads >> c.audio_ffn >>
        c.audio_layers >> c.text_heads >> c.text_ffn >> c.text_layers >> c.text_positions >>
        c.vocab_size;
    input >> std::quoted(s.config_file) >> std::quoted(s.tokenizer_file) >>
        std::quoted(s.scale_file) >> std::quoted(s.mel_filter_file) >> std::quoted(s.mel_weights) >>
        std::quoted(s.fc1_library);
    int32_t no_speech, timestamp;
    input >> no_speech >> timestamp;
    if (no_speech >= 0)
        s.no_speech = no_speech;
    if (timestamp >= 0)
        s.timestamp_begin = timestamp;
    size_t count;
    input >> count;
    for (size_t i = 0; i < count; ++i) {
        std::string p;
        input >> std::quoted(p);
        s.weights.push_back(p);
    }
    input >> count;
    for (size_t i = 0; i < count; ++i) {
        std::string key;
        float value;
        input >> std::quoted(key) >> value;
        s.scales.emplace(key, value);
    }
    if (!input)
        throw std::runtime_error("invalid native network fixture source");
    return s;
}
static std::pair<size_t, std::string> dtype(fw::tensor_dtype type) {
    switch (type) {
    case fw::tensor_dtype::fp32:
        return {4, "float32"};
    case fw::tensor_dtype::fp16:
        return {2, "float16"};
    case fw::tensor_dtype::int32:
        return {4, "int32"};
    case fw::tensor_dtype::boolean:
        return {1, "bool"};
    default:
        throw std::runtime_error("unexpected network dtype");
    }
}
static tensoracle_dtype backend_type(fw::tensor_dtype type) {
    switch (type) {
    case fw::tensor_dtype::fp32:
        return TENSORACLE_FLOAT32;
    case fw::tensor_dtype::fp16:
        return TENSORACLE_FLOAT16;
    case fw::tensor_dtype::int32:
        return TENSORACLE_INT32;
    case fw::tensor_dtype::boolean:
        return TENSORACLE_BOOL;
    }
    throw std::runtime_error("unknown model dtype");
}
int main(int argc, char **argv) {
    try {
        const char *root_env = std::getenv("LFFW_NETWORK_FIXTURE_ROOT");
        if (!root_env) {
            std::cerr << "set LFFW_NETWORK_FIXTURE_ROOT for native GPU comparisons\n";
            return 77;
        }
        const fs::path root = root_env;
        const std::string name = argc > 1 ? argv[1] : "mel", p = argc > 2 ? argv[2] : "fp32";
        const auto kind = name == "mel"       ? fw::graph_kind::mel
                          : name == "encoder" ? fw::graph_kind::encoder
                          : name == "decoder" ? fw::graph_kind::decoder
                                              : fw::graph_kind::sampled_decoder;
        const auto precision = p == "fp32"   ? fw::precision::fp32
                               : p == "fp16" ? fw::precision::fp16
                                             : fw::precision::fp8;
        const int batch = argc > 3 ? std::stoi(argv[3]) : 5,
                  history = argc > 4 ? std::stoi(argv[4]) : 1;
        const std::string mode = argc > 5 ? argv[5] : "run";
        context ctx;
        int major = 0, minor = 0;
        if (precision == fw::precision::fp8) {
            check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                       ctx.device));
            check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                       ctx.device));
        }
        const auto policy = fw::test::gpu_test_policy(kind, precision, major, minor);
        expect_fp8 = policy.inspect_fp8;
        if (policy.skip) {
            std::cerr << "FP8 network tests require sm89 or sm120; selected GPU is sm" << major
                      << minor << '\n';
            return 77;
        }
        if (precision == fw::precision::fp8)
            std::cout << "FP8 GPU test runs on sm" << major << minor << std::endl;
        const auto suffix =
            name + "-" + p + "-b" + std::to_string(batch) + "-h" + std::to_string(history);
        const auto output = root / (suffix + "-native");
        fs::create_directories(output);
        auto s = source(root);
        if (precision == fw::precision::fp16 &&
            (kind == fw::graph_kind::decoder || kind == fw::graph_kind::sampled_decoder)) {
            const auto candidates = kind == fw::graph_kind::sampled_decoder ? batch : 1;
            for (int layer = 0; layer < s.config.text_layers; ++layer)
                for (int candidate = 0; candidate < candidates; ++candidate)
                    expected_plugin_names.push_back(
                        "model.decoder.layers." + std::to_string(layer) + ".fc1.gelu" +
                        (candidates > 1 ? "." + std::to_string(candidate) : ""));
        }
        expected_plugins = int(expected_plugin_names.size());
        if (expect_fp8) {
            auto attention = [&](const std::string &base,
                                 const std::vector<std::string> &projections) {
                for (const auto &part : projections)
                    expected_linears.insert(base + "." + part + "_proj");
            };
            if (kind == fw::graph_kind::encoder) {
                for (int i = 0; i < s.config.audio_layers; ++i) {
                    auto base = "model.encoder.layers." + std::to_string(i);
                    attention(base + ".self_attn", {"q", "k", "v", "out"});
                    expected_linears.insert(base + ".fc1");
                    expected_linears.insert(base + ".fc2");
                }
                for (int i = 0; i < s.config.text_layers; ++i)
                    attention("model.decoder.layers." + std::to_string(i) + ".encoder_attn",
                              {"k", "v"});
            } else
                for (int i = 0; i < s.config.text_layers; ++i) {
                    auto base = "model.decoder.layers." + std::to_string(i);
                    attention(base + ".self_attn", {"q", "k", "v", "out"});
                    attention(base + ".encoder_attn", {"q", "out"});
                    expected_linears.insert(base + ".fc1");
                    expected_linears.insert(base + ".fc2");
                }
        }
        fw::engine_build build(s, kind, precision, batch);
        const bool measure_time = std::getenv("LFFW_NETWORK_MEASURE_TIME") != nullptr;
        std::chrono::steady_clock::time_point started;
        if (measure_time)
            started = std::chrono::steady_clock::now();
        auto handle = build.open();
        std::ofstream(output / "engine-path.txt") << handle.path();
        if (measure_time)
            std::cout
                << "open_seconds "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << ' ';
        std::cout << "engine_bytes " << fs::file_size(handle.path()) << " path " << handle.path()
                  << std::endl;
        if (mode == "reuse" && observation.builds != 0)
            throw std::runtime_error("fresh process rebuilt an existing fixture engine");
        if (mode == "memory") {
            size_t free, total;
            check(cuMemGetInfo(&free, &total));
            const auto before = free;
            auto executions = build.create_sampled_executions(handle);
            check(cuMemGetInfo(&free, &total));
            std::cout << "sample_batch " << batch << " executions " << executions.size()
                      << " longest_history " << executions.back().history
                      << " scratch_size_bytes " << observation.scratch_bytes
                      << " bucket_execution_delta_bytes " << (int64_t(before) - int64_t(free))
                      << std::endl;
            return 0;
        }
        if (mode == "corrupt") {
            auto path = handle.path();
            handle = {};
            // This fixture's auxiliary identity isolates its default-cache file.
            std::ofstream(path, std::ios::binary | std::ios::trunc) << "broken fixture engine";
            handle = build.open();
            std::cout << "recovered_corrupt_engine " << handle.path() << std::endl;
        }
        if (mode == "condition") {
            auto path = handle.path();
            handle = {};
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            std::string bytes(65536, '\0');
            file.read(bytes.data(), bytes.size());
            bytes.resize(file.gcount());
            auto version = build.declaration().graph_version;
            auto pos = bytes.find(version);
            if (pos == std::string::npos)
                throw std::runtime_error("fixture engine header has no graph version");
            file.clear();
            file.seekp(pos);
            file.put(version[0] == '0' ? '1' : '0');
            file.close();
            handle = build.open();
            std::cout << "recovered_condition_engine " << handle.path() << std::endl;
        }
        if (mode == "corrupt" && (observation.builds != 1 || observation.corrupt != 1))
            throw std::runtime_error("corrupt engine was not rebuilt exactly once");
        if (mode == "condition" && (observation.builds != 1 || observation.condition != 1))
            throw std::runtime_error("condition mismatch was not rebuilt exactly once");
        std::cout << "cache_opens " << observation.opens << " network_callbacks "
                  << observation.builds << " corrupt_statuses " << observation.corrupt
                  << " condition_statuses " << observation.condition << std::endl;
        auto execution = build.create_execution(
            handle, kind == fw::graph_kind::decoder || kind == fw::graph_kind::sampled_decoder
                        ? std::optional<int64_t>(history)
                        : std::nullopt);
        if (kind == fw::graph_kind::decoder || kind == fw::graph_kind::sampled_decoder) {
            bool rejected = false;
            try {
                execution.set_input_shapes({});
            } catch (const fw::model_error &error) {
                if (error.status() != TENSORACLE_INVALID_ARGUMENT)
                    throw;
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("incomplete shapes did not preserve backend status 1");
            std::cout << "invalid shapes preserve model_error status 1" << std::endl;
        }
        auto inputs = tensoracle::open_weights(root / (suffix + "-inputs.safetensors"));
        std::vector<std::unique_ptr<buffer>> buffers;
        std::vector<fw::tensor_address> addresses;
        std::map<std::string, std::vector<char>> results;
        std::map<std::string, std::vector<int64_t>> shapes;
        std::map<std::string, std::string> types;
        stream queue;
        for (auto &spec : build.declaration().io) {
            auto shape = execution.get_tensor_shape(spec.name.c_str());
            auto type = dtype(execution.get_tensor_dtype(spec.name.c_str()));
            size_t count = 1;
            for (auto d : shape)
                count *= size_t(d);
            const auto bytes = count * type.first;
            buffers.push_back(std::make_unique<buffer>(bytes));
            auto ptr = buffers.back()->value;
            addresses.push_back({spec.name, reinterpret_cast<void *>(ptr)});
            if (spec.input) {
                auto v = inputs.view().at(spec.name.c_str());
                if (v.byte_size != bytes || v.dtype != backend_type(spec.dtype) ||
                    v.rank != shape.size() || !std::equal(shape.begin(), shape.end(), v.dimensions))
                    throw std::runtime_error("fixture input layout differs: " + spec.name);
                check(cuMemcpyHtoDAsync(ptr, v.data, bytes, queue.value));
            } else {
                results[spec.name].resize(bytes);
                shapes[spec.name] = std::move(shape);
                types[spec.name] = type.second;
            }
        }
        execution.enqueue(addresses, queue.value);
        for (size_t i = 0; i < addresses.size(); ++i) {
            auto found = results.find(addresses[i].name);
            if (found != results.end())
                check(cuMemcpyDtoHAsync(found->second.data(), buffers[i]->value,
                                        found->second.size(), queue.value));
        }
        check(cuStreamSynchronize(queue.value));
        std::ofstream metadata(output / "outputs.json");
        metadata << '{';
        bool first = true;
        for (auto &result : results) {
            std::ofstream(output / (result.first + ".bin"), std::ios::binary)
                .write(result.second.data(), result.second.size());
            if (!first)
                metadata << ',';
            first = false;
            metadata << std::quoted(result.first)
                     << ":{\"dtype\":" << std::quoted(types[result.first]) << ",\"shape\":[";
            bool axis = false;
            for (auto d : shapes[result.first]) {
                if (axis)
                    metadata << ',';
                axis = true;
                metadata << d;
            }
            metadata << "]}";
        }
        metadata << '}';
        metadata.close();
        auto reference = tensoracle::open_weights(root / (suffix + "-reference.safetensors"));
        for (const auto &result : results) {
            auto expected = reference.view().at(result.first.c_str());
            if (expected.dtype != backend_type(execution.get_tensor_dtype(result.first.c_str())))
                throw std::runtime_error("reference dtype differs: " + result.first);
            compare_network_output(result.first, result.second, shapes.at(result.first), expected,
                                   precision == fw::precision::fp32 || kind == fw::graph_kind::mel);
        }
        if (precision == fw::precision::fp32 || kind == fw::graph_kind::mel)
            std::cout << "all native outputs match the 1.x reference" << std::endl;
        else
            std::cout << "I/O, graph precision and execution verified; tensor values are diagnostic"
                      << std::endl;

        std::cout << "outputs " << output << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
