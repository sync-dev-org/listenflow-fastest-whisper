// Native replacement for the caller-owned plugin graph fixtures. No Python API is used.
#include "kernel_reference.hpp"
#include <tensoracle/builder.hpp>
#include <tensoracle/cubin.hpp>
#include <tensoracle/plugins.hpp>
#include <NvInfer.h>
#include <cuda.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace nvinfer1;
using namespace lffw_test;
namespace {
void require(bool condition, std::string const& message)
{
    if (!condition) throw std::runtime_error(message);
}
void cuda_check(CUresult status)
{
    if (status != CUDA_SUCCESS) throw std::runtime_error("CUDA status " + std::to_string(status));
}
struct Context {
    CUdevice device{}; CUcontext value{};
    Context() {
        cuda_check(cuInit(0)); cuda_check(cuDeviceGet(&device, 0));
        int major{}, minor{};
        cuda_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        cuda_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        require(major * 10 + minor == 120, "plugin integration measurements require sm120");
        cuda_check(cuDevicePrimaryCtxRetain(&value, device)); cuda_check(cuCtxPushCurrent(value));
    }
    ~Context() { CUcontext old{}; cuCtxPopCurrent(&old); cuDevicePrimaryCtxRelease(device); }
};
struct Buffer {
    CUdeviceptr value{};
    explicit Buffer(std::size_t bytes) { cuda_check(cuMemAlloc(&value, bytes)); }
    ~Buffer() { cuMemFree(value); }
};
struct Stream {
    CUstream value{};
    Stream() { cuda_check(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~Stream() { cuStreamSynchronize(value); cuStreamDestroy(value); }
};
struct Logger : ILogger {
    std::string errors;
    void log(Severity severity, char const* message) noexcept override {
        if (severity <= Severity::kERROR) {
            try { errors += std::string(message) + '\n'; } catch (...) {}
        }
    }
};
Dims dims(std::vector<int64_t> const& values)
{
    Dims result{}; result.nbDims = values.size();
    std::copy(values.begin(), values.end(), result.d); return result;
}
template<class T> void write(std::ostream& stream, T const& value)
{
    stream.write(reinterpret_cast<char const*>(&value), sizeof(value));
    require(bool(stream), "cannot write fixture");
}
template<class T> T read(std::istream& stream)
{
    T value{}; stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    require(bool(stream), "truncated fixture"); return value;
}
std::vector<char> bytes(fs::path const& path)
{
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "cannot open " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}
void save(fs::path const& path, std::vector<char> const& contents)
{
    std::ofstream output(path, std::ios::binary);
    output.write(contents.data(), contents.size()); require(bool(output), "cannot write " + path.string());
}
struct Case { std::vector<uint16_t> input, expected; bool exact{}; };
void save_cases(fs::path const& path, std::vector<Case> const& cases)
{
    std::ofstream output(path, std::ios::binary);
    write(output, uint32_t(cases.size()));
    for (auto const& item : cases) {
        write(output, uint32_t(item.input.size())); write(output, uint32_t(item.expected.size()));
        write(output, uint32_t(item.exact));
        for (auto v : item.input) write(output, v);
        for (auto v : item.expected) write(output, v);
    }
}
std::vector<Case> load_cases(fs::path const& path)
{
    std::ifstream input(path, std::ios::binary);
    auto count = read<uint32_t>(input); require(count <= 16, "invalid case count");
    std::vector<Case> cases(count);
    for (auto& item : cases) {
        auto inputs = read<uint32_t>(input), outputs = read<uint32_t>(input);
        require(inputs <= 4097 && outputs <= 5120, "invalid case shape");
        item.exact = read<uint32_t>(input) != 0;
        item.input.resize(inputs); item.expected.resize(outputs);
        for (auto& v : item.input) v = read<uint16_t>(input);
        for (auto& v : item.expected) v = read<uint16_t>(input);
    }
    require(input.peek() == std::char_traits<char>::eof(), "trailing fixture data"); return cases;
}
void save_weights(fs::path const& path, std::string const& kind, Projection const& data)
{
    struct Tensor { std::string name, shape; std::vector<uint16_t> const* values; };
    std::vector<uint16_t> scalar{half_bits(-.375)};
    std::vector<Tensor> tensors;
    if (kind == "bias") tensors = {{"bias", "[1]", &scalar}};
    else {
        tensors = {{kind == "qkv" ? "qkv_weight" : "weight", "[" + std::to_string(data.rows) + ",1280]", &data.weight},
            {kind == "qkv" ? "q_bias" : "bias", "[" + std::to_string(data.bias.size()) + "]", &data.bias}};
        if (kind == "qkv") tensors.push_back({"v_bias", "[1280]", &data.v_bias});
    }
    std::ostringstream json; json << '{'; std::size_t offset = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        auto const& tensor = tensors[i]; auto next = offset + tensor.values->size() * 2;
        if (i) json << ',';
        json << '"' << tensor.name << "\":{\"dtype\":\"F16\",\"shape\":" << tensor.shape
            << ",\"data_offsets\":[" << offset << ',' << next << "]}";
        offset = next;
    }
    json << '}'; auto header = json.str(); while (header.size() % 8) header += ' ';
    std::ofstream output(path, std::ios::binary); write(output, uint64_t(header.size()));
    output.write(header.data(), header.size());
    for (auto const& tensor : tensors)
        output.write(reinterpret_cast<char const*>(tensor.values->data()), tensor.values->size() * 2);
    require(bool(output), "cannot write safetensors fixture");
}
void prepare(std::string const& kind, fs::path const& root, char const* real)
{
    fs::create_directories(root);
    Projection data(kind == "fc1");
    if (real && kind == "fc1") {
        auto source = tensoracle::open_weights(real);
        auto weight = source.view().at("weight"), bias = source.view().at("bias");
        require(weight.dtype == TENSORACLE_FLOAT16 && weight.element_count == data.weight.size() &&
            bias.dtype == TENSORACLE_FLOAT16 && bias.element_count == data.bias.size(), "incompatible real fc1 weights");
        std::copy_n(static_cast<uint16_t const*>(weight.data), data.weight.size(), data.weight.begin());
        std::copy_n(static_cast<uint16_t const*>(bias.data), data.bias.size(), data.bias.begin());
    }
    save_weights(root / "weights.safetensors", kind, data);
    std::vector<Case> cases;
    auto append = [&](bool exact) {
        Case item; item.input = data.x; item.exact = exact;
        for (auto value : data.reference(kind == "fc1")) item.expected.push_back(half_bits(value));
        cases.push_back(std::move(item));
    };
    if (kind == "bias") {
        for (int count : {1, 127, 128, 129, 4097}) {
            Case item; item.exact = true;
            for (int i = 0; i < count; ++i) {
                auto value = half_bits(std::sin(i * .17) * 5); item.input.push_back(value);
                item.expected.push_back(half_bits(half_value(value) - .375));
            }
            cases.push_back(std::move(item));
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            append(false);
            std::rotate(data.x.begin(), data.x.begin() + 13, data.x.end());
            for (auto& value : data.x) value = half_bits(-half_value(value));
        }
        std::fill(data.x.begin(), data.x.end(), 0); append(true);
        if (kind == "qkv") { data.x[731] = half_bits(1); append(true); }
        else {
            double norm = 0;
            for (int c = 0; c < 1280; ++c) norm += std::pow(half_value(data.weight[17 * 1280 + c]), 2);
            require(norm > 0, "zero cancellation row");
            for (double target : {0., -.75, .001}) {
                for (int c = 0; c < 1280; ++c) data.x[c] = half_bits(half_value(data.weight[17 * 1280 + c]) *
                    (target - half_value(data.bias[17])) / norm);
                append(false);
            }
        }
    }
    save_cases(root / "cases.bin", cases);
    if (kind == "fc1") {
        std::fill(data.weight.begin(), data.weight.end(), 0);
        std::fill(data.bias.begin(), data.bias.end(), half_bits(-1));
        std::fill(data.x.begin(), data.x.end(), 0); data.x[0] = data.x[1] = half_bits(1);
        double sentinels[]{.0001,.0002,.0003,.0004,.0005,.0006,.001,.0015};
        for (int row = 0; row < data.rows; ++row) {
            data.weight[row * 1280] = half_bits(1); data.weight[row * 1280 + 1] = half_bits(sentinels[row % 8]);
        }
        save_weights(root / "rounding.safetensors", kind, data);
        cases.clear(); append(true); save_cases(root / "rounding.bin", cases);
    }
    // Exercise the actual v2 reader without CUDA initialization.
    auto verified = tensoracle::open_weights(root / "weights.safetensors");
    require(verified.view().size() == (kind == "bias" ? 1 : kind == "qkv" ? 3 : 2), "fixture lost tensors");
}
struct Options {
    std::string kind, mode, bundle, library; fs::path root;
    bool rounding() const { return mode == "rounding"; }
    fs::path weights() const { return root / (rounding() ? "rounding.safetensors" : "weights.safetensors"); }
    bool legacy() const { return mode == "legacy-build"; }
    std::string name() const { return kind == "bias" ? "LffwFp16Bias" : kind == "qkv" ? "LffwFp16Qkv" : "LffwFp16Fc1"; }
    int output() const { return kind == "qkv" ? 3840 : 5120; }
};
void define_network(INetworkDefinition& network, tensoracle::weights_view const& weights,
    Options const& o, std::string const& digest, int invalid = -1,
    std::vector<std::vector<float>>* invalid_storage = nullptr)
{
    auto version = o.kind == "qkv" && !o.legacy() ? "2" : "1";
    auto* creator = dynamic_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator(o.name().c_str(), version, "listenflow.fastest_whisper"));
    require(creator, "creator missing before graph construction");
    std::vector<PluginField> fields{{"bundle_sha256", digest.data(), PluginFieldType::kCHAR, 64}};
    uint16_t bias = half_bits(-.375);
    std::vector<std::vector<uint16_t>> legacy_values;
    if (o.kind == "bias") fields.emplace_back("bias", &bias, PluginFieldType::kFLOAT16, 1);
    if (o.legacy()) {
        for (auto const* name : {"qkv_weight", "q_bias", "v_bias"}) {
            auto value = weights.at(name); auto* first = static_cast<uint16_t const*>(value.data);
            legacy_values.emplace_back(first, first + value.element_count);
        }
        for (int i = 0; i < 3; ++i) fields.emplace_back(
            std::array<char const*,3>{"qkv_weight","q_bias","v_bias"}[i], legacy_values[i].data(),
            PluginFieldType::kFLOAT16, legacy_values[i].size());
    }
    PluginFieldCollection collection{int(fields.size()), fields.data()};
    std::unique_ptr<IPluginV3> plugin(creator->createPlugin("projection", &collection, TensorRTPhase::kBUILD));
    require(bool(plugin), "valid plugin fields rejected");
    bias = half_bits(99); // Fields must own their data after creation.
    for (auto& values : legacy_values) std::fill(values.begin(), values.end(), half_bits(NAN));
    std::vector<int64_t> shape = o.kind == "bias" ? std::vector<int64_t>{-1} : std::vector<int64_t>{1,1,1280};
    DataType type = DataType::kHALF;
    if (invalid == 0) shape = {1280};
    if (invalid == 1) shape = {1,1,1279};
    if (invalid == 2) shape = {1,2,1280};
    if (invalid == 3) type = DataType::kFLOAT;
    if (invalid == 4 && o.kind == "qkv") shape = {2,1,1280};
    auto* features = network.addInput("features", type, dims(shape)); require(features, "cannot add features");
    std::vector<ITensor*> inputs{features};
    if (o.kind != "bias" && !o.legacy()) {
        std::vector<char const*> names{o.kind == "qkv" ? "qkv_weight" : "weight", o.kind == "qkv" ? "q_bias" : "bias"};
        if (o.kind == "qkv") names.push_back("v_bias");
        for (std::size_t index = 0; index < names.size(); ++index) {
            auto value = weights.at(names[index]);
            std::vector<int64_t> constant_shape(value.dimensions, value.dimensions + value.rank);
            auto constant_type = DataType::kHALF;
            // Use shape/type failures from the former Python graph probes.
            int first = o.kind == "qkv" ? 5 : 4;
            if (invalid == first && index == 0) constant_shape = {1280, o.output()};
            if (invalid == first + 1 && index == 0) constant_shape = {1, o.output(), 1280};
            if (invalid == first + 2 && index == 1 && o.kind == "qkv") constant_shape = {1, int64_t(value.element_count)};
            if (invalid == first + 3 && index == names.size() - 1) constant_shape = {int64_t(value.element_count) - 1};
            if (o.kind == "qkv") {
                if (invalid == first + 4 && index == 0) constant_type = DataType::kFLOAT;
                if (invalid == first + 5 && index == 1) constant_type = DataType::kFLOAT;
                if (invalid == first + 6 && index == 2) constant_type = DataType::kFLOAT;
            } else {
                if (invalid == 6 && index == 0) constant_type = DataType::kFLOAT;
                if (invalid == 8 && index == 1) constant_type = DataType::kFLOAT;
            }
            void const* constant_data = value.data;
            if (constant_type == DataType::kFLOAT) {
                require(invalid_storage, "invalid constant storage missing");
                invalid_storage->emplace_back(value.element_count, 0.f);
                constant_data = invalid_storage->back().data();
            }
            Weights tensor{constant_type, constant_data, int64_t(value.element_count)};
            auto* constant = network.addConstant(dims(constant_shape), tensor);
            require(constant, "cannot add constant"); constant->setName(names[index]); inputs.push_back(constant->getOutput(0));
        }
    }
    auto* layer = network.addPluginV3(inputs.data(), inputs.size(), nullptr, 0, *plugin);
    require(layer, "plugin input contract rejected");
    auto* output = layer->getOutput(0); output->setName("result"); network.markOutput(*output);
}
tensoracle::builder_options declaration(Options const& o)
{
    tensoracle::builder_options values;
    values.name = "lffw-native-" + o.kind; values.graph_version = o.rounding() ? "rounding" : "1";
    values.weights = {o.weights().string()}; values.plugins = {o.library};
    if (o.kind == "bias") {
        values.inputs = {{"features", {1}, {128}, {4097}}};
        values.io = {{"features", TENSORACLE_FLOAT16, {-1}, true}, {"result", TENSORACLE_FLOAT16, {-1}, false}};
    } else {
        values.inputs = {{"features", {1,1,1280}, {1,1,1280}, {1,1,1280}}};
        values.io = {{"features", TENSORACLE_FLOAT16, {1,1,1280}, true}, {"result", TENSORACLE_FLOAT16, {1,1,o.output()}, false}};
    }
    values.workspace_limit = 128 * 1024 * 1024; values.optimization_level = 0; return values;
}
template<class Execution>
std::vector<uint16_t> infer(Execution& execution, Options const& o, std::vector<Case> const& cases)
{
    std::vector<uint16_t> all;
    Buffer source(4097 * 2), target(5120 * 2); Stream stream;
    for (auto const& item : cases) {
        if (o.kind == "bias") execution.set_input_shapes({{"features", {int64_t(item.input.size())}}});
        auto shape = o.kind == "bias" ? std::vector<int64_t>{int64_t(item.input.size())} : std::vector<int64_t>{1,1,o.output()};
        require(execution.get_tensor_shape("result") == shape, "engine output shape changed");
        require(execution.get_tensor_dtype("result") == TENSORACLE_FLOAT16, "engine output dtype changed");
        cuda_check(cuMemsetD8(target.value, 0xff, 5120 * 2));
        cuda_check(cuMemcpyHtoD(source.value, item.input.data(), item.input.size() * 2));
        execution.enqueue({{"features", reinterpret_cast<void*>(source.value)}, {"result", reinterpret_cast<void*>(target.value)}}, stream.value);
        cuda_check(cuStreamSynchronize(stream.value));
        std::vector<uint16_t> actual(item.expected.size()); cuda_check(cuMemcpyDtoH(actual.data(), target.value, actual.size() * 2));
        for (std::size_t i = 0; i < actual.size(); ++i) {
            double observed = half_value(actual[i]), expected = half_value(item.expected[i]);
            auto tolerance = std::ldexp(1., o.kind == "qkv" ? -12 : -11) + std::ldexp(1., -10) * std::abs(expected);
            require(std::isfinite(observed) && (item.exact ? observed == expected : std::abs(observed - expected) <= tolerance),
                "native plugin graph numerical mismatch at " + std::to_string(i));
        }
        all.insert(all.end(), actual.begin(), actual.end());
    }
    return all;
}
struct Event { std::string name; uint64_t context{}, pointer{}, size{}; int status{}; };
std::vector<Event> events(fs::path const& path)
{
    std::ifstream input(path); require(bool(input), "driver observer did not create its log");
    std::vector<Event> result; std::string line;
    while (std::getline(input, line)) {
        std::replace(line.begin(), line.end(), '|', ' '); std::istringstream fields(line); Event value;
        fields >> value.name >> value.context >> value.pointer >> value.size >> value.status;
        require(bool(fields) && value.status == 0, "driver observer reported a failed or truncated operation"); result.push_back(value);
    }
    return result;
}
void check_resources(Options const& o, bool closed)
{
    auto values = events(o.root / "driver.log");
    std::map<std::string, std::set<std::pair<uint64_t,uint64_t>>> pointers;
    std::multiset<std::pair<uint64_t,uint64_t>> loads, unloads; std::set<uint64_t> contexts;
    std::vector<std::string> names = o.kind == "qkv" ? std::vector<std::string>{"weight","q_bias","v_bias"} : std::vector<std::string>{"fc1_weight","fc1_bias"};
    for (auto const& value : values) {
        require(value.name != "alloc" && value.name != "upload" && value.name != "free", "plugin allocated or uploaded its own constants");
        contexts.insert(value.context);
        if (value.name == "load") { require(value.size == 120, "plugin selected another SM"); loads.emplace(value.context,value.pointer); }
        if (value.name == "unload") unloads.emplace(value.context,value.pointer);
        for (auto const& name : names) if (value.name == name) pointers[name].emplace(value.context,value.pointer);
    }
    require(contexts.size() == 1 && !loads.empty(), "observer did not see the owning CUDA context");
    for (auto const& name : names) require(pointers[name].size() == 1, "execution contexts did not share " + name);
    if (closed) require(loads.size() == 5 && loads == unloads, "plugin modules were not released by their last owners");
}
void context_lifecycle(tensoracle::engine_handle engine, Options const& o, std::vector<Case> const& cases)
{
    std::vector<tensoracle::engine_execution> executions;
    std::vector<uint16_t> expected;
    for (int i = 0; i < 4; ++i) {
        executions.push_back(engine.create_execution()); auto actual = infer(executions.back(), o, cases);
        if (i == 0) expected = actual; else require(actual == expected, "new context output changed");
        check_resources(o, false);
    }
    executions.erase(executions.begin());
    for (auto& execution : executions) require(infer(execution, o, cases) == expected, "closing one context broke a survivor");
    check_resources(o, false);
    executions.push_back(engine.create_execution()); require(infer(executions.back(), o, cases) == expected, "replacement context output changed");
    check_resources(o, false);
    engine = tensoracle::engine_handle{};
    while (executions.size() > 1) executions.erase(executions.begin());
    require(infer(executions.front(), o, cases) == expected, "last execution lost its engine");
    check_resources(o, false); executions.clear(); check_resources(o, true);
}
std::vector<char> serialized(Options const& o, std::string const& digest, int invalid = -1)
{
    auto weights = tensoracle::open_weights(o.weights()); Logger logger;
    std::vector<std::vector<float>> invalid_storage;
    std::unique_ptr<IBuilder> builder(createInferBuilder(logger)); require(bool(builder), "cannot create native builder");
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    require(bool(network), "cannot create native network");
    define_network(*network, weights.view(), o, digest, invalid, &invalid_storage);
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 128 * 1024 * 1024); config->setBuilderOptimizationLevel(0);
    config->clearFlag(BuilderFlag::kTF32);
    if (o.kind == "bias") {
        auto* profile = builder->createOptimizationProfile();
        for (auto [selector, count] : {std::pair{OptProfileSelector::kMIN,1}, {OptProfileSelector::kOPT,128}, {OptProfileSelector::kMAX,4097}})
            require(profile->setDimensions("features", selector, dims({count})), "invalid raw bias profile");
        require(config->addOptimizationProfile(profile) >= 0, "invalid raw bias profile");
    }
    std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    require(bool(plan), "native graph serialization rejected: " + logger.errors);
    auto* first = static_cast<char const*>(plan->data()); return {first, first + plan->size()};
}
struct RawExecution {
    std::unique_ptr<IExecutionContext> value;
    void set_input_shapes(std::vector<tensoracle::input_shape> const& shapes) {
        for (auto const& shape : shapes)
            require(value->setInputShape(shape.name.c_str(), dims(shape.dimensions)), "raw input shape rejected");
        require(value->inferShapes(0, nullptr) == 0, "raw shape inference failed");
    }
    std::vector<int64_t> get_tensor_shape(char const* name) {
        auto shape = value->getTensorShape(name); return {shape.d, shape.d + shape.nbDims};
    }
    tensoracle_dtype get_tensor_dtype(char const* name) {
        require(value->getEngine().getTensorDataType(name) == DataType::kHALF, "raw output dtype changed");
        return TENSORACLE_FLOAT16;
    }
    void enqueue(std::vector<tensoracle::tensor_address> const& addresses, tensoracle_cuda_stream stream) {
        for (auto const& address : addresses)
            require(value->setTensorAddress(address.name.c_str(), address.address), "raw tensor address rejected");
        require(value->enqueueV3(reinterpret_cast<cudaStream_t>(stream)), "raw enqueue failed");
    }
};
void serialized_values(Options const& o, std::string const& digest)
{
    tensoracle::register_libraries({o.library});
    auto plan = serialized(o, digest); Logger logger;
    if (o.legacy()) save(o.root / "legacy.plan", plan);
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger)); require(bool(runtime), "cannot create raw runtime");
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    require(bool(engine), "cannot reopen native serialized plugin plan");
    RawExecution execution{std::unique_ptr<IExecutionContext>(engine->createExecutionContext())};
    require(bool(execution.value), "cannot create raw plugin execution");
    auto output = infer(execution, o, load_cases(o.root / "cases.bin"));
    std::vector<char> raw(output.size() * 2); std::memcpy(raw.data(), output.data(), raw.size());
    save(o.root / (o.legacy() ? "legacy.output" : "serialized.output"), raw);
}
void legacy_reject(Options const& o)
{
    require(getPluginRegistry()->getCreator("LffwFp16Qkv", "1", "listenflow.fastest_whisper") == nullptr, "upgraded runtime registered legacy QKV");
    auto plan = bytes(o.root / "legacy.plan"); Logger logger;
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger)); require(bool(runtime), "cannot create runtime");
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (engine) {
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
        require(!context, "upgraded runtime accepted a legacy QKV plan");
    }
    require(!logger.errors.empty(), "legacy failure provided no TensorRT diagnostic");
    std::cout << "legacy plan rejected: " << logger.errors;
}
void reject_inputs(Options const& o, std::string const& digest)
{
    // A working positive graph prevents unavailable builders from making every
    // negative case appear to reject the intended input contract.
    serialized(o, digest);
    int count = o.kind == "qkv" ? 12 : 9;
    for (int index = 0; index < count; ++index) {
        bool rejected = false;
        try { serialized(o, digest, index); } catch (std::exception const&) { rejected = true; }
        require(rejected, "native graph accepted invalid input case " + std::to_string(index));
    }
    std::cout << "incompatible input graphs rejected=" << count << '\n';
}
} // namespace
int main(int argc, char** argv)
{
    try {
        require(argc >= 4, "usage: plugin_engine_runtime MODE KIND ROOT [BUNDLE PLUGIN [REAL_WEIGHTS]]");
        Options o; o.mode = argv[1]; o.kind = argv[2]; o.root = argv[3];
        require(o.kind == "bias" || o.kind == "qkv" || o.kind == "fc1", "unknown kernel");
        if (o.mode == "prepare") { prepare(o.kind, o.root, argc > 4 ? argv[4] : nullptr); return 0; }
        require(argc == 6, "runtime requires bundle and plugin"); o.bundle = argv[4]; o.library = argv[5];
        auto digest = tensoracle::inspect_bundle(o.bundle.c_str(), o.library.c_str()).sha256();
        Context context;
        if (o.mode == "serialized" || o.mode == "legacy-build") { serialized_values(o, digest); return 0; }
        if (o.mode == "legacy-reject" || o.mode == "reject") {
            tensoracle::register_libraries({o.library});
            if (o.mode == "legacy-reject") legacy_reject(o);
            else reject_inputs(o, digest);
            return 0;
        }
        auto options = declaration(o); bool reused = o.mode == "reuse" || o.mode == "contexts";
        std::size_t build_calls = 0;
        auto engine = tensoracle::open_engine(options, [&](INetworkDefinition& network, tensoracle::weights_view const& weights) {
            require(!reused, "saved engine called graph or materialized constants again"); ++build_calls;
            define_network(network, weights, o, digest);
        });
        require(build_calls == (reused ? 0 : 1), "unexpected graph build count");
        auto cases = load_cases(o.root / (o.rounding() ? "rounding.bin" : "cases.bin"));
        if (o.mode == "contexts") context_lifecycle(std::move(engine), o, cases);
        else {
            auto execution = engine.create_execution(); auto output = infer(execution, o, cases);
            std::vector<char> raw(output.size() * 2); std::memcpy(raw.data(), output.data(), raw.size());
            save(o.root / (o.mode + ".output"), raw);
            std::ofstream(o.root / (o.mode + ".path")) << engine.path();
            std::cout << "cases=" << cases.size() << " build_calls=" << build_calls << '\n';
        }
        return 0;
    } catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
