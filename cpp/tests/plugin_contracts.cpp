// CPU checks of the TensorRT creator contract. Loading a library does not initialize CUDA.
#include <NvInfer.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;
namespace {
void require(bool value, char const* reason)
{
    if (!value) throw std::runtime_error(reason);
}
Dims dims(std::initializer_list<std::int64_t> values)
{
    Dims result{};
    result.nbDims = static_cast<int>(values.size());
    std::copy(values.begin(), values.end(), result.d);
    return result;
}
PluginTensorDesc desc(Dims shape)
{
    PluginTensorDesc result{};
    result.dims = shape;
    result.type = DataType::kHALF;
    result.format = TensorFormat::kLINEAR;
    return result;
}
IPluginV3OneRuntime* runtime(IPluginV3& plugin)
{
    return static_cast<IPluginV3OneRuntime*>(plugin.getCapabilityInterface(PluginCapabilityType::kRUNTIME));
}
void check_shapes(IPluginV3& plugin, std::string const& name)
{
    bool bias = name == "LffwFp16Bias";
    bool qkv = name == "LffwFp16Qkv";
    std::vector<PluginTensorDesc> input;
    PluginTensorDesc output = desc(dims({1, 1, qkv ? 3840 : 5120}));
    if (bias) {
        input = {desc(dims({129}))};
        output = desc(dims({129}));
    } else {
        input = {desc(dims({1, 1, 1280})), desc(dims({qkv ? 3840 : 5120, 1280})), desc(dims({1280}))};
        if (qkv) input.push_back(desc(dims({1280})));
        else input.back() = desc(dims({5120}));
    }
    auto* run = runtime(plugin);
    auto* build = static_cast<IPluginV3OneBuild*>(plugin.getCapabilityInterface(PluginCapabilityType::kBUILD));
    auto valid = [&](auto const& inputs, auto const& target) {
        return run->onShapeChange(inputs.data(), inputs.size(), &target, 1) == 0;
    };
    require(valid(input, output), "valid fp16 plugin shapes were rejected");
    require(build->getWorkspaceSize(nullptr, 0, nullptr, 0) == 0, "plugin acquired workspace");
    std::vector<DynamicPluginTensorDesc> dynamic;
    for (auto const& item : input) {
        DynamicPluginTensorDesc value{};
        value.desc = item;
        value.min = value.max = item.dims;
        dynamic.push_back(value);
    }
    DynamicPluginTensorDesc target{};
    target.desc = output;
    target.min = target.max = output.dims;
    require(build->configurePlugin(dynamic.data(), dynamic.size(), &target, 1) == 0,
        "build rejected valid fp16 plugin inputs");
    for (std::size_t index = 0; index < input.size(); ++index) {
        auto bad = input;
        bad[index].type = DataType::kFLOAT;
        require(!valid(bad, output), "plugin accepted fp32 input");
        bad = input;
        bad[index].format = TensorFormat::kCHW2;
        require(!valid(bad, output), "plugin accepted non-LINEAR input");
        bad = input;
        --bad[index].dims.d[bad[index].dims.nbDims - 1];
        require(!valid(bad, output), "plugin accepted incompatible input width");
    }
    auto bad_output = output;
    bad_output.type = DataType::kFLOAT;
    require(!valid(input, bad_output), "plugin accepted fp32 output");
    bad_output = output;
    ++bad_output.dims.d[bad_output.dims.nbDims - 1];
    require(!valid(input, bad_output), "plugin accepted incompatible output shape");
    auto bad = input;
    bad[0].dims = dims({1280});
    require(bias || !valid(bad, output), "projection accepted rank-one features");
    if (bias) {
        for (auto length : {std::int64_t{1}, std::int64_t{127}, std::int64_t{128},
                 std::int64_t{129}, std::int64_t{4097}, std::int64_t{2147483647}}) {
            input[0].dims = output.dims = dims({length});
            require(valid(input, output), "bias rejected valid count boundary");
        }
        for (auto length : {std::int64_t{0}, std::int64_t{-2}, std::int64_t{2147483648}}) {
            input[0].dims = output.dims = dims({length});
            require(!valid(input, output), "bias accepted invalid count");
        }
    } else {
        bad = input;
        bad[0].dims = dims({1, 2, 1280});
        require(!valid(bad, output), "projection accepted multiple tokens");
        bad[0].dims = dims({2, 1, 1280});
        require(!valid(bad, output), "projection accepted a batch");
        bad = input;
        bad[1].dims = dims({1280, qkv ? 3840 : 5120});
        require(!valid(bad, output), "projection accepted transposed weights");
        bad[1].dims = dims({1, qkv ? 3840 : 5120, 1280});
        require(!valid(bad, output), "projection accepted rank-three weights");
        bad = input;
        bad[2].dims = dims({1, qkv ? 1280 : 5120});
        require(!valid(bad, output), "projection accepted rank-two bias");
    }
}
void check_creator(IPluginCreatorV3One& creator, char const* digest)
{
    std::string name = creator.getPluginName();
    bool bias = name == "LffwFp16Bias";
    require(name == "LffwFp16Fc1" || name == "LffwFp16Qkv" || bias, "unexpected creator name");
    require(std::string(creator.getPluginVersion()) == (name == "LffwFp16Qkv" ? "2" : "1"),
        "plugin version changed");
    require(std::string(creator.getPluginNamespace()) == "listenflow.fastest_whisper", "namespace changed");
    auto* names = creator.getFieldNames();
    require(names && names->nbFields == (bias ? 2 : 1), "creator field count changed");
    require(std::string(names->fields[0].name) == "bundle_sha256" &&
        names->fields[0].type == PluginFieldType::kCHAR && names->fields[0].length == 64,
        "bundle digest field changed");
    if (bias) require(std::string(names->fields[1].name) == "bias" &&
        names->fields[1].type == PluginFieldType::kFLOAT16 && names->fields[1].length == 1,
        "bias field changed");
    std::string digest_copy(digest, 64);
    std::uint16_t bias_value = 0xb600; // -0.375 in IEEE binary16.
    std::vector<PluginField> fields{{"bundle_sha256", digest_copy.data(), PluginFieldType::kCHAR, 64}};
    if (bias) fields.emplace_back("bias", &bias_value, PluginFieldType::kFLOAT16, 1);
    auto create = [&](std::vector<PluginField> const& values, TensorRTPhase phase = TensorRTPhase::kBUILD) {
        PluginFieldCollection collection{static_cast<int>(values.size()), values.data()};
        return std::unique_ptr<IPluginV3>(creator.createPlugin("test", &collection, phase));
    };
    auto plugin = create(fields);
    require(bool(plugin), "valid plugin fields rejected");
    digest_copy.assign(64, '0');
    bias_value = 0;
    auto* saved = runtime(*plugin)->getFieldsToSerialize();
    require(saved && saved->nbFields == (bias ? 2 : 1), "serialization lost fields");
    require(std::memcmp(saved->fields[0].data, digest, 64) == 0, "plugin borrowed caller digest bytes");
    if (bias) require(*static_cast<std::uint16_t const*>(saved->fields[1].data) == 0xb600,
        "plugin borrowed caller bias bytes");
    auto restored = std::unique_ptr<IPluginV3>(creator.createPlugin("restored", saved, TensorRTPhase::kRUNTIME));
    require(bool(restored), "serialized fields cannot restore plugin");
    auto clone = std::unique_ptr<IPluginV3>(plugin->clone());
    require(bool(clone), "plugin clone failed");
    check_shapes(*restored, name);
    require(!create({}), "creator accepted missing fields");
    require(!create(fields), "creator accepted another bundle's digest");
    fields[0].data = digest;
    auto invalid = fields;
    invalid[0].length = 63;
    require(!create(invalid), "creator accepted a short digest");
    invalid = fields;
    invalid[0].type = PluginFieldType::kFLOAT32;
    require(!create(invalid), "creator accepted wrong digest type");
    invalid = fields;
    invalid.emplace_back("extra", &bias_value, PluginFieldType::kFLOAT16, 1);
    require(!create(invalid), "creator accepted additional fields");
    if (bias) {
        invalid = fields;
        invalid.pop_back();
        require(!create(invalid), "bias creator accepted missing bias");
        invalid = fields;
        invalid[1].type = PluginFieldType::kFLOAT32;
        require(!create(invalid), "bias creator accepted fp32 bias");
        invalid = fields;
        invalid[1].length = 2;
        require(!create(invalid), "bias creator accepted vector bias");
    }
    std::cout << name << " " << creator.getPluginVersion() << ": creator, fields, clone and shapes pass\n";
}
} // namespace
int main(int argc, char** argv)
{
    try {
        require(argc == 3 || argc == 4, "usage: plugin_contracts LIBRARY EXPECTED_CREATORS [CREATOR]");
        auto* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        auto get = reinterpret_cast<IPluginCreatorInterface* const* (*)(std::int32_t&)>(dlsym(library, "getCreators"));
        auto* digest = static_cast<char const*>(dlsym(library, "tensoracle_bundle_sha256"));
        require(get && digest, "required registration or bundle export missing");
        std::int32_t count{};
        auto creators = get(count);
        require(count == std::stoi(argv[2]), "creator count changed");
        std::vector<std::string> names;
        for (int i = 0; i < count; ++i) {
            auto* creator = dynamic_cast<IPluginCreatorV3One*>(creators[i]);
            require(creator, "creator does not implement the TensorRT V3 interface");
            names.emplace_back(creator->getPluginName());
            if (argc == 3 || names.back() == argv[3]) check_creator(*creator, digest);
        }
        require(std::count(names.begin(), names.end(), "LffwFp16Fc1") == 1, "production fc1 creator missing");
        if (argc == 4) require(std::count(names.begin(), names.end(), argv[3]) == 1, "requested creator missing");
        if (count == 3) require(std::count(names.begin(), names.end(), "LffwFp16Qkv") == 1 &&
            std::count(names.begin(), names.end(), "LffwFp16Bias") == 1, "development creators missing");
        dlclose(library);
        return 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
