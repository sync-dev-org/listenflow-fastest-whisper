#include "tensoracle/cubin_plugin.hpp"

#include <array>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" char const tensoracle_bundle_sha256[];
nvinfer1::IPluginCreatorInterface* qkv_creator() noexcept;
nvinfer1::IPluginCreatorInterface* fc1_creator() noexcept;

namespace {
using namespace nvinfer1;
using tensoracle::cubin::DeviceBytes;
using tensoracle::cubin::Fields;
using tensoracle::cubin::Kernel;
using tensoracle::cubin::PluginBase;
constexpr char kName[] = "LffwFp16Bias";
constexpr char kNamespace[] = "listenflow.fastest_whisper";
constexpr std::array<int, 3> kSms{86, 89, 120};
constexpr unsigned int kBlock = 128;

tensoracle::cubin::PluginIO half_io()
{
    tensoracle::cubin::TensorSpec half{DataType::kHALF, TensorFormat::kLINEAR};
    return {{half}, half, 0};
}

bool valid_fields(Fields const& fields) noexcept
{
    auto const* digest = fields.find("bundle_sha256");
    auto const* bias = fields.find("bias");
    return fields.values().size() == 2 && digest != nullptr && bias != nullptr &&
        digest->type == PluginFieldType::kCHAR && digest->length == 64 &&
        std::memcmp(digest->bytes.data(), tensoracle_bundle_sha256, 64) == 0 &&
        bias->type == PluginFieldType::kFLOAT16 && bias->length == 1;
}

class BiasPlugin final : public PluginBase {
public:
    explicit BiasPlugin(Fields values)
        : PluginBase(kName, kNamespace, std::move(values), half_io()),
          bias_(fields().find("bias")->bytes)
    {
        for (std::size_t index = 0; index < kSms.size(); ++index) {
            auto const id = "sm" + std::to_string(kSms[index]) + ".bias_fp16";
            auto const* image = tensoracle::cubin::find_bundle_image(id.c_str());
            if (image == nullptr) throw std::invalid_argument("Missing cubin: " + id);
            kernels_[index] = std::make_unique<Kernel>(*image);
        }
    }

    IPluginV3* clone() noexcept override
    {
        try { return new BiasPlugin(fields()); }
        catch (...) { return nullptr; }
    }

    std::int32_t enqueue(PluginTensorDesc const* input_descriptors,
        PluginTensorDesc const* output_descriptors, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (input_descriptors == nullptr || output_descriptors == nullptr || inputs == nullptr ||
            outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr ||
            validate_shape(&input_descriptors[0].dims, 1, output_descriptors[0].dims) != 0 ||
            input_descriptors[0].dims.d[0] < 1) return -1;

        CUdevice device{};
        int major{}, minor{};
        if (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS)
            return -1;
        std::size_t index = 0;
        while (index < kSms.size() && kSms[index] != major * 10 + minor) ++index;
        if (index == kSms.size()) return -1;

        CUdeviceptr bias{};
        if (bias_.for_current_context(bias) != CUDA_SUCCESS) return -1;
        CUdeviceptr source = reinterpret_cast<CUdeviceptr>(inputs[0]);
        CUdeviceptr target = reinterpret_cast<CUdeviceptr>(outputs[0]);
        auto count = static_cast<std::int32_t>(input_descriptors[0].dims.d[0]);
        void* arguments[] = {&source, &bias, &target, &count};
        auto const grid = (static_cast<unsigned int>(count) + kBlock - 1) / kBlock;
        return kernels_[index]->launch(grid, kBlock, arguments, std::size(arguments),
            reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS ? 0 : -1;
    }

protected:
    std::int32_t validate_shape(Dims const* inputs, std::int32_t input_count,
        Dims const& output) const noexcept override
    {
        if (inputs == nullptr || input_count != 1 || inputs[0].nbDims != 1 || output.nbDims != 1)
            return -1;
        auto const valid_length = [](std::int64_t length) {
            return length == -1 || (length > 0 && length <= std::numeric_limits<std::int32_t>::max());
        };
        auto const source = inputs[0].d[0];
        auto const target = output.d[0];
        return valid_length(source) && valid_length(target) &&
            (source == -1 || target == -1 || source == target) ? 0 : -1;
    }

private:
    DeviceBytes bias_;
    std::array<std::unique_ptr<Kernel>, 3> kernels_{};
};

class Creator final : public IPluginCreatorV3One {
public:
    Creator()
    {
        names_[0] = PluginField("bundle_sha256", nullptr, PluginFieldType::kCHAR, 64);
        names_[1] = PluginField("bias", nullptr, PluginFieldType::kFLOAT16, 1);
        collection_.nbFields = static_cast<std::int32_t>(names_.size());
        collection_.fields = names_.data();
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &collection_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* source,
        TensorRTPhase phase) noexcept override
    {
        if (phase != TensorRTPhase::kBUILD && phase != TensorRTPhase::kRUNTIME) return nullptr;
        try {
            Fields fields;
            if (!Fields::parse(source, fields) || !valid_fields(fields)) return nullptr;
            return new BiasPlugin(std::move(fields));
        } catch (...) { return nullptr; }
    }

private:
    std::array<PluginField, 2> names_{};
    PluginFieldCollection collection_{};
};
Creator gCreator;
}  // namespace

extern "C" __attribute__((visibility("default")))
void setLoggerFinder(nvinfer1::ILoggerFinder*) noexcept {}

extern "C" __attribute__((visibility("default")))
nvinfer1::IPluginCreatorInterface* const* getCreators(std::int32_t& count) noexcept
{
    static nvinfer1::IPluginCreatorInterface* creators[] = {&gCreator, qkv_creator(), fc1_creator()};
    count = static_cast<std::int32_t>(std::size(creators));
    return creators;
}
