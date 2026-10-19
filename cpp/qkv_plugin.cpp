#include "tensoracle/cubin_plugin.hpp"

#include <array>
#include <cstring>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" char const tensoracle_bundle_sha256[];

namespace {
using namespace nvinfer1;
using tensoracle::cubin::Fields;
using tensoracle::cubin::Kernel;
using tensoracle::cubin::PluginBase;
constexpr char kName[] = "LffwFp16Qkv";
constexpr char kVersion[] = "2";
constexpr char kNamespace[] = "listenflow.fastest_whisper";
constexpr std::array<int, 3> kSms{86, 89, 120};
constexpr std::int32_t kWidth = 1280;
constexpr unsigned int kRows = 4;

tensoracle::cubin::PluginIO half_io()
{
    tensoracle::cubin::TensorSpec half{DataType::kHALF, TensorFormat::kLINEAR};
    return {{half, half, half, half}, half, 0};
}

bool valid_fields(Fields const& fields) noexcept
{
    auto const* digest = fields.find("bundle_sha256");
    return fields.values().size() == 1 && digest != nullptr &&
        digest->type == PluginFieldType::kCHAR && digest->length == 64 &&
        std::memcmp(digest->bytes.data(), tensoracle_bundle_sha256, 64) == 0;
}

class QkvPlugin final : public PluginBase {
public:
    explicit QkvPlugin(Fields values)
        : PluginBase(kName, kNamespace, std::move(values), half_io())
    {
        for (std::size_t index = 0; index < kSms.size(); ++index) {
            auto const id = "sm" + std::to_string(kSms[index]) + ".qkv_fp16";
            auto const* image = tensoracle::cubin::find_bundle_image(id.c_str());
            if (image == nullptr) throw std::invalid_argument("Missing cubin: " + id);
            kernels_[index] = std::make_unique<Kernel>(*image);
        }
    }

    char const* getPluginVersion() const noexcept override { return kVersion; }

    IPluginV3* clone() noexcept override
    {
        try { return new QkvPlugin(fields()); }
        catch (...) { return nullptr; }
    }

    std::int32_t getOutputShapes(DimsExprs const* inputs, std::int32_t input_count,
        DimsExprs const*, std::int32_t shape_input_count, DimsExprs* outputs,
        std::int32_t output_count, IExprBuilder& builder) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || input_count != 4 ||
            shape_input_count != 0 || output_count != 1 || inputs[0].nbDims != 3) return -1;
        outputs[0].nbDims = 3;
        outputs[0].d[0] = builder.constant(1);
        outputs[0].d[1] = builder.constant(1);
        outputs[0].d[2] = builder.constant(3 * kWidth);
        return 0;
    }

    std::int32_t enqueue(PluginTensorDesc const* input_descriptors,
        PluginTensorDesc const* output_descriptors, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (input_descriptors == nullptr || output_descriptors == nullptr || inputs == nullptr ||
            outputs == nullptr || outputs[0] == nullptr ||
            output_descriptors[0].type != DataType::kHALF ||
            output_descriptors[0].format != TensorFormat::kLINEAR) return -1;
        std::array<Dims, 4> shapes{};
        for (std::size_t index = 0; index < shapes.size(); ++index) {
            if (inputs[index] == nullptr || input_descriptors[index].type != DataType::kHALF ||
                input_descriptors[index].format != TensorFormat::kLINEAR) return -1;
            shapes[index] = input_descriptors[index].dims;
        }
        if (validate_shape(shapes.data(), shapes.size(), output_descriptors[0].dims) != 0) return -1;

        CUdevice device{};
        int major{}, minor{};
        if (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS)
            return -1;
        std::size_t index = 0;
        while (index < kSms.size() && kSms[index] != major * 10 + minor) ++index;
        if (index == kSms.size()) return -1;

        CUdeviceptr weight = reinterpret_cast<CUdeviceptr>(inputs[1]);
        CUdeviceptr q_bias = reinterpret_cast<CUdeviceptr>(inputs[2]);
        CUdeviceptr v_bias = reinterpret_cast<CUdeviceptr>(inputs[3]);
        CUdeviceptr source = reinterpret_cast<CUdeviceptr>(inputs[0]);
        CUdeviceptr target = reinterpret_cast<CUdeviceptr>(outputs[0]);
        void* arguments[] = {&source, &weight, &q_bias, &v_bias, &target};
        return kernels_[index]->launch(3 * kWidth / kRows, 128, arguments, std::size(arguments),
            reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS ? 0 : -1;
    }

protected:
    std::int32_t validate_shape(Dims const* inputs, std::int32_t input_count,
        Dims const& output) const noexcept override
    {
        return inputs != nullptr && input_count == 4 && inputs[0].nbDims == 3 &&
            inputs[0].d[0] == 1 && inputs[0].d[1] == 1 && inputs[0].d[2] == kWidth &&
            inputs[1].nbDims == 2 && inputs[1].d[0] == 3 * kWidth && inputs[1].d[1] == kWidth &&
            inputs[2].nbDims == 1 && inputs[2].d[0] == kWidth &&
            inputs[3].nbDims == 1 && inputs[3].d[0] == kWidth &&
            output.nbDims == 3 && output.d[0] == 1 && output.d[1] == 1 &&
            output.d[2] == 3 * kWidth ? 0 : -1;
    }

private:
    std::array<std::unique_ptr<Kernel>, 3> kernels_{};
};

class QkvCreator final : public IPluginCreatorV3One {
public:
    QkvCreator()
    {
        names_[0] = PluginField("bundle_sha256", nullptr, PluginFieldType::kCHAR, 64);
        collection_.nbFields = static_cast<std::int32_t>(names_.size());
        collection_.fields = names_.data();
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &collection_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* source,
        TensorRTPhase phase) noexcept override
    {
        if (phase != TensorRTPhase::kBUILD && phase != TensorRTPhase::kRUNTIME) return nullptr;
        try {
            Fields fields;
            if (!Fields::parse(source, fields) || !valid_fields(fields)) return nullptr;
            return new QkvPlugin(std::move(fields));
        } catch (...) { return nullptr; }
    }

private:
    std::array<PluginField, 1> names_{};
    PluginFieldCollection collection_{};
};
}  // namespace

nvinfer1::IPluginCreatorInterface* qkv_creator() noexcept
{
    static QkvCreator creator;
    return &creator;
}
