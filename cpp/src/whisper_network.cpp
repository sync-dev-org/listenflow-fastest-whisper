#include "cache_recovery.hpp"
#include "model_error.hpp"
#include "network_graph.hpp"
#include "precision_policy.hpp"
#include "sha256.hpp"
#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <listenflow/whisper_network.hpp>
#include <locale>
#include <sstream>

namespace listenflow::fastest_whisper {
namespace {
using shape = std::vector<int64_t>;
std::string precision_name(precision p) {
    switch (p) {
    case precision::fp32:
        return "fp32";
    case precision::fp16:
        return "fp16";
    case precision::fp8:
        return "fp8-1";
    }
    throw std::invalid_argument("unknown Whisper precision");
}
bool decoder(graph_kind k) { return k == graph_kind::decoder || k == graph_kind::sampled_decoder; }
void require(bool condition, const char *reason) {
    if (!condition)
        throw std::invalid_argument(reason);
}
void safetensors(const std::string &file) {
    require(!file.empty() && std::filesystem::path(file).extension() == ".safetensors",
            "build weights must be safetensors");
}
std::string graph_identity(const model_source &s, graph_kind kind, precision p, int32_t batch) {
    detail::sha256 digest;
    digest.field(detail::file_digest(s.config_file));
    if (decoder(kind))
        digest.field(detail::file_digest(s.tokenizer_file));
    if (kind == graph_kind::mel)
        digest.field(detail::file_digest(s.mel_filter_file));
    else if (p == precision::fp8)
        digest.field(detail::file_digest(s.scale_file));
    // Concrete parsed values also determine the graph. Include them even if a
    // caller passes values that differ from the corresponding auxiliary bytes.
    auto &c = s.config;
    std::ostringstream values;
    values.imbue(std::locale::classic());
    values << std::hexfloat;
    for (auto v :
         {c.n_mels, c.audio_positions, c.audio_state, c.audio_heads, c.audio_ffn, c.audio_layers,
          c.text_heads, c.text_ffn, c.text_layers, c.text_positions, c.vocab_size})
        values << v << ',';
    if (decoder(kind)) {
        values << "nospeech=" << (s.no_speech ? std::to_string(*s.no_speech) : "none");
        values << ";timestamp="
               << (s.timestamp_begin ? std::to_string(*s.timestamp_begin) : "none");
    }
    if (kind != graph_kind::mel && p == precision::fp8)
        for (auto &item : s.scales)
            values << ';' << item.first.size() << ':' << item.first << '=' << item.second;
    digest.field(values.str());
    std::string version = kind == graph_kind::mel ? "2" : kind == graph_kind::encoder ? "3" : "5";
    version += "-" + precision_name(p) + "-native3";
    if (kind == graph_kind::sampled_decoder)
        version += "-sampled-1-b" + std::to_string(batch);
    return version + "-aux-" + digest.finish();
}
} // namespace
engine_build::engine_build(model_source source, graph_kind kind, precision dtype,
                           int32_t sample_batch)
    : source_(std::move(source)), kind_(kind), precision_(dtype),
      batch_(kind == graph_kind::sampled_decoder ? sample_batch : 1) {
    const auto &s = source_;
    const auto &c = s.config;
    for (auto n :
         {c.n_mels, c.audio_positions, c.audio_state, c.audio_heads, c.audio_ffn, c.audio_layers,
          c.text_heads, c.text_ffn, c.text_layers, c.text_positions, c.vocab_size})
        require(n > 0 && n <= std::numeric_limits<int32_t>::max(),
                "Whisper dimensions must be positive int32 values");
    require(c.audio_state % c.audio_heads == 0 && c.audio_state % c.text_heads == 0,
            "Whisper state must divide attention heads");
    require(c.text_positions >= 64,
            "Whisper text positions must include the optimum profile length 64");
    require(batch_ > 0, "sample batch must be positive");
    require(kind == graph_kind::mel || kind == graph_kind::encoder || decoder(kind),
            "unknown Whisper graph");
    (void)precision_name(dtype);
    for (auto id : {s.no_speech, s.timestamp_begin})
        if (id)
            require(*id >= 0 && *id < c.vocab_size, "token ID is outside vocabulary");
    auto &o = options_;
    o.workspace_limit = int64_t(4) * 1024 * 1024 * 1024;
    o.optimization_level = 5;
    o.name = "listenflow-fastest-whisper-" + std::string(kind == graph_kind::mel       ? "mel"
                                                         : kind == graph_kind::encoder ? "encoder"
                                                         : kind == graph_kind::decoder
                                                             ? "decoder"
                                                             : "sampled-decoder");
    if (kind == graph_kind::mel) {
        safetensors(s.mel_weights);
        o.weights = {s.mel_weights};
    } else {
        require(!s.weights.empty(), "checkpoint safetensors are required");
        for (auto &file : s.weights)
            safetensors(file);
        o.weights = s.weights;
    }
    if (decoder(kind) && dtype == precision::fp16) {
        require(!s.fc1_library.empty(), "required fp16 fc1 library is missing");
        o.plugins = {s.fc1_library};
    }
    o.graph_version = graph_identity(s, kind, dtype, batch_);
    auto tensor = [&](std::string name, tensor_dtype type, shape value, bool input) {
        o.io.push_back({std::move(name), type, std::move(value), input});
    };
    auto fixed = [&](std::string name, tensor_dtype type, shape value) {
        tensor(name, type, value, true);
        auto &spec = o.io.back();
        spec.minimum = value;
        spec.optimum = value;
        spec.maximum = std::move(value);
    };
    auto ranged = [&](std::string name, tensor_dtype type, shape value, shape low, shape optimum,
                      shape high) {
        tensor(name, type, std::move(value), true);
        auto &spec = o.io.back();
        spec.minimum = std::move(low);
        spec.optimum = std::move(optimum);
        spec.maximum = std::move(high);
    };
    auto output = [&](std::string name, tensor_dtype type, shape value) {
        tensor(std::move(name), type, std::move(value), false);
    };
    const auto type = dtype == precision::fp32 ? tensor_dtype::fp32 : tensor_dtype::fp16;
    const int64_t width = c.audio_state / c.text_heads;
    const shape cross{c.text_layers, 1, c.text_heads, c.audio_positions, width};
    if (kind == graph_kind::mel) {
        fixed("audio", tensor_dtype::fp32, {480400});
        output("mel", tensor_dtype::fp32, {c.n_mels, 3000});
        output("log_mel", tensor_dtype::fp32, {c.n_mels, 3000});
    } else if (kind == graph_kind::encoder) {
        fixed("mel", tensor_dtype::fp32, {1, c.n_mels, c.audio_positions * 2});
        output("encoder", type, {1, c.audio_positions, c.audio_state});
        output("cross_keys", type, cross);
        output("cross_values", type, cross);
    } else {
        fixed("tokens", tensor_dtype::int32, {batch_, 1});
        fixed("positions", tensor_dtype::int32, {batch_, 1});
        ranged("mask", tensor_dtype::fp32, {1, 1, 1, -1}, {1, 1, 1, 2}, {1, 1, 1, 65},
               {1, 1, 1, c.text_positions + 1});
        const shape self{c.text_layers, batch_, c.text_heads, -1, width};
        for (const auto *name : {"self_keys", "self_values"})
            ranged(name, type, self, {c.text_layers, batch_, c.text_heads, 1, width},
                   {c.text_layers, batch_, c.text_heads, 64, width},
                   {c.text_layers, batch_, c.text_heads, c.text_positions, width});
        fixed("cross_keys", type, cross);
        fixed("cross_values", type, cross);
        fixed("suppression", tensor_dtype::fp32, {batch_, 1, c.vocab_size});
        if (kind == graph_kind::sampled_decoder) {
            fixed("temperature", tensor_dtype::fp32, {1, 1, 1});
            fixed("uniform", tensor_dtype::fp32, {batch_, 1, 1});
            fixed("timestamps", tensor_dtype::boolean, {1, 1, 1});
        }
        output("logits", type, {batch_, 1, c.vocab_size});
        if (kind == graph_kind::decoder) {
            output("token", tensor_dtype::int32, {1, 1, 1});
            output("greedy", tensor_dtype::fp32, {1, 1, 3});
        } else
            output("sampled", tensor_dtype::fp32, {batch_, 1, 3});
        output("present_keys", type, self);
        output("present_values", type, self);
    }
}
namespace {
tensoracle_dtype backend_dtype(tensor_dtype type) {
    switch (type) {
    case tensor_dtype::fp32:
        return TENSORACLE_FLOAT32;
    case tensor_dtype::fp16:
        return TENSORACLE_FLOAT16;
    case tensor_dtype::int32:
        return TENSORACLE_INT32;
    case tensor_dtype::boolean:
        return TENSORACLE_BOOL;
    }
    throw std::invalid_argument("unsupported model tensor dtype");
}
tensor_dtype model_dtype(tensoracle_dtype type) {
    switch (type) {
    case TENSORACLE_FLOAT32:
        return tensor_dtype::fp32;
    case TENSORACLE_FLOAT16:
        return tensor_dtype::fp16;
    case TENSORACLE_INT32:
        return tensor_dtype::int32;
    case TENSORACLE_BOOL:
        return tensor_dtype::boolean;
    }
    throw std::runtime_error("unsupported backend tensor dtype");
}
tensoracle::builder_options backend_options(const build_declaration &declaration) {
    tensoracle::builder_options options;
    options.name = declaration.name;
    options.graph_version = declaration.graph_version;
    options.weights = declaration.weights;
    options.plugins = declaration.plugins;
    options.workspace_limit = declaration.workspace_limit;
    options.optimization_level = declaration.optimization_level;
    for (const auto &spec : declaration.io) {
        options.io.push_back({spec.name, backend_dtype(spec.dtype), spec.shape, spec.input});
        if (spec.input)
            options.inputs.push_back({spec.name, spec.minimum, spec.optimum, spec.maximum});
    }
    return options;
}
void scratch_cuda_check(CUresult status, const char *operation) {
    if (status != CUDA_SUCCESS)
        throw detail::cuda_error(status == CUDA_ERROR_OUT_OF_MEMORY ? TENSORACLE_OUT_OF_MEMORY
                                                                  : TENSORACLE_INTERNAL_ERROR,
                                 std::string(operation) + " failed: CUDA status " +
                                     std::to_string(status));
}
// The engine and every execution retain this owner. Destroying an execution
// releases its backend handle before dropping its share of the scratch.
struct engine_state {
    tensoracle::engine_handle handle;
    CUcontext context = nullptr;
    CUdeviceptr scratch = 0;
    size_t scratch_bytes;
    explicit engine_state(tensoracle::engine_handle value)
        : handle(std::move(value)), scratch_bytes(std::max(size_t(1), handle.scratch_size())) {
        scratch_cuda_check(cuCtxGetCurrent(&context), "scratch cuCtxGetCurrent");
        scratch_cuda_check(cuMemAlloc(&scratch, scratch_bytes), "scratch cuMemAlloc");
    }
    ~engine_state() {
        try { close(); } catch (...) {}
    }
    void close() {
        if (!scratch)
            return;
        scratch_cuda_check(cuCtxPushCurrent(context), "scratch cuCtxPushCurrent");
        const auto status = cuMemFree(scratch);
        if (status == CUDA_SUCCESS)
            scratch = 0;
        CUcontext previous = nullptr;
        const auto restored = cuCtxPopCurrent(&previous);
        scratch_cuda_check(status, "scratch cuMemFree");
        scratch_cuda_check(restored, "scratch cuCtxPopCurrent");
    }
    engine_state(const engine_state &) = delete;
};
// Cached-plugin probes and all buckets borrow the same allocation. Callers
// serialize work on an engine, including CUDA graph replays using its scratch.
tensoracle::engine_execution new_execution(const engine_state &engine) {
    return engine.handle.create_execution(reinterpret_cast<void *>(engine.scratch),
                                          engine.scratch_bytes);
}
} // namespace
struct whisper_engine::implementation {
    std::shared_ptr<engine_state> engine;
    explicit implementation(std::shared_ptr<engine_state> value) : engine(std::move(value)) {}
};
struct network_execution::implementation {
    std::shared_ptr<engine_state> engine;
    tensoracle::engine_execution execution;
    explicit implementation(std::shared_ptr<engine_state> value)
        : engine(std::move(value)), execution(new_execution(*engine)) {}
};
whisper_engine::whisper_engine() = default;
whisper_engine::~whisper_engine() = default;
whisper_engine::whisper_engine(whisper_engine &&) noexcept = default;
whisper_engine &whisper_engine::operator=(whisper_engine &&) noexcept = default;
whisper_engine::whisper_engine(std::unique_ptr<implementation> value) : impl_(std::move(value)) {}
void whisper_engine::close() {
    if (impl_ && impl_->engine.use_count() == 1)
        impl_->engine->close();
    impl_.reset();
}
std::string whisper_engine::path() const {
    if (!impl_)
        throw std::invalid_argument("Whisper engine is closed");
    return detail::model_call([&] { return impl_->engine->handle.path(); });
}
network_execution::network_execution() = default;
network_execution::~network_execution() = default;
network_execution::network_execution(network_execution &&) noexcept = default;
network_execution &network_execution::operator=(network_execution &&) noexcept = default;
network_execution::network_execution(std::unique_ptr<implementation> value)
    : impl_(std::move(value)) {}
void network_execution::close() {
    if (!impl_)
        return;
    impl_->execution.close();
    if (impl_->engine.use_count() == 1)
        impl_->engine->close();
    impl_.reset();
}
void network_execution::set_input_shapes(const std::vector<input_shape> &values) {
    if (!impl_)
        throw std::invalid_argument("Whisper execution is closed");
    detail::model_call([&] {
        std::vector<tensoracle::input_shape> shapes;
        for (const auto &v : values)
            shapes.push_back({v.name, v.dimensions});
        impl_->execution.set_input_shapes(shapes);
    });
}
std::vector<int64_t> network_execution::get_tensor_shape(const char *name) {
    if (!impl_)
        throw std::invalid_argument("Whisper execution is closed");
    return detail::model_call([&] { return impl_->execution.get_tensor_shape(name); });
}
tensor_dtype network_execution::get_tensor_dtype(const char *name) const {
    if (!impl_)
        throw std::invalid_argument("Whisper execution is closed");
    return detail::model_call([&] { return model_dtype(impl_->execution.get_tensor_dtype(name)); });
}
void network_execution::enqueue(const std::vector<tensor_address> &values, void *stream) {
    if (!impl_)
        throw std::invalid_argument("Whisper execution is closed");
    detail::model_call([&] {
        std::vector<tensoracle::tensor_address> addresses;
        for (const auto &v : values)
            addresses.push_back({v.name, v.address});
        impl_->execution.enqueue(addresses, static_cast<tensoracle_cuda_stream>(stream));
    });
}
whisper_engine engine_build::open() const {
    return detail::model_call([&] {
        if (precision_ == precision::fp8) {
            CUdevice device;
            int major = 0, minor = 0;
            if (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
                cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                     device) != CUDA_SUCCESS ||
                cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                     device) != CUDA_SUCCESS)
                throw std::runtime_error(
                    "cannot determine fp8 capability in the caller's current CUDA context");
            detail::check_precision(precision_, major, minor);
        }
        const auto declaration = backend_options(options_);
        auto handle = detail::recover_saved_engine(
            !options_.plugins.empty(),
            [&](bool force, bool &built) {
                auto options = declaration;
                options.force_rebuild = force;
                // Converted constants/plugins outlive both serializations.
                detail::weight_storage storage;
                auto engine = tensoracle::open_engine(
                    options, [&](nvinfer1::INetworkDefinition &net,
                                 const tensoracle::weights_view &weights) {
                        built = true;
                        detail::define_network(net, weights, source_, kind_, precision_, batch_,
                                               declaration, storage);
                    });
                return std::make_shared<engine_state>(std::move(engine));
            },
            [](auto &engine) {
                // TensorRT can defer plugin rejection to context creation,
                // reported by tensoracle as INTERNAL_ERROR. Only
                // this cached-engine probe, never callback/OOM/input errors, retries.
                auto execution = new_execution(*engine);
            });
        return whisper_engine(std::make_unique<whisper_engine::implementation>(std::move(handle)));
    });
}
std::vector<input_shape> engine_build::bucket_shapes(int64_t history) const {
    require(decoder(kind_), "only decoders have history buckets");
    require(history >= 1 && history <= source_.config.text_positions,
            "history is outside decoder profile");
    std::vector<input_shape> result;
    for (const auto &spec : options_.io)
        if (spec.input && std::find(spec.shape.begin(), spec.shape.end(), -1) != spec.shape.end()) {
            auto shape = spec.shape;
            for (auto &d : shape)
                if (d == -1)
                    d = history + (spec.name == "mask" ? 1 : 0);
            result.push_back({spec.name, std::move(shape)});
        }
    return result;
}
network_execution engine_build::create_execution(const whisper_engine &handle,
                                                 std::optional<int64_t> history) const {
    if (!handle.impl_)
        throw std::invalid_argument("Whisper engine is closed");
    return detail::model_call([&] {
        auto execution = network_execution(std::make_unique<network_execution::implementation>(
            handle.impl_->engine));
        if (history)
            execution.set_input_shapes(bucket_shapes(*history));
        return execution;
    });
}
std::vector<bucket_execution>
engine_build::create_sampled_executions(const whisper_engine &handle) const {
    require(kind_ == graph_kind::sampled_decoder, "bucket execution set requires sampled decoder");
    std::vector<bucket_execution> result;
    result.reserve(cache_buckets.size());
    for (auto history : cache_buckets)
        if (history <= source_.config.text_positions)
            result.push_back({history, create_execution(handle, history)});
    return result;
}
} // namespace listenflow::fastest_whisper
