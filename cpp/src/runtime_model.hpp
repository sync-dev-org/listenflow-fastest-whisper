#pragma once
#include "runtime_executor.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>

namespace listenflow::fastest_whisper::runtime_detail {
inline std::string text(const char *value) { return value ? value : ""; }
inline model_source source_options(const lffw_model_options *options) {
    require(options && options->struct_size >= sizeof(*options), "model options require all known fields");
    if (options->abi_major != LFFW_ABI_MAJOR) throw failure(LFFW_ABI_MISMATCH, "runtime ABI major differs from header");
    require(options->config && options->config->struct_size >= sizeof(lffw_config), "model config requires all known fields");
    auto &c = *options->config;
    for (auto dimension : {c.n_mels, c.audio_positions, c.audio_state, c.audio_heads, c.audio_ffn, c.audio_layers,
                           c.text_heads, c.text_ffn, c.text_layers, c.text_positions, c.vocab_size})
        require(dimension > 0 && dimension <= INT32_MAX, "model dimensions must be positive int32 values");
    require(c.text_positions >= 64 && c.text_positions <= cache_buckets.back(), "decoder context must be between 64 and 448");
    require(c.audio_state % c.audio_heads == 0 && c.audio_state % c.text_heads == 0, "attention heads must divide model width");
    require(c.audio_positions * 2 == 3000, "encoder features must contain 3000 mel frames");
    require(options->device >= 0, "CUDA device must be nonnegative");
    require(options->precision >= LFFW_FP32 && options->precision <= LFFW_FP8, "unknown model precision");
    require(options->sampled_batch == 0 || options->sampled_batch > 1, "sampled batch must be zero or greater than one");
    require(options->weight_count && options->weights, "checkpoint safetensors are required");
    model_source result;
    result.config = {c.n_mels, c.audio_positions, c.audio_state, c.audio_heads, c.audio_ffn, c.audio_layers,
                     c.text_heads, c.text_ffn, c.text_layers, c.text_positions, c.vocab_size};
    for (size_t i = 0; i < options->weight_count; ++i) {
        require(options->weights[i] && *options->weights[i], "weight path is empty");
        result.weights.emplace_back(options->weights[i]);
    }
    result.config_file = text(options->config_file); result.tokenizer_file = text(options->tokenizer_file);
    result.scale_file = text(options->scale_file); result.mel_filter_file = text(options->mel_filter_file);
    result.mel_weights = text(options->mel_weights); result.fc1_library = text(options->fc1_library);
    require(!result.config_file.empty() && !result.tokenizer_file.empty() && !result.mel_filter_file.empty() && !result.mel_weights.empty(),
            "config, tokenizer and mel asset paths are required");
    require(!options->scale_count || options->scales, "scale array is required");
    for (size_t i = 0; i < options->scale_count; ++i) {
        auto *value = options->scales[i];
        require(value && value->struct_size >= sizeof(*value) && value->name && *value->name && std::isfinite(value->value), "invalid model scale");
        require(result.scales.emplace(value->name, value->value).second, "duplicate model scale");
    }
    for (auto id : {options->no_speech_token, options->timestamp_begin_token})
        require(id == -1 || (id >= 0 && id < c.vocab_size), "special token ID is outside vocabulary");
    if (options->no_speech_token >= 0) result.no_speech = options->no_speech_token;
    if (options->timestamp_begin_token >= 0) result.timestamp_begin = options->timestamp_begin_token;
    return result;
}
inline std::vector<int32_t> token_list(const int32_t *values, size_t count, int64_t vocab) {
    require(!count || values, "suppression token list is required");
    std::vector<int32_t> result;
    for (size_t i = 0; i < count; ++i) {
        require(values[i] >= 0 && values[i] < vocab, "suppression token ID is outside vocabulary");
        result.push_back(values[i]);
    }
    return result;
}
struct model_state {
    std::mutex mutex;
    bool closed = false;
    model_source source;
    precision dtype;
    int32_t device_index, sampled_batch;
    CUdevice device{};
    CUcontext context = nullptr;
    std::unique_ptr<buffer_pool> pool;
    std::unique_ptr<gpu_queue> queue;
    std::unique_ptr<executor> mel, encoder, decoder, sampled;
    std::array<buffer, 2> suppression;
    std::array<double, 4> seconds{};
    std::array<std::string, 4> paths;
    lffw_dtype encoder_type = LFFW_FLOAT32, decoder_type = LFFW_FLOAT32;
    std::vector<int32_t> suppressed, begin_suppressed;
    explicit model_state(const lffw_model_options *options)
        : source(source_options(options)), dtype(static_cast<precision>(options->precision)),
          device_index(options->device), sampled_batch(options->sampled_batch),
          suppressed(token_list(options->suppressed_tokens, options->suppressed_token_count, source.config.vocab_size)),
          begin_suppressed(token_list(options->begin_suppressed_tokens, options->begin_suppressed_token_count, source.config.vocab_size)) {}
    ~model_state() { try { close(); } catch (...) {} }
    void initialize() {
        cuda_check(cuInit(0), "cuInit");
        int count = 0; cuda_check(cuDeviceGetCount(&count), "cuDeviceGetCount");
        require(device_index < count, "selected CUDA device is unavailable");
        cuda_check(cuDeviceGet(&device, device_index), "cuDeviceGet");
        int major = 0, minor = 0;
        cuda_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute major");
        cuda_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute minor");
        const int sm = major * 10 + minor;
        if (sm != 86 && sm != 89 && sm != 120)
            throw failure(LFFW_UNSUPPORTED_PRECISION, "unsupported CUDA compute capability sm" + std::to_string(sm));
        if (dtype == precision::fp8 && sm == 86)
            throw failure(LFFW_UNSUPPORTED_PRECISION, "fp8 is unsupported on sm86; select fp32 or fp16");
        cuda_check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
        context_guard active(context);
        pool = std::make_unique<buffer_pool>(context); queue = std::make_unique<gpu_queue>();
        auto open = [&](auto &target, graph_kind kind, size_t index, int batch = 1) {
            auto started = std::chrono::steady_clock::now();
            target = std::make_unique<executor>(context, *pool, *queue, source, kind, dtype, batch);
            seconds[index] = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            paths[index] = target->path;
        };
        open(mel, graph_kind::mel, 0); open(encoder, graph_kind::encoder, 1);
        open(decoder, graph_kind::decoder, 2);
        if (sampled_batch) open(sampled, graph_kind::sampled_decoder, 3, sampled_batch);
        encoder_type = encoder->spec("encoder").dtype == tensor_dtype::fp16 ? LFFW_FLOAT16 : LFFW_FLOAT32;
        decoder_type = decoder->spec("logits").dtype == tensor_dtype::fp16 ? LFFW_FLOAT16 : LFFW_FLOAT32;
        std::vector<float> mask(size_t(source.config.vocab_size), 0);
        for (auto token : suppressed) mask[size_t(token)] = -std::numeric_limits<float>::infinity();
        for (size_t which = 0; which < 2; ++which) {
            if (which) for (auto token : begin_suppressed) mask[size_t(token)] = -std::numeric_limits<float>::infinity();
            auto bytes = mask.size() * sizeof(float);
            suppression[which] = pool->acquire(bytes);
            // Retain staging with the executor if completion fails: model
            // cleanup drains its queue before destroying pinned host storage.
            auto &staging = decoder->host("initial_suppression", bytes);
            std::memcpy(staging.pointer(), mask.data(), bytes);
            queue->pending = true;
            cuda_check(cuMemcpyHtoDAsync(suppression[which]->address, staging.pointer(), bytes, queue->stream), "cuMemcpyHtoDAsync suppression");
            queue->finish();
        }
    }
    void close() {
        if (closed && !context) return;
        closed = true;
        cleanup_errors errors;
        {
            context_guard active(context);
            if (queue) errors.attempt([&] { queue->drain(); });
            auto release = [&](auto &value) {
                if (value) errors.attempt([&] { value->close(); value.reset(); });
            };
            release(sampled); release(decoder); release(encoder); release(mel);
            bool graphs = false;
            for (auto *value : {sampled.get(), decoder.get(), encoder.get(), mel.get()})
                graphs = graphs || (value && value->has_graphs());
            // A failed graph release still owns its execution and addresses.
            if (!graphs) {
                release(pool);
                suppression = {};
                release(queue);
            }
        }
        if (context && !sampled && !decoder && !encoder && !mel && !pool && !queue)
            errors.attempt([&] {
                cuda_check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease"); context = nullptr;
            });
        errors.rethrow();
    }
};
struct audio_state {
    std::shared_ptr<model_state> model;
    buffer encoder, keys, values;
};
struct decoder_state {
    std::shared_ptr<audio_state> audio;
    lffw_decoder_kind kind;
    size_t batch, length = 0;
    bool valid = true;
    buffer keys, values;
};
struct prefix_state {
    std::shared_ptr<decoder_state> decoder;
    size_t length;
    buffer keys, values;
};
inline size_t rows(const model_state &model, size_t batch) {
    const auto &c = model.source.config;
    return multiply(multiply(size_t(c.text_layers), batch), size_t(c.text_heads));
}
inline size_t rowbytes(const model_state &model) {
    return multiply(size_t(model.source.config.audio_state / model.source.config.text_heads),
                    dtype_bytes(model.decoder->spec("self_keys").dtype));
}
}
