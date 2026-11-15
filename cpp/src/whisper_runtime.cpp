#include <listenflow/whisper_runtime.h>
#include "runtime_model.hpp"
#include "model_error.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include "runtime_error.hpp"
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
#include "runtime_diagnostics.hpp"
#define LFFW_DIAGNOSTIC_OPERATION(name) listenflow::fastest_whisper::diagnostic::operation diagnostic_operation(name)
#else
#define LFFW_DIAGNOSTIC_OPERATION(name)
#endif

namespace rd = listenflow::fastest_whisper::runtime_detail;
namespace fw = listenflow::fastest_whisper;
struct lffw_model { std::shared_ptr<rd::model_state> state; };
struct lffw_mel { std::shared_ptr<rd::model_state> model; rd::buffer value; };
struct lffw_encoded_audio { std::shared_ptr<rd::audio_state> state; };
struct lffw_decoder { std::shared_ptr<rd::decoder_state> state; };
struct lffw_prefix { std::shared_ptr<rd::prefix_state> state; };

namespace {
lffw_error emergency;
lffw_status status_of(std::exception_ptr failure) noexcept {
    try { std::rethrow_exception(failure); }
    catch (const rd::wait_restore_error &error) { return status_of(error.primary); }
    catch (const rd::cleanup_error &error) { return status_of(error.primary); }
    catch (const rd::failure &error) { return error.status; }
    catch (const fw::detail::cuda_error &error) {
        return error.status() == TENSORACLE_OUT_OF_MEMORY ? LFFW_OUT_OF_MEMORY : LFFW_CUDA_ERROR;
    }
    catch (const fw::model_error &error) { return error.status(); }
    catch (const std::bad_alloc &) { return LFFW_OUT_OF_MEMORY; }
    catch (const std::invalid_argument &) { return LFFW_INVALID_ARGUMENT; }
    catch (const std::system_error &) { return LFFW_SYSTEM_ERROR; }
    catch (...) { return LFFW_INTERNAL_ERROR; }
}
lffw_status fail(lffw_status status, const char *message, lffw_error **error) noexcept {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
    listenflow::fastest_whisper::diagnostic::status(status);
#endif
    if (error) {
        try { *error = new lffw_error{message}; }
        catch (...) { *error = &emergency; return LFFW_OUT_OF_MEMORY; }
    }
    return status;
}
template <class Operation> lffw_status protect(lffw_error **error, Operation operation) noexcept {
    if (error) *error = nullptr;
    try { operation(); return LFFW_SUCCESS; }
    catch (const std::exception &failure) { return fail(status_of(std::current_exception()), failure.what(), error); }
    catch (...) { return fail(LFFW_INTERNAL_ERROR, "unknown runtime failure", error); }
}
template <class T, class Value> void field(T *out, size_t offset, Value value) {
    if (offset <= out->struct_size && sizeof(value) <= out->struct_size - offset)
        std::memcpy(reinterpret_cast<char *>(out) + offset, &value, sizeof(value));
}
template <class Operation>
void locked(const std::shared_ptr<rd::model_state> &model, Operation operation,
            bool allow_closed = false, bool gpu = true) {
    rd::require(bool(model), "model handle is required");
    std::unique_lock<std::mutex> lock(model->mutex, std::try_to_lock);
    if (!lock.owns_lock()) throw rd::failure(LFFW_BUSY, "another operation owns this model");
    if (model->closed && !allow_closed) throw rd::failure(LFFW_CLOSED, "Whisper model is closed");
    rd::context_guard active(gpu && !model->closed ? model->context : nullptr);
    try { operation(); }
    catch (...) {
        auto primary = std::current_exception();
        if (model->queue && model->queue->pending) {
            try { model->queue->drain(); }
            catch (const std::exception &cleanup) {
                std::string message = "native operation failed";
                try { std::rethrow_exception(primary); }
                catch (const std::exception &error) { message = error.what(); }
                catch (...) {}
                throw rd::failure(status_of(primary), message + "; cleanup also failed: " + cleanup.what());
            }
        }
        std::rethrow_exception(primary);
    }
}
template <class T, class Owner>
void destroy_child(T *handle, Owner owner) {
    if (!handle) return;
    auto model = owner(handle);
    locked(model, [&] { delete handle; }, true, false);
}
std::shared_ptr<rd::model_state> owner(const lffw_decoder *handle) {
    rd::require(handle && handle->state, "decoder handle is required"); return handle->state->audio->model;
}
void token(const rd::model_state &model, int32_t value) {
    rd::require(value >= 0 && value < model.source.config.vocab_size, "token is outside model vocabulary");
}
void output_selection(lffw_selection *out, int32_t token, float logprob, float speech) {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
    fw::diagnostic::output(token);
#endif
    rd::require(out && out->struct_size >= sizeof(size_t), "selection output requires struct_size");
    field(out, offsetof(lffw_selection, token), token);
    field(out, offsetof(lffw_selection, logprob), logprob);
    field(out, offsetof(lffw_selection, no_speech_probability), speech);
}
void read_waveform(const float *samples, size_t count) {
    rd::require(!count || samples, "waveform storage is required");
    for (size_t i = 0; i < count; ++i) rd::require(std::isfinite(samples[i]), "waveform must contain finite samples");
}
std::shared_ptr<rd::audio_state> encode(rd::model_state &model, const std::shared_ptr<rd::model_state> &owner,
                                        const std::map<std::string, CUdeviceptr> &inputs) {
    auto result = std::make_shared<rd::audio_state>(); result->model = owner;
    result->encoder = model.pool->acquire(model.encoder->capacity("encoder"));
    result->keys = model.pool->acquire(model.encoder->capacity("cross_keys"));
    result->values = model.pool->acquire(model.encoder->capacity("cross_values"));
    auto addresses = inputs;
    addresses["encoder"] = result->encoder->address;
    addresses["cross_keys"] = result->keys->address; addresses["cross_values"] = result->values->address;
    model.encoder->run(addresses); model.queue->finish(); return result;
}
std::map<std::string, CUdeviceptr> decoder_inputs(rd::decoder_state &chain,
                                                  rd::executor &executor, const int32_t *tokens,
                                                  size_t position, size_t history, size_t bucket) {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
    fw::diagnostic::span observed("cache_prepare", "decoder", bucket);
    fw::diagnostic::input(tokens[0], chain.kind);
#endif
    std::vector<int32_t> positions(chain.batch, int32_t(position));
    std::vector<float> mask(bucket + 1, -std::numeric_limits<float>::infinity());
    std::fill(mask.begin() + 1, mask.begin() + history, 0);
    mask.back() = 0;
    executor.upload("tokens", tokens, chain.batch * sizeof(int32_t));
    executor.upload("positions", positions.data(), positions.size() * sizeof(int32_t));
    executor.upload("mask", mask.data(), mask.size() * sizeof(float));
    return {{"cross_keys", chain.audio->keys->address}, {"cross_values", chain.audio->values->address}};
}
void validate_chain(const rd::decoder_state &chain, lffw_decoder_kind kind) {
    rd::require(chain.kind == kind, "decoder kind does not match operation");
    if (!chain.valid) throw rd::failure(LFFW_INVALID_STATE, "decoder failed; create a new decoding chain");
    rd::require(chain.length < size_t(chain.audio->model->source.config.text_positions), "decoder exceeds checkpoint context length");
}
template <class Operation> void mutable_step(rd::decoder_state &chain, Operation operation) {
    try { operation(); ++chain.length; }
    catch (...) { chain.valid = false; throw; }
}
void append_mutable(rd::decoder_state &chain, rd::executor &executor, size_t history, size_t bucket) {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
    fw::diagnostic::span observed("cache_append", "decoder", bucket);
#endif
    auto &model = *chain.audio->model;
    if (history >= size_t(model.source.config.text_positions)) return;
    auto copies = rd::cache_transfer(*model.queue);
    rd::advance_cache(copies, chain.keys->address, executor.storage("present_keys")->address,
                      history, bucket, rd::rows(model, chain.batch), rd::rowbytes(model));
    rd::advance_cache(copies, chain.values->address, executor.storage("present_values")->address,
                      history, bucket, rd::rows(model, chain.batch), rd::rowbytes(model));
}
}

extern "C" const char *lffw_error_message(const lffw_error *error) noexcept {
    if (!error) return "runtime operation failed without a diagnostic";
    if (error == &emergency) return "could not allocate runtime error diagnostic";
    return error->message.c_str();
}
extern "C" void lffw_error_destroy(lffw_error *error) noexcept { if (error != &emergency) delete error; }
extern "C" lffw_status lffw_get_version(lffw_version *out, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_get_version");
    return protect(error, [&] {
        rd::require(out && out->struct_size >= sizeof(size_t), "version output requires struct_size");
        field(out, offsetof(lffw_version, major), uint32_t(LFFW_ABI_MAJOR));
        field(out, offsetof(lffw_version, minor), uint32_t(0)); field(out, offsetof(lffw_version, patch), uint32_t(0));
    });
}
extern "C" lffw_status lffw_model_open(const lffw_model_options *options, lffw_model **out, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_model_open");
    if (out) *out = nullptr;
    return protect(error, [&] {
        rd::require(out, "model output is required");
        auto result = std::make_unique<lffw_model>();
        result->state = std::make_shared<rd::model_state>(options);
        result->state->initialize(); *out = result.release();
    });
}
extern "C" lffw_status lffw_model_get_info(const lffw_model *handle, lffw_model_info *out, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_model_get_info");
    return protect(error, [&] {
        rd::require(handle && out && out->struct_size >= sizeof(size_t), "model and info output are required");
        locked(handle->state, [&] {
            auto &model = *handle->state;
            field(out, offsetof(lffw_model_info, encoder_dtype), model.encoder_type);
            field(out, offsetof(lffw_model_info, decoder_dtype), model.decoder_type);
            field(out, offsetof(lffw_model_info, sampled_batch), model.sampled_batch);
            field(out, offsetof(lffw_model_info, mel_open_seconds), model.seconds[0]);
            field(out, offsetof(lffw_model_info, encoder_open_seconds), model.seconds[1]);
            field(out, offsetof(lffw_model_info, decoder_open_seconds), model.seconds[2]);
            field(out, offsetof(lffw_model_info, sampled_open_seconds), model.seconds[3]);
            field(out, offsetof(lffw_model_info, mel_engine_path), model.paths[0].c_str());
            field(out, offsetof(lffw_model_info, encoder_engine_path), model.paths[1].c_str());
            field(out, offsetof(lffw_model_info, decoder_engine_path), model.paths[2].c_str());
            field(out, offsetof(lffw_model_info, sampled_engine_path), model.sampled_batch ? model.paths[3].c_str() : nullptr);
        }, true, false);
    });
}
extern "C" lffw_status lffw_model_close(lffw_model *handle, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_model_close");
    return protect(error, [&] {
        rd::require(handle, "model handle is required");
        locked(handle->state, [&] { handle->state->close(); }, true, false);
    });
}
extern "C" lffw_status lffw_model_destroy(lffw_model *handle, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_model_destroy");
    if (!handle) { if (error) *error = nullptr; return LFFW_SUCCESS; }
    auto status = lffw_model_close(handle, error);
    if (status != LFFW_BUSY) delete handle;
    return status;
}
extern "C" lffw_status lffw_mel_compute(lffw_model *handle, const float *samples, size_t count,
                                       lffw_mel **out, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_mel_compute");
    if (out) *out = nullptr;
    return protect(error, [&] {
        rd::require(handle && out, "model and mel output are required");
        rd::require(count <= 480000, "waveform exceeds 30 seconds"); read_waveform(samples, count);
        locked(handle->state, [&] {
            auto &model = *handle->state;
            auto result = std::make_unique<lffw_mel>(); result->model = handle->state;
            result->value = model.pool->acquire(model.mel->capacity("mel"));
            std::vector<float> centered(480400, 0);
            if (count) std::memcpy(centered.data() + 200, samples, count * sizeof(float));
            for (size_t i = 0; i < 200; ++i) { centered[i] = centered[400 - i]; centered[480200 + i] = centered[480198 - i]; }
            model.mel->upload("audio", centered.data(), centered.size() * sizeof(float));
            model.mel->run({{"mel", result->value->address}}); model.queue->finish(); *out = result.release();
        });
    });
}
extern "C" lffw_status lffw_mel_copy(const lffw_mel *handle, float *out, size_t count, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_mel_copy");
    return protect(error, [&] {
        rd::require(handle && out, "mel and host output are required");
        locked(handle->model, [&] {
            auto &model = *handle->model; auto bytes = model.mel->capacity("mel");
            rd::require(count >= bytes / sizeof(float), "mel host output is too small");
            model.mel->download("copy", handle->value->address, bytes); model.queue->finish(); model.mel->copy_host("copy", out, bytes);
        });
    });
}
extern "C" lffw_status lffw_log_mel_block(lffw_model *handle, const float *samples, size_t count,
                                        float *out, size_t capacity, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_log_mel_block");
    return protect(error, [&] {
        rd::require(handle && out && count == 480400, "centered mel block requires 480400 samples and host output");
        read_waveform(samples, count);
        locked(handle->state, [&] {
            auto &model = *handle->state; auto &mel = *model.mel; auto bytes = mel.capacity("log_mel");
            rd::require(capacity >= bytes / sizeof(float), "log mel host output is too small");
            mel.upload("audio", samples, count * sizeof(float)); mel.run();
            mel.download("log_mel", mel.storage("log_mel")->address, bytes); model.queue->finish(); mel.copy_host("log_mel", out, bytes);
        });
    });
}
extern "C" lffw_status lffw_mel_destroy(lffw_mel *handle, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_mel_destroy");
    return protect(error, [&] { destroy_child(handle, [](auto value) { return value->model; }); });
}
extern "C" lffw_status lffw_encode(lffw_model *handle, const lffw_mel *features,
                                   lffw_encoded_audio **out, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_encode");
    if (out) *out = nullptr;
    return protect(error, [&] {
        rd::require(handle && features && out && features->model == handle->state, "encoder requires mel from the same model");
        locked(handle->state, [&] {
            auto result = std::make_unique<lffw_encoded_audio>();
            result->state = encode(*handle->state, handle->state, {{"mel", features->value->address}}); *out = result.release();
        });
    });
}
extern "C" lffw_status lffw_encode_features(lffw_model *handle, const float *features, size_t count,
                                            lffw_encoded_audio **out, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_encode_features");
    if (out) *out = nullptr;
    return protect(error, [&] {
        rd::require(handle && features && out, "model, host mel and encoder output are required");
        locked(handle->state, [&] {
            auto &model = *handle->state;
            rd::require(count == model.encoder->capacity("mel") / sizeof(float), "mel must match model feature shape");
            auto result = std::make_unique<lffw_encoded_audio>();
            model.encoder->upload("mel", features, count * sizeof(float));
            result->state = encode(model, handle->state, {}); *out = result.release();
        });
    });
}
extern "C" lffw_status lffw_encoded_copy(const lffw_encoded_audio *handle, void *out, size_t capacity,
                                        lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_encoded_copy");
    return protect(error, [&] {
        rd::require(handle && out, "encoded audio and host output are required"); auto model = handle->state->model;
        locked(model, [&] {
            auto bytes = model->encoder->capacity("encoder"); rd::require(capacity >= bytes, "encoded host output is too small");
            model->encoder->download("copy", handle->state->encoder->address, bytes); model->queue->finish();
            model->encoder->copy_host("copy", out, bytes);
        });
    });
}
extern "C" lffw_status lffw_encoded_destroy(lffw_encoded_audio *handle, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_encoded_destroy");
    return protect(error, [&] { destroy_child(handle, [](auto value) { return value->state->model; }); });
}
extern "C" lffw_status lffw_decoder_create(const lffw_encoded_audio *audio, lffw_decoder_kind kind,
                                           int32_t batch, lffw_decoder **out, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_decoder_create");
    if (out) *out = nullptr;
    return protect(error, [&] {
        rd::require(audio && out, "encoded audio and decoder output are required");
        auto model = audio->state->model;
        locked(model, [&] {
            rd::require(kind >= LFFW_PREFIX && kind <= LFFW_SAMPLED, "unknown decoder kind");
            rd::require(kind == LFFW_SAMPLED ? model->sampled_batch && batch == model->sampled_batch : batch == 1,
                        "decoder batch differs from prepared engine");
            auto result = std::make_unique<lffw_decoder>(); result->state = std::make_shared<rd::decoder_state>();
            auto &chain = *result->state;
            chain.audio = audio->state; chain.kind = kind; chain.batch = size_t(batch);
            if (kind != LFFW_PREFIX) {
                auto bytes = rd::multiply(rd::multiply(rd::rows(*model, chain.batch), size_t(model->source.config.text_positions + 1)), rd::rowbytes(*model));
                chain.keys = model->pool->acquire(bytes); chain.values = model->pool->acquire(bytes);
                auto dummy = rd::rows(*model, chain.batch) * rd::rowbytes(*model);
                model->queue->zero(chain.keys->address, dummy); model->queue->zero(chain.values->address, dummy);
                model->queue->finish();
            }
            *out = result.release();
        });
    });
}
extern "C" lffw_status lffw_prefix_step(lffw_decoder *handle, const lffw_prefix *parent, int32_t value,
                                        void *logits, size_t capacity, lffw_prefix **out, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_prefix_step");
    if (out) *out = nullptr;
    return protect(error, [&] {
        auto model = owner(handle);
        rd::require(out && logits, "prefix and logits outputs are required");
        locked(model, [&] {
            auto &chain = *handle->state; validate_chain(chain, LFFW_PREFIX); token(*model, value);
            rd::require(!parent || parent->state->decoder == handle->state, "prefix belongs to another decoder");
            auto position = parent ? parent->state->length : 0;
            rd::require(position < size_t(model->source.config.text_positions), "prefix exceeds checkpoint context length");
            auto &executor = *model->decoder;
            auto bytes = size_t(model->source.config.vocab_size) * rd::dtype_bytes(executor.spec("logits").dtype);
            rd::require(capacity >= bytes, "logits host output is too small");
            const auto history = position + 1, bucket = rd::cache_bucket(history);
            const auto rows = rd::rows(*model, 1), rowbytes = rd::rowbytes(*model);
            auto result = std::make_unique<lffw_prefix>(); result->state = std::make_shared<rd::prefix_state>();
            result->state->decoder = handle->state; result->state->length = position + 1;
            // Geometric capacities bound idle KV storage by the longest prefix,
            // rather than retaining a different allocation for every length.
            size_t capacity_positions = 2;
            while (capacity_positions < history + 1) capacity_positions *= 2;
            auto prefix_bytes = rd::multiply(rd::multiply(rows, capacity_positions), rowbytes);
            result->state->keys = model->pool->acquire(prefix_bytes);
            result->state->values = model->pool->acquire(prefix_bytes);
            auto addresses = decoder_inputs(chain, executor, &value, position, history, bucket);
            for (auto name : {"self_keys", "self_values"}) {
                auto target = executor.storage(name); model->queue->zero(target->address, rows * bucket * rowbytes);
                if (parent) {
                    auto source = std::string(name) == "self_keys" ? parent->state->keys : parent->state->values;
                    model->queue->copy_rows(target->address, bucket * rowbytes, source->address, history * rowbytes,
                                             history * rowbytes, rows);
                }
            }
            if (executor.suppression_content != 2) {
                model->queue->zero(executor.storage("suppression")->address, executor.capacity("suppression"));
                executor.suppression_content = 2;
            }
            executor.run(addresses, bucket);
            auto copies = rd::cache_transfer(*model->queue);
            rd::compact_cache(copies, result->state->keys->address, executor.storage("present_keys")->address,
                              history, bucket, rows, rowbytes);
            rd::compact_cache(copies, result->state->values->address, executor.storage("present_values")->address,
                              history, bucket, rows, rowbytes);
            executor.download("logits", executor.storage("logits")->address, bytes);
            model->queue->finish(); executor.copy_host("logits", logits, bytes); *out = result.release();
        });
    });
}
extern "C" lffw_status lffw_greedy_step(lffw_decoder *handle, int32_t value, uint32_t first, uint32_t score,
                                        lffw_selection *out, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_greedy_step");
    return protect(error, [&] {
        auto model = owner(handle);
        locked(model, [&] {
            auto &chain = *handle->state; validate_chain(chain, LFFW_GREEDY); token(*model, value);
            rd::require(first <= 1 && score <= 1, "first and score flags must be zero or one");
            rd::require(out && out->struct_size >= sizeof(size_t), "selection output requires struct_size");
            const auto history = chain.length + 1, bucket = rd::cache_bucket(history);
            auto &executor = *model->decoder;
            mutable_step(chain, [&] {
                auto addresses = decoder_inputs(chain, executor, &value, chain.length, history, bucket);
                addresses["self_keys"] = chain.keys->address; addresses["self_values"] = chain.values->address;
                if (executor.suppression_content != int(first)) {
                    model->queue->copy(executor.storage("suppression")->address, model->suppression[first]->address,
                                        executor.capacity("suppression")); executor.suppression_content = int(first);
                }
                executor.run(addresses, bucket); append_mutable(chain, executor, history, bucket);
                auto name = score ? "greedy" : "token";
                executor.download(name, executor.storage(name)->address, score ? 3 * sizeof(float) : sizeof(int32_t));
                model->queue->finish();
                if (score) {
                    float result[3]; executor.copy_host(name, result, sizeof(result));
                    output_selection(out, int32_t(result[0]), result[1], result[2]);
                } else {
                    int32_t result; executor.copy_host(name, &result, sizeof(result));
                    output_selection(out, result, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
                }
            });
        });
    });
}
extern "C" lffw_status lffw_sampled_step(lffw_decoder *handle, const int32_t *tokens, size_t batch,
                                         const float *suppression, size_t elements, float temperature,
                                         const float *uniform, size_t uniforms, uint32_t timestamps,
                                         lffw_selection *const *out, size_t count, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_sampled_step");
    return protect(error, [&] {
        auto model = owner(handle);
        locked(model, [&] {
            auto &chain = *handle->state; validate_chain(chain, LFFW_SAMPLED);
            rd::require(tokens && suppression && uniform && out && batch == chain.batch && uniforms == batch && count == batch,
                        "sampled inputs and outputs must match the prepared batch");
            rd::require(elements == rd::multiply(batch, size_t(model->source.config.vocab_size)), "sampled suppression has wrong shape");
            rd::require(std::isfinite(temperature) && temperature > 0 && timestamps <= 1, "invalid sampling temperature or timestamps flag");
            for (size_t row = 0; row < batch; ++row) {
                token(*model, tokens[row]);
                rd::require(std::isfinite(uniform[row]) && uniform[row] >= 0 && uniform[row] < 1, "sampling uniform must be in [0,1)");
                rd::require(out[row] && out[row]->struct_size >= sizeof(size_t), "sampled output requires struct_size");
            }
            for (size_t i = 0; i < elements; ++i)
                rd::require(!std::isnan(suppression[i]) && suppression[i] != std::numeric_limits<float>::infinity(), "invalid sampled suppression");
            const auto history = chain.length + 1, bucket = rd::cache_bucket(history);
            auto &executor = *model->sampled;
            mutable_step(chain, [&] {
                auto addresses = decoder_inputs(chain, executor, tokens, chain.length, history, bucket);
                addresses["self_keys"] = chain.keys->address; addresses["self_values"] = chain.values->address;
                executor.upload("suppression", suppression, elements * sizeof(float));
                executor.upload("temperature", &temperature, sizeof(temperature));
                executor.upload("uniform", uniform, batch * sizeof(float));
                uint8_t flag = uint8_t(timestamps); executor.upload("timestamps", &flag, sizeof(flag));
                executor.run(addresses, bucket); append_mutable(chain, executor, history, bucket);
                std::vector<float> results(batch * 3);
                executor.download("sampled", executor.storage("sampled")->address, results.size() * sizeof(float));
                model->queue->finish(); executor.copy_host("sampled", results.data(), results.size() * sizeof(float));
                for (size_t row = 0; row < batch; ++row)
                    output_selection(out[row], int32_t(results[row * 3]), results[row * 3 + 1], results[row * 3 + 2]);
            });
        });
    });
}
extern "C" lffw_status lffw_prefix_destroy(lffw_prefix *handle, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_prefix_destroy");
    return protect(error, [&] { destroy_child(handle, [](auto value) { return value->state->decoder->audio->model; }); });
}
extern "C" lffw_status lffw_decoder_destroy(lffw_decoder *handle, lffw_error **error) noexcept {
    LFFW_DIAGNOSTIC_OPERATION("lffw_decoder_destroy");
    return protect(error, [&] { destroy_child(handle, [](auto value) { return value->state->audio->model; }); });
}
