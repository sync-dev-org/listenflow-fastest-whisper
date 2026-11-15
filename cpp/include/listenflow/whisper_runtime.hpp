#pragma once
#include "whisper_runtime.h"
#include <memory>
#include <stdexcept>
#include <utility>

namespace listenflow::fastest_whisper::runtime {
class error : public std::runtime_error {
  public:
    error(lffw_status status, const char *message) : std::runtime_error(message), status_(status) {}
    lffw_status status() const noexcept { return status_; }
  private:
    lffw_status status_;
};
struct error_deleter { void operator()(lffw_error *p) const noexcept { lffw_error_destroy(p); } };
inline void check(lffw_status status, lffw_error *diagnostic) {
    std::unique_ptr<lffw_error, error_deleter> owner(diagnostic);
    if (status != LFFW_SUCCESS) throw error(status, lffw_error_message(diagnostic));
}
template <class Function, class... Args>
void invoke(Function function, Args &&...args) {
    lffw_error *diagnostic = nullptr;
    auto status = function(std::forward<Args>(args)..., &diagnostic);
    check(status, diagnostic);
}
template <class T, lffw_status (*Release)(T *, lffw_error **) noexcept>
struct deleter { void operator()(T *p) const noexcept { (void)Release(p, nullptr); } };
template <class T, lffw_status (*Release)(T *, lffw_error **) noexcept>
using owned = std::unique_ptr<T, deleter<T, Release>>;
using mel = owned<lffw_mel, lffw_mel_destroy>;
using encoded_audio = owned<lffw_encoded_audio, lffw_encoded_destroy>;
using decoder = owned<lffw_decoder, lffw_decoder_destroy>;
using prefix = owned<lffw_prefix, lffw_prefix_destroy>;
template <class T, lffw_status (*Release)(T *, lffw_error **) noexcept>
void close(owned<T, Release> &handle) {
    if (!handle) return;
    lffw_error *diagnostic = nullptr;
    auto status = Release(handle.get(), &diagnostic);
    if (status != LFFW_BUSY) (void)handle.release();
    check(status, diagnostic);
}
inline lffw_version version() {
    lffw_version result{}; result.struct_size = sizeof(result);
    invoke(lffw_get_version, &result); return result;
}

class model {
  public:
    explicit model(const lffw_model_options &options) {
        lffw_model *result = nullptr;
        lffw_error *diagnostic = nullptr;
        auto status = lffw_model_open(&options, &result, &diagnostic);
        owned<lffw_model, lffw_model_destroy> pending(result);
        check(status, diagnostic);
        handle_ = std::move(pending);
    }
    model(model &&) noexcept = default;
    model &operator=(model &&) noexcept = default;
    lffw_model *get() const noexcept { return handle_.get(); }
    void close() { invoke(lffw_model_close, get()); }
    lffw_model_info info() const {
        lffw_model_info result{};
        result.struct_size = sizeof(result);
        invoke(lffw_model_get_info, get(), &result);
        return result;
    }
    runtime::mel mel(const float *samples, size_t count) {
        lffw_mel *result = nullptr;
        invoke(lffw_mel_compute, get(), samples, count, &result);
        return runtime::mel(result);
    }
    void log_mel_block(const float *samples, size_t count, float *out, size_t capacity) {
        invoke(lffw_log_mel_block, get(), samples, count, out, capacity);
    }
    encoded_audio encode(const runtime::mel &features) {
        lffw_encoded_audio *result = nullptr;
        invoke(lffw_encode, get(), features.get(), &result);
        return encoded_audio(result);
    }
    encoded_audio encode_features(const float *features, size_t count) {
        lffw_encoded_audio *result = nullptr;
        invoke(lffw_encode_features, get(), features, count, &result);
        return encoded_audio(result);
    }
  private:
    owned<lffw_model, lffw_model_destroy> handle_;
};
inline decoder create_decoder(const encoded_audio &audio, lffw_decoder_kind kind, int32_t batch) {
    lffw_decoder *result = nullptr;
    invoke(lffw_decoder_create, audio.get(), kind, batch, &result);
    return decoder(result);
}
inline prefix prefix_step(decoder &chain, const prefix *parent, int32_t token,
                          void *logits, size_t capacity) {
    lffw_prefix *result = nullptr;
    invoke(lffw_prefix_step, chain.get(), parent ? parent->get() : nullptr,
           token, logits, capacity, &result);
    return prefix(result);
}
inline lffw_selection greedy_step(decoder &chain, int32_t token, bool first, bool score) {
    lffw_selection result{};
    result.struct_size = sizeof(result);
    invoke(lffw_greedy_step, chain.get(), token, uint32_t(first), uint32_t(score), &result);
    return result;
}
inline void sampled_step(decoder &chain, const int32_t *tokens, size_t batch,
                         const float *suppression, size_t elements, float temperature,
                         const float *uniform, bool timestamps, lffw_selection *const *out) {
    invoke(lffw_sampled_step, chain.get(), tokens, batch, suppression, elements, temperature,
           uniform, batch, uint32_t(timestamps), out, batch);
}
inline void copy(const mel &value, float *out, size_t elements) {
    invoke(lffw_mel_copy, value.get(), out, elements);
}
inline void copy(const encoded_audio &value, void *out, size_t bytes) {
    invoke(lffw_encoded_copy, value.get(), out, bytes);
}
}
