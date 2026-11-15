#ifndef LISTENFLOW_WHISPER_RUNTIME_H
#define LISTENFLOW_WHISPER_RUNTIME_H
#include <stddef.h>
#include <stdint.h>

#define LFFW_ABI_MAJOR 1
#define LFFW_API __attribute__((visibility("default")))
#if defined(__cplusplus)
#define LFFW_NOEXCEPT noexcept
extern "C" {
#else
#define LFFW_NOEXCEPT
#endif

typedef int32_t lffw_status;
enum {
    LFFW_SUCCESS = 0, LFFW_INVALID_ARGUMENT = 1, LFFW_ABI_MISMATCH = 2,
    LFFW_OUT_OF_MEMORY = 3, LFFW_INTERNAL_ERROR = 4, LFFW_CLOSED = 5,
    LFFW_CUDA_ERROR = 6, LFFW_SYSTEM_ERROR = 7, LFFW_INVALID_STATE = 8,
    LFFW_UNSUPPORTED_PRECISION = 9, LFFW_BUSY = 10,
    LFFW_CORRUPT_ENGINE = 100, LFFW_ENGINE_CONDITION_MISMATCH = 101,
    LFFW_ENGINE_IO_MISMATCH = 102, LFFW_PLUGIN_LIBRARY_ERROR = 110
};
typedef int32_t lffw_precision;
enum { LFFW_FP32 = 0, LFFW_FP16 = 1, LFFW_FP8 = 2 };
typedef int32_t lffw_dtype;
enum { LFFW_FLOAT32 = 0, LFFW_FLOAT16 = 1 };
typedef int32_t lffw_decoder_kind;
enum { LFFW_PREFIX = 0, LFFW_GREEDY = 1, LFFW_SAMPLED = 2 };
typedef struct lffw_model lffw_model;
typedef struct lffw_mel lffw_mel;
typedef struct lffw_encoded_audio lffw_encoded_audio;
typedef struct lffw_decoder lffw_decoder;
typedef struct lffw_prefix lffw_prefix;
typedef struct lffw_error lffw_error;

typedef struct lffw_version {
    size_t struct_size;
    uint32_t major, minor, patch;
} lffw_version;
typedef struct lffw_config {
    size_t struct_size;
    int64_t n_mels, audio_positions, audio_state, audio_heads, audio_ffn, audio_layers;
    int64_t text_heads, text_ffn, text_layers, text_positions, vocab_size;
} lffw_config;
typedef struct lffw_scale {
    size_t struct_size;
    const char *name;
    float value;
} lffw_scale;
typedef struct lffw_model_options {
    size_t struct_size;
    uint32_t abi_major;
    const lffw_config *config;
    lffw_precision precision;
    int32_t device;
    const char *const *weights;
    size_t weight_count;
    const char *config_file, *tokenizer_file, *scale_file, *mel_filter_file;
    const char *mel_weights, *fc1_library;
    const lffw_scale *const *scales;
    size_t scale_count;
    int32_t no_speech_token, timestamp_begin_token; /* -1 means absent */
    int32_t sampled_batch; /* 0 means no sampled engine; otherwise selected best_of > 1 */
    /* Borrowed token lists; open copies them, validates IDs, builds the normal
     * mask and its union with begin_suppressed_tokens, and uploads both once. */
    const int32_t *suppressed_tokens;
    size_t suppressed_token_count;
    const int32_t *begin_suppressed_tokens;
    size_t begin_suppressed_token_count;
} lffw_model_options;
typedef struct lffw_model_info {
    size_t struct_size;
    lffw_dtype decoder_dtype, encoder_dtype;
    int32_t sampled_batch;
    double mel_open_seconds, encoder_open_seconds, decoder_open_seconds, sampled_open_seconds;
    const char *mel_engine_path, *encoder_engine_path, *decoder_engine_path, *sampled_engine_path;
} lffw_model_info;
typedef struct lffw_selection {
    size_t struct_size;
    int32_t token;
    float logprob, no_speech_probability;
} lffw_selection;

/* Arrays/strings are borrowed for a synchronous call; open copies its options.
 * Error outputs are optional and return owned diagnostics, never TLS strings.
 * Every returned handle is owned by the caller. All create outputs are NULL on
 * failure. Host output contents are unspecified on failure. */
LFFW_API lffw_status lffw_get_version(lffw_version *, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_model_open(const lffw_model_options *, lffw_model **, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_model_get_info(const lffw_model *, lffw_model_info *, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_model_close(lffw_model *, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_model_destroy(lffw_model *, lffw_error **) LFFW_NOEXCEPT;

/* Short audio is <=480000 finite float32 samples. Native code pads and reflects.
 * Mel handles contain completed device float32[n_mels,3000] storage. */
LFFW_API lffw_status lffw_mel_compute(lffw_model *, const float *, size_t,
                           lffw_mel **, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_mel_copy(const lffw_mel *, float *, size_t, lffw_error **) LFFW_NOEXCEPT;
/* Long-form caller prepares one centered/padded 480400-sample block and supplies
 * n_mels*3000 float32 slots. Output is log_mel, before global clip/normalization. */
LFFW_API lffw_status lffw_log_mel_block(lffw_model *, const float *, size_t,
                             float *, size_t, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_mel_destroy(lffw_mel *, lffw_error **) LFFW_NOEXCEPT;

LFFW_API lffw_status lffw_encode(lffw_model *, const lffw_mel *,
                      lffw_encoded_audio **, lffw_error **) LFFW_NOEXCEPT;
/* Host features are float32[n_mels,2*audio_positions]. */
LFFW_API lffw_status lffw_encode_features(lffw_model *, const float *, size_t,
                               lffw_encoded_audio **, lffw_error **) LFFW_NOEXCEPT;
/* Encoder output has audio_positions*audio_state values in encoder_dtype. */
LFFW_API lffw_status lffw_encoded_copy(const lffw_encoded_audio *, void *, size_t,
                            lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_encoded_destroy(lffw_encoded_audio *, lffw_error **) LFFW_NOEXCEPT;

/* PREFIX/GREEDY require batch=1; SAMPLED requires the prepared sampled batch.
 * Opening a decoder creates chain state, not another engine/context set. */
LFFW_API lffw_status lffw_decoder_create(const lffw_encoded_audio *, lffw_decoder_kind, int32_t,
                              lffw_decoder **, lffw_error **) LFFW_NOEXCEPT;
/* Exactly one new token; NULL parent starts at the dummy cache position.
 * The returned prefix is immutable; the parent's KV is never overwritten.
 * Output logits have vocab_size values in decoder_dtype; capacity is in bytes. */
LFFW_API lffw_status lffw_prefix_step(lffw_decoder *, const lffw_prefix *, int32_t,
                           void *, size_t, lffw_prefix **, lffw_error **) LFFW_NOEXCEPT;
/* Mutable greedy chain: first and score are 0/1. first selects the resident
 * normal+begin mask until the first generated token has been selected, including
 * prompt prefill; otherwise it selects the resident normal mask. No host mask
 * is passed. score=0 returns token only, with both probability fields NaN. */
LFFW_API lffw_status lffw_greedy_step(lffw_decoder *, int32_t, uint32_t first, uint32_t score,
                           lffw_selection *, lffw_error **) LFFW_NOEXCEPT;
/* Mutable batch: one position across all B rows. suppression is float32[B,V];
 * temperature>0, uniform is float32[B] in [0,1), timestamps is 0/1. Outputs
 * use pointers so appending a field does not change the caller's array stride. */
LFFW_API lffw_status lffw_sampled_step(lffw_decoder *, const int32_t *, size_t,
                            const float *, size_t, float, const float *, size_t, uint32_t,
                            lffw_selection *const *, size_t, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_prefix_destroy(lffw_prefix *, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_decoder_destroy(lffw_decoder *, lffw_error **) LFFW_NOEXCEPT;

/* Diagnostic helpers do not allocate or initialize CUDA. NULL message returns
 * a fixed fallback string; destroy(NULL) succeeds. */
LFFW_API const char *lffw_error_message(const lffw_error *) LFFW_NOEXCEPT;
LFFW_API void lffw_error_destroy(lffw_error *) LFFW_NOEXCEPT;

#if defined(__cplusplus)
}
#endif
#endif
