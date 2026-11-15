#ifndef LISTENFLOW_WHISPER_RUNTIME_DIAGNOSTICS_H
#define LISTENFLOW_WHISPER_RUNTIME_DIAGNOSTICS_H
#include <listenflow/whisper_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct lffw_diagnostic_session lffw_diagnostic_session;
/* Development builds only. A session observes synchronous C ABI calls on the
 * thread calling begin; it never owns a product handle. Getter strings are
 * borrowed until the next begin or destroy. Python bindings copy them. */
enum {
    LFFW_DIAGNOSTIC_HOST = 1,
    LFFW_DIAGNOSTIC_EVENTS = 2,
    LFFW_DIAGNOSTIC_WAIT = 4
};
typedef struct lffw_diagnostic_options {
    size_t struct_size;
    uint32_t flags;
} lffw_diagnostic_options;
typedef struct lffw_diagnostic_counts {
    size_t struct_size;
    size_t operation_count;
} lffw_diagnostic_counts;
typedef struct lffw_diagnostic_operation {
    size_t struct_size;
    uint64_t sequence;
    const char *name;
    int32_t status;
    int32_t decoder_kind;
    int32_t input_token, output_token;
    uint32_t token_valid; /* bit 0: input token; bit 1: output token */
    uint64_t start_ns, end_ns;
    size_t span_count, sleep_count;
} lffw_diagnostic_operation;
typedef struct lffw_diagnostic_span {
    size_t struct_size;
    const char *phase, *graph, *tensor;
    uint64_t start_ns, end_ns, cpu_ns;
    double gpu_ms;
    uint32_t gpu_time_valid; /* zero means no timing event measurement */
    uint32_t graph_replay, graph_capture;
    size_t bucket, bytes, query_count;
} lffw_diagnostic_span;
typedef struct lffw_diagnostic_sleep {
    size_t struct_size;
    uint64_t start_ns, end_ns, requested_ns;
} lffw_diagnostic_sleep;
LFFW_API lffw_status lffw_diagnostic_create(const lffw_diagnostic_options *, lffw_diagnostic_session **, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_diagnostic_begin(lffw_diagnostic_session *, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_diagnostic_end(lffw_diagnostic_session *, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_diagnostic_get_counts(const lffw_diagnostic_session *, lffw_diagnostic_counts *, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_diagnostic_get_operation(const lffw_diagnostic_session *, size_t, lffw_diagnostic_operation *, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_diagnostic_get_span(const lffw_diagnostic_session *, size_t, size_t, lffw_diagnostic_span *, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_diagnostic_get_sleep(const lffw_diagnostic_session *, size_t, size_t, lffw_diagnostic_sleep *, lffw_error **) LFFW_NOEXCEPT;
LFFW_API lffw_status lffw_diagnostic_destroy(lffw_diagnostic_session *, lffw_error **) LFFW_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#endif
