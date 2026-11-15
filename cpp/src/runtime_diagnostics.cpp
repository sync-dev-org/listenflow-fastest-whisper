#include "runtime_diagnostics.hpp"
#include "runtime_error.hpp"
#include <cstring>
#include <stdexcept>
#include <thread>

namespace diag = listenflow::fastest_whisper::diagnostic;
struct lffw_diagnostic_session {
    std::shared_ptr<diag::state> value;
    std::thread::id thread;
    bool bound = false, active = false;
};
namespace listenflow::fastest_whisper::diagnostic {
thread_local std::weak_ptr<state> selected;
thread_local std::shared_ptr<state> recording;
thread_local std::shared_ptr<operation_record> current;
thread_local size_t wait_index = SIZE_MAX;
void failed() noexcept { if (recording) recording->failed = true; }
void status(int32_t code) noexcept { if (current) current->value.status = code; }
void input(int32_t token, int32_t kind) noexcept {
    if (current) { current->value.input_token = token; current->value.decoder_kind = kind; current->value.token_valid |= 1; }
}
void output(int32_t token) noexcept {
    if (current) { current->value.output_token = token; current->value.token_valid |= 2; }
}
void query() noexcept {
    if (current && wait_index < current->spans.size()) ++current->spans[wait_index].value.query_count;
}
operation::operation(const char *name) noexcept {
    if (current) return;
    recording = selected.lock();
    if (!recording) return;
    try {
        // Bound retained diagnostic data even when a caller never ends a run.
        if (recording->operations.size() >= 65536) { failed(); recording.reset(); return; }
        auto value = std::make_shared<operation_record>();
        value->name = name; value->value.sequence = recording->operations.size();
        value->value.start_ns = now(); value->value.decoder_kind = -1;
        recording->operations.push_back(value);
        current = std::move(value); root_ = true;
    } catch (...) { failed(); recording.reset(); }
}
operation::~operation() {
    if (!root_) return;
    current->value.end_ns = now(); current.reset(); recording.reset(); wait_index = SIZE_MAX;
}
span::span(const char *phase, const char *graph, size_t bucket, const char *tensor, size_t bytes) noexcept {
    if (!current || !recording) return;
    try {
        if (current->spans.size() >= 65536) { failed(); return; }
        owner = current; index = owner->spans.size();
        span_record result; result.phase = phase; result.graph = graph; result.tensor = tensor;
        result.value.start_ns = now(); result.value.bucket = bucket; result.value.bytes = bytes;
        cpu_ = now(CLOCK_THREAD_CPUTIME_ID);
        owner->spans.push_back(std::move(result));
        if (std::strcmp(phase, "wait") == 0) { previous_wait_ = wait_index; wait_index = index; }
    } catch (...) { owner.reset(); index = SIZE_MAX; failed(); }
}
span::~span() {
    if (!owner || index >= owner->spans.size()) return;
    auto &result = owner->spans[index].value;
    result.end_ns = now(); result.cpu_ns = now(CLOCK_THREAD_CPUTIME_ID) - cpu_;
    if (owner->spans[index].phase == "wait") wait_index = previous_wait_;
}
sleep::~sleep() {
    if (!current || !recording || !(recording->flags & LFFW_DIAGNOSTIC_WAIT)) return;
    try {
        if (current->sleeps.size() >= 65536) { failed(); return; }
        current->sleeps.push_back({sizeof(lffw_diagnostic_sleep), started_, now(), requested_});
    } catch (...) { failed(); }
}
}
namespace {
template<class Function> lffw_status protect(lffw_error **error, Function function) noexcept {
    if (error) *error = nullptr;
    try { function(); return LFFW_SUCCESS; }
    catch (lffw_status status) {
        if (error) {
            try { *error = new lffw_error{status == LFFW_BUSY ? "another diagnostic session is active" : "diagnostic collection is incomplete"}; }
            catch (...) { return LFFW_OUT_OF_MEMORY; }
        }
        return status;
    }
    catch (const std::exception &failure) {
        if (error) { try { *error = new lffw_error{failure.what()}; } catch (...) {} }
        return dynamic_cast<const std::bad_alloc *>(&failure) ? LFFW_OUT_OF_MEMORY : LFFW_INVALID_ARGUMENT;
    } catch (...) { return LFFW_INTERNAL_ERROR; }
}
void require(bool condition, const char *message) { if (!condition) throw std::invalid_argument(message); }
void thread(const lffw_diagnostic_session *session) {
    require(session && session->bound && session->thread == std::this_thread::get_id(), "diagnostic session belongs to another thread or has not begun");
}
const diag::state &snapshot(const lffw_diagnostic_session *session) {
    thread(session); require(!session->active, "end diagnostic recording before reading it");
    if (session->value->failed) throw lffw_status(LFFW_INTERNAL_ERROR);
    return *session->value;
}
template<class T> void write(T *out, const T &value) {
    require(out && out->struct_size >= sizeof(T), "diagnostic output requires all known fields");
    std::memcpy(out, &value, sizeof(T)); out->struct_size = sizeof(T);
}
const diag::operation_record &operation(const lffw_diagnostic_session *session, size_t index) {
    const auto &value = snapshot(session); require(index < value.operations.size(), "diagnostic operation index is out of range");
    return *value.operations[index];
}
}
extern "C" lffw_status lffw_diagnostic_create(const lffw_diagnostic_options *options, lffw_diagnostic_session **out, lffw_error **error) noexcept {
    if (out) *out = nullptr;
    return protect(error, [&] {
        require(out && options && options->struct_size >= sizeof(*options), "diagnostic options and output are required");
        require(options->flags && !(options->flags & ~uint32_t(7)), "unknown diagnostic flags");
        auto value = std::make_unique<lffw_diagnostic_session>();
        value->value = std::make_shared<diag::state>(); value->value->flags = options->flags;
        *out = value.release();
    });
}
extern "C" lffw_status lffw_diagnostic_begin(lffw_diagnostic_session *session, lffw_error **error) noexcept {
    return protect(error, [&] {
        require(session, "diagnostic session is required");
        if (!diag::selected.expired()) throw lffw_status(LFFW_BUSY);
        require(!session->bound || session->thread == std::this_thread::get_id(), "diagnostic session belongs to another thread");
        session->thread = std::this_thread::get_id(); session->bound = true;
        session->value->operations.clear(); session->value->failed = false;
        session->active = true; diag::selected = session->value;
    });
}
extern "C" lffw_status lffw_diagnostic_end(lffw_diagnostic_session *session, lffw_error **error) noexcept {
    return protect(error, [&] {
        thread(session);
        if (session->active) { diag::selected.reset(); session->active = false; }
    });
}
extern "C" lffw_status lffw_diagnostic_get_counts(const lffw_diagnostic_session *session, lffw_diagnostic_counts *out, lffw_error **error) noexcept {
    return protect(error, [&] { write(out, lffw_diagnostic_counts{sizeof(*out), snapshot(session).operations.size()}); });
}
extern "C" lffw_status lffw_diagnostic_get_operation(const lffw_diagnostic_session *session, size_t index, lffw_diagnostic_operation *out, lffw_error **error) noexcept {
    return protect(error, [&] {
        const auto &record = operation(session, index); auto value = record.value;
        value.name = record.name.c_str(); value.span_count = record.spans.size(); value.sleep_count = record.sleeps.size();
        write(out, value);
    });
}
extern "C" lffw_status lffw_diagnostic_get_span(const lffw_diagnostic_session *session, size_t operation_index, size_t index, lffw_diagnostic_span *out, lffw_error **error) noexcept {
    return protect(error, [&] {
        const auto &record = operation(session, operation_index);
        require(index < record.spans.size(), "diagnostic span index is out of range");
        const auto &span = record.spans[index]; auto value = span.value;
        value.phase = span.phase.c_str(); value.graph = span.graph.c_str(); value.tensor = span.tensor.c_str();
        write(out, value);
    });
}
extern "C" lffw_status lffw_diagnostic_get_sleep(const lffw_diagnostic_session *session, size_t operation_index, size_t index, lffw_diagnostic_sleep *out, lffw_error **error) noexcept {
    return protect(error, [&] {
        const auto &record = operation(session, operation_index);
        require(index < record.sleeps.size(), "diagnostic sleep index is out of range");
        write(out, record.sleeps[index]);
    });
}
extern "C" lffw_status lffw_diagnostic_destroy(lffw_diagnostic_session *session, lffw_error **error) noexcept {
    if (error) *error = nullptr;
    delete session; return LFFW_SUCCESS;
}
