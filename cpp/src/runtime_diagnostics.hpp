#pragma once
#include <listenflow/whisper_runtime_diagnostics.h>
#include <memory>
#include <string>
#include <vector>
#include <time.h>

namespace listenflow::fastest_whisper::diagnostic {
inline uint64_t now(clockid_t clock = CLOCK_MONOTONIC) noexcept {
    timespec value{};
    clock_gettime(clock, &value);
    return uint64_t(value.tv_sec) * 1000000000 + uint64_t(value.tv_nsec);
}
struct span_record {
    std::string phase, graph, tensor;
    lffw_diagnostic_span value{};
};
struct operation_record {
    std::string name;
    lffw_diagnostic_operation value{};
    std::vector<span_record> spans;
    std::vector<lffw_diagnostic_sleep> sleeps;
};
struct state {
    uint32_t flags;
    bool failed = false;
    std::vector<std::shared_ptr<operation_record>> operations;
};
extern thread_local std::weak_ptr<state> selected;
extern thread_local std::shared_ptr<state> recording;
extern thread_local std::shared_ptr<operation_record> current;
extern thread_local size_t wait_index;
void failed() noexcept;
void status(int32_t code) noexcept;
void input(int32_t token, int32_t kind) noexcept;
void output(int32_t token) noexcept;
void query() noexcept;
class operation {
    bool root_ = false;
  public:
    explicit operation(const char *name) noexcept;
    ~operation();
};
class span {
    uint64_t cpu_ = 0;
    size_t previous_wait_ = SIZE_MAX;
  public:
    std::shared_ptr<operation_record> owner;
    size_t index = SIZE_MAX;
    explicit span(const char *phase, const char *graph = "", size_t bucket = 0,
                  const char *tensor = "", size_t bytes = 0) noexcept;
    ~span();
    span(const span &) = delete;
};
class sleep {
    uint64_t started_, requested_;
  public:
    explicit sleep(uint64_t requested) : started_(now()), requested_(requested) {}
    ~sleep();
};
}
