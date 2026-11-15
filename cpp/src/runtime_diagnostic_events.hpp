#pragma once
#include "runtime_diagnostics.hpp"
#include <cuda.h>

namespace listenflow::fastest_whisper::diagnostic {
struct event_pair {
    CUevent start = nullptr, stop = nullptr;
    std::shared_ptr<state> session;
    std::shared_ptr<operation_record> operation;
    size_t index = 0;
    bool recorded = false;
};
class event_log {
    std::vector<std::unique_ptr<event_pair>> pending_;
  public:
    event_pair *begin(const span &value, CUstream stream) noexcept {
        if (!value.owner || !recording || !(recording->flags & LFFW_DIAGNOSTIC_EVENTS)) return nullptr;
        try {
            auto pair = std::make_unique<event_pair>();
            pair->session = recording; pair->operation = value.owner; pair->index = value.index;
            // Retain ownership before calling the driver, including partial failures.
            pending_.push_back(std::move(pair));
            auto *result = pending_.back().get();
            if (cuEventCreate(&result->start, 0) != CUDA_SUCCESS ||
                cuEventCreate(&result->stop, 0) != CUDA_SUCCESS ||
                cuEventRecord(result->start, stream) != CUDA_SUCCESS) {
                failed(); return nullptr;
            }
            return result;
        } catch (...) { failed(); return nullptr; }
    }
    // Called only after the product has drained its stream. Never synchronizes.
    void complete() noexcept {
        for (auto entry = pending_.begin(); entry != pending_.end();) {
            auto &pair = **entry;
            if (pair.recorded && pair.start && pair.stop) {
                float elapsed = 0;
                auto result = cuEventElapsedTime(&elapsed, pair.start, pair.stop);
                if (result == CUDA_SUCCESS) {
                    auto &value = pair.operation->spans[pair.index].value;
                    value.gpu_ms = elapsed; value.gpu_time_valid = 1;
                } else pair.session->failed = true;
                pair.recorded = false;
            }
            auto release = [&](CUevent &event) {
                if (!event) return;
                if (cuEventDestroy(event) == CUDA_SUCCESS) event = nullptr;
                else pair.session->failed = true;
            };
            release(pair.stop); release(pair.start);
            if (!pair.start && !pair.stop) entry = pending_.erase(entry);
            else ++entry;
        }
    }
    bool empty() const noexcept { return pending_.empty(); }
};
class gpu_span {
    span value_;
    event_pair *pair_ = nullptr;
    CUstream stream_;
  public:
    gpu_span(event_log &events, CUstream stream, const char *phase, const char *graph = "",
             size_t bucket = 0, const char *tensor = "", size_t bytes = 0, bool replay = false)
        : value_(phase, graph, bucket, tensor, bytes), stream_(stream) {
        if (value_.owner) value_.owner->spans[value_.index].value.graph_replay = replay;
        pair_ = events.begin(value_, stream);
    }
    ~gpu_span() {
        if (!pair_) return;
        if (cuEventRecord(pair_->stop, stream_) == CUDA_SUCCESS) pair_->recorded = true;
        else pair_->session->failed = true;
    }
};
}
