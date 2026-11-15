#pragma once
#include "runtime_driver.hpp"
#include <listenflow/whisper_network.hpp>
#include <functional>

namespace listenflow::fastest_whisper::runtime_detail {
inline size_t cache_bucket(size_t history) {
    require(history >= 1, "decoder history must contain the dummy position");
    for (auto boundary : cache_buckets) if (history <= size_t(boundary)) return size_t(boundary);
    throw std::invalid_argument("decoder history exceeds 448 positions");
}
struct cache_copies {
    std::function<void(CUdeviceptr, size_t, CUdeviceptr, size_t, size_t, size_t)> rows;
    std::function<void(CUdeviceptr, size_t)> zero;
};
inline cache_copies cache_transfer(gpu_queue &queue) {
    return {[&](auto target, auto pitch, auto source, auto source_pitch, auto bytes, auto rows) {
                queue.copy_rows(target, pitch, source, source_pitch, bytes, rows);
            }, [&](auto target, auto bytes) { queue.zero(target, bytes); }};
}
inline void compact_cache(const cache_copies &copy, CUdeviceptr target, CUdeviceptr source,
                          size_t history, size_t bucket, size_t rows, size_t rowbytes) {
    copy.rows(target, (history + 1) * rowbytes, source, (bucket + 1) * rowbytes, history * rowbytes, rows);
    copy.rows(target + history * rowbytes, (history + 1) * rowbytes,
              source + bucket * rowbytes, (bucket + 1) * rowbytes, rowbytes, rows);
}
inline void advance_cache(const cache_copies &copy, CUdeviceptr target, CUdeviceptr source,
                          size_t history, size_t bucket, size_t rows, size_t rowbytes) {
    const auto next = cache_bucket(history + 1);
    if (next != bucket) {
        copy.zero(target, rows * next * rowbytes);
        copy.rows(target, next * rowbytes, source, (bucket + 1) * rowbytes,
                  (history + 1) * rowbytes, rows);
    } else {
        copy.rows(target + history * rowbytes, bucket * rowbytes,
                  source + bucket * rowbytes, (bucket + 1) * rowbytes, rowbytes, rows);
    }
}
}
