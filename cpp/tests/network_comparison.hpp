#pragma once
#include "weight_storage.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

inline double network_scalar(const void *bytes, tensoracle_dtype dtype, size_t index) {
    const auto *data = static_cast<const char *>(bytes);
    if (dtype == TENSORACLE_FLOAT16) {
        uint16_t v;
        std::memcpy(&v, data + 2 * index, 2);
        return listenflow::fastest_whisper::detail::from_half(v);
    }
    if (dtype == TENSORACLE_FLOAT32) {
        float v;
        std::memcpy(&v, data + 4 * index, 4);
        return v;
    }
    if (dtype == TENSORACLE_INT32) {
        int32_t v;
        std::memcpy(&v, data + 4 * index, 4);
        return v;
    }
    throw std::runtime_error("unsupported reference output type");
}
inline void compare_network_output(const std::string &name, const std::vector<char> &actual,
                                   const std::vector<int64_t> &shape,
                                   const tensoracle_weight &reference, bool enforce_values = true) {
    if (reference.rank != shape.size() ||
        !std::equal(shape.begin(), shape.end(), reference.dimensions) ||
        reference.byte_size != actual.size())
        throw std::runtime_error("reference output layout differs: " + name);
    if (!enforce_values)
        return;
    // Existing product bounds: test_gpu_mel.py / test_fp32_gpu.py. Fused result
    // bounds are test_fp16_gpu.py's score/silence limits and pytest's default
    // relative component. Token IDs are always exact, including float32 IDs.
    double atol = 2e-2, rtol = 2e-2;
    if (name == "mel" || name == "log_mel")
        atol = rtol = 3e-5;
    else if (name == "encoder" || name == "cross_keys" || name == "cross_values")
        rtol = 5e-3;
    for (size_t i = 0; i < reference.element_count; ++i) {
        const double a = network_scalar(actual.data(), reference.dtype, i),
                     b = network_scalar(reference.data, reference.dtype, i);
        double abs_limit = atol, rel_limit = rtol;
        const bool fused = name == "greedy" || name == "sampled";
        if (reference.dtype == TENSORACLE_INT32 || (fused && i % 3 == 0))
            abs_limit = rel_limit = 0;
        else if (fused) {
            abs_limit = i % 3 == 1 ? 2e-4 : 2e-5;
            rel_limit = 1e-6;
        }
        if (std::isnan(a) && std::isnan(b))
            continue;
        if (a == b)
            continue;
        if (!std::isfinite(a) || !std::isfinite(b) ||
            std::abs(a - b) > abs_limit + rel_limit * std::abs(b))
            throw std::runtime_error("native output differs from 1.x reference: " + name +
                                     " at element " + std::to_string(i));
    }
}
