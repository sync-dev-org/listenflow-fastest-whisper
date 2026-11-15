#pragma once
#include <NvInfer.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <tensoracle/builder.hpp>
#include <vector>

namespace listenflow::fastest_whisper::detail {
inline float from_half(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16, exponent = (h >> 10) & 31, fraction = h & 1023,
             bits;
    if (!exponent) {
        if (!fraction)
            bits = sign;
        else {
            int e = -14;
            while (!(fraction & 1024)) {
                fraction <<= 1;
                --e;
            }
            bits = sign | uint32_t(e + 127) << 23 | (fraction & 1023) << 13;
        }
    } else if (exponent == 31)
        bits = sign | 0x7f800000u | fraction << 13;
    else
        bits = sign | (exponent + 112) << 23 | fraction << 13;
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}
inline uint16_t to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const uint16_t sign = (bits >> 16) & 0x8000;
    const uint32_t fraction = bits & 0x7fffff, exponent = (bits >> 23) & 255;
    if (exponent == 255)
        return sign | 0x7c00 | (fraction ? 0x200 : 0);
    int e = int(exponent) - 127;
    if (e > 15)
        return sign | 0x7c00;
    if (e < -25)
        return sign;
    uint32_t mantissa = fraction | 0x800000;
    int shift = e < -14 ? -e - 1 : 13;
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((uint32_t(1) << shift) - 1), midpoint = uint32_t(1)
                                                                                   << (shift - 1);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1)))
        ++rounded;
    if (e < -14)
        return sign | uint16_t(rounded);
    return sign | uint16_t(((e + 14) << 10) + rounded);
}
inline std::vector<float> read_weight(const tensoracle::weights_view &weights,
                                      const std::string &key,
                                      const std::vector<int64_t> &expected) {
    auto view = weights.at(key.c_str());
    if (view.rank != expected.size() ||
        !std::equal(expected.begin(), expected.end(), view.dimensions))
        throw std::invalid_argument("weight shape differs: " + key);
    std::vector<float> result(view.element_count);
    for (size_t i = 0; i < result.size(); ++i) {
        switch (view.dtype) {
        case TENSORACLE_FLOAT32:
            result[i] = static_cast<const float *>(view.data)[i];
            break;
        case TENSORACLE_FLOAT16:
            result[i] = from_half(static_cast<const uint16_t *>(view.data)[i]);
            break;
        case TENSORACLE_BFLOAT16: {
            uint32_t bits = uint32_t(static_cast<const uint16_t *>(view.data)[i]) << 16;
            std::memcpy(&result[i], &bits, 4);
            break;
        }
        case TENSORACLE_FLOAT64:
            result[i] = float(static_cast<const double *>(view.data)[i]);
            break;
        default:
            throw std::invalid_argument("weight must have floating-point storage: " + key);
        }
    }
    return result;
}
// TensorRT borrows constant bytes until BOTH serializations finish. Deques
// retain each allocation across additions and across the callback return.
struct weight_storage {
    std::deque<std::unique_ptr<nvinfer1::IPluginV3>> plugins;
    std::deque<std::vector<float>> floats;
    std::deque<std::vector<uint16_t>> halves;
    std::deque<std::vector<int32_t>> integers;
    nvinfer1::Weights floating(std::vector<float> value, bool half) {
        auto count = int64_t(value.size());
        if (half) {
            auto &out = halves.emplace_back();
            out.reserve(value.size());
            for (float f : value)
                out.push_back(to_half(f));
            return {nvinfer1::DataType::kHALF, out.data(), count};
        }
        auto &out = floats.emplace_back(std::move(value));
        return {nvinfer1::DataType::kFLOAT, out.data(), count};
    }
    nvinfer1::Weights integer(std::vector<int32_t> value) {
        auto &out = integers.emplace_back(std::move(value));
        return {nvinfer1::DataType::kINT32, out.data(), int64_t(out.size())};
    }
};
} // namespace listenflow::fastest_whisper::detail
