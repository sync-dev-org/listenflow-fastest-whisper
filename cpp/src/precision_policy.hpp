#pragma once
#include <listenflow/whisper_network.hpp>
#include <stdexcept>

namespace listenflow::fastest_whisper::detail {
inline void check_precision(precision dtype, int major, int minor) {
    if (dtype == precision::fp8 && major == 8 && minor == 6)
        throw std::invalid_argument("fp8 is unsupported on sm86; use an sm89 or sm120 GPU");
}
} // namespace listenflow::fastest_whisper::detail
