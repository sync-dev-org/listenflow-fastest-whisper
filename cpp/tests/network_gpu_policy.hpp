#pragma once

#include <listenflow/whisper_network.hpp>

namespace listenflow::fastest_whisper::test {
inline bool skip_fp8_gpu_test(int major, int minor) {
    return !((major == 8 && minor == 9) || (major == 12 && minor == 0));
}
struct network_gpu_test_policy {
    bool skip;
    bool inspect_fp8;
};
inline network_gpu_test_policy gpu_test_policy(graph_kind kind, precision value, int major,
                                               int minor) {
    const bool inspect_fp8 = value == precision::fp8 && kind != graph_kind::mel;
    return {value == precision::fp8 && skip_fp8_gpu_test(major, minor), inspect_fp8};
}
} // namespace listenflow::fastest_whisper::test
