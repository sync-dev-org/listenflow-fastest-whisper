#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace listenflow::fastest_whisper::test {
struct fc1_tensor_layout {
    std::vector<int64_t> shape;
    bool fp16;
};
inline void verify_fc1_layer(const std::string &name, const std::string &expected_name,
                             const std::vector<fc1_tensor_layout> &inputs,
                             const std::vector<fc1_tensor_layout> &outputs) {
    if (name != expected_name)
        throw std::runtime_error("fc1 plugin layer name or order differs");
    if (inputs.size() != 3 || outputs.size() != 1)
        throw std::runtime_error("fc1 plugin requires three inputs and one output");
    static const std::array<std::vector<int64_t>, 3> shapes{{{1, 1, 1280}, {5120, 1280}, {5120}}};
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].shape != shapes[i])
            throw std::runtime_error("fc1 plugin input shape differs: " + std::to_string(i));
        if (!inputs[i].fp16)
            throw std::runtime_error("fc1 plugin input dtype must be fp16: " + std::to_string(i));
    }
    if (outputs[0].shape != std::vector<int64_t>{1, 1, 5120})
        throw std::runtime_error("fc1 plugin output shape differs");
    if (!outputs[0].fp16)
        throw std::runtime_error("fc1 plugin output dtype must be fp16");
}
} // namespace listenflow::fastest_whisper::test
