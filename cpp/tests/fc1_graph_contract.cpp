#include "fc1_graph_contract.hpp"
#include <iostream>
#include <stdexcept>

namespace ft = listenflow::fastest_whisper::test;
static void rejected(const std::string &name, const std::vector<ft::fc1_tensor_layout> &inputs,
                     const std::vector<ft::fc1_tensor_layout> &outputs, const char *why) {
    try {
        ft::verify_fc1_layer(name, "model.decoder.layers.0.fc1.gelu", inputs, outputs);
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error(why);
}
int main() {
    try {
        const std::string name = "model.decoder.layers.0.fc1.gelu";
        const std::vector<ft::fc1_tensor_layout> inputs{
            {{1, 1, 1280}, true}, {{5120, 1280}, true}, {{5120}, true}};
        const std::vector<ft::fc1_tensor_layout> outputs{{{1, 1, 5120}, true}};
        ft::verify_fc1_layer(name, name, inputs, outputs);
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto invalid = inputs;
            invalid[i].shape.back() += 1;
            rejected(name, invalid, outputs, "fc1 input shape mutation was accepted");
            invalid = inputs;
            invalid[i].fp16 = false;
            rejected(name, invalid, outputs, "fc1 input dtype mutation was accepted");
        }
        auto invalid_outputs = outputs;
        invalid_outputs[0].shape.back() = 1280;
        rejected(name, inputs, invalid_outputs, "fc1 output shape mutation was accepted");
        invalid_outputs = outputs;
        invalid_outputs[0].fp16 = false;
        rejected(name, inputs, invalid_outputs, "fc1 output dtype mutation was accepted");
        auto missing_input = inputs;
        missing_input.pop_back();
        rejected(name, missing_input, outputs, "fc1 accepted an omitted bias input");
        auto extra_input = inputs;
        extra_input.push_back(inputs[0]);
        rejected(name, extra_input, outputs, "fc1 accepted an extra input");
        rejected(name, inputs, {}, "fc1 accepted an omitted output");
        auto extra_output = outputs;
        extra_output.push_back(outputs[0]);
        rejected(name, inputs, extra_output, "fc1 accepted an extra output");
        rejected("model.decoder.layers.1.fc1.gelu", inputs, outputs,
                 "fc1 accepted the wrong layer name or order");
        std::cout << "fc1 layers preserve names, three fp16 inputs and one fp16 output\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
