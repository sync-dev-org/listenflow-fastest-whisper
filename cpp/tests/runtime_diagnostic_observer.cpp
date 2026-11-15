#include <listenflow/whisper_runtime_diagnostics.h>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

extern "C" int lffw_test_live_resources();
// SP-LFFASTESTWHISPER-001, SP-LFFASTESTWHISPER-017: actual runtime calls
// retain their results and release timing events after normal completion.
int main() {
    char directory[] = "/tmp/lffw-diagnostic-XXXXXX";
    assert(mkdtemp(directory));
    const auto file = std::string(directory) + "/source.safetensors";
    std::ofstream(file) << "diagnostic source";
    lffw_config config{sizeof(config), 80, 1500, 16, 1, 32, 1, 1, 32, 1, 448, 16};
    const char *weights[] = {file.c_str()};
    lffw_model_options options{};
    options.struct_size = sizeof(options); options.abi_major = LFFW_ABI_MAJOR;
    options.config = &config; options.weights = weights; options.weight_count = 1;
    options.config_file = options.tokenizer_file = options.mel_filter_file = options.mel_weights = file.c_str();
    options.no_speech_token = options.timestamp_begin_token = -1;
    lffw_model *model = nullptr;
    assert(lffw_model_open(&options, &model, nullptr) == LFFW_SUCCESS);
    const auto baseline_resources = lffw_test_live_resources();
    lffw_diagnostic_options flags{sizeof(flags), 7};
    lffw_diagnostic_session *session = nullptr;
    assert(lffw_diagnostic_create(&flags, &session, nullptr) == LFFW_SUCCESS);
    assert(lffw_diagnostic_begin(session, nullptr) == LFFW_SUCCESS);
    std::vector<float> features(80 * 3000, 0);
    lffw_encoded_audio *audio = nullptr;
    assert(lffw_encode_features(model, features.data(), features.size(), &audio, nullptr) == LFFW_SUCCESS);
    assert(lffw_diagnostic_end(session, nullptr) == LFFW_SUCCESS);
    lffw_diagnostic_counts counts{sizeof(counts), 0};
    assert(lffw_diagnostic_get_counts(session, &counts, nullptr) == LFFW_SUCCESS);
    assert(counts.operation_count == 1);
    lffw_diagnostic_operation operation{}; operation.struct_size = sizeof(operation);
    assert(lffw_diagnostic_get_operation(session, 0, &operation, nullptr) == LFFW_SUCCESS);
    bool enqueue = false, upload = false, wait = false;
    for (size_t i = 0; i < operation.span_count; ++i) {
        lffw_diagnostic_span span{}; span.struct_size = sizeof(span);
        assert(lffw_diagnostic_get_span(session, 0, i, &span, nullptr) == LFFW_SUCCESS);
        if (std::string(span.phase) == "enqueue") {
            enqueue = true; assert(span.gpu_time_valid && span.gpu_ms > 0);
        }
        if (std::string(span.phase) == "h2d") {
            upload = true; assert(span.bytes == features.size() * sizeof(float));
            assert(span.gpu_time_valid && span.gpu_ms > 0);
        }
        if (std::string(span.phase) == "wait") { wait = true; assert(span.query_count == 1); }
    }
    assert(enqueue && upload && wait);
    assert(lffw_diagnostic_destroy(session, nullptr) == LFFW_SUCCESS);
    assert(lffw_encoded_destroy(audio, nullptr) == LFFW_SUCCESS);
    assert(lffw_test_live_resources() >= baseline_resources);
    assert(lffw_model_destroy(model, nullptr) == LFFW_SUCCESS);
    assert(lffw_test_live_resources() == 0);
    std::filesystem::remove_all(directory);
}
