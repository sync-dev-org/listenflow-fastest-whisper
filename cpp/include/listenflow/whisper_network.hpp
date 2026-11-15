#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace listenflow::fastest_whisper {
enum class precision { fp32, fp16, fp8 };
enum class graph_kind { mel, encoder, decoder, sampled_decoder };
enum class tensor_dtype { fp32, fp16, int32, boolean };

// The original backend status is preserved, including values introduced by a
// future compatible backend. The C ABI can copy status() without translation.
class model_error : public std::runtime_error {
  public:
    model_error(int32_t status, std::string message)
        : std::runtime_error(std::move(message)), status_(status) {}
    int32_t status() const noexcept { return status_; }

  private:
    int32_t status_;
};
struct model_config {
    int64_t n_mels, audio_positions, audio_state, audio_heads, audio_ffn, audio_layers;
    int64_t text_heads, text_ffn, text_layers, text_positions, vocab_size;
};
struct model_source {
    model_config config{};
    std::vector<std::string> weights;
    std::string config_file, tokenizer_file, scale_file, mel_filter_file;
    std::string mel_weights, fc1_library;
    std::map<std::string, float> scales;
    std::optional<int32_t> no_speech, timestamp_begin;
};
struct tensor_spec {
    std::string name;
    tensor_dtype dtype;
    std::vector<int64_t> shape;
    bool input;
    // Input profiles live with their tensor, instead of a separate I/O list.
    // Output profiles are empty; fixed inputs have three identical bounds.
    std::vector<int64_t> minimum, optimum, maximum;
};
struct build_declaration {
    std::string name, graph_version;
    std::vector<std::string> weights, plugins;
    std::vector<tensor_spec> io;
    int64_t workspace_limit;
    int32_t optimization_level;
};
struct input_shape {
    std::string name;
    std::vector<int64_t> dimensions;
};
struct tensor_address {
    std::string name;
    void *address;
};

class whisper_engine {
  public:
    whisper_engine();
    ~whisper_engine();
    whisper_engine(whisper_engine &&) noexcept;
    whisper_engine &operator=(whisper_engine &&) noexcept;
    whisper_engine(const whisper_engine &) = delete;
    whisper_engine &operator=(const whisper_engine &) = delete;
    std::string path() const;
    void close();

  private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
    explicit whisper_engine(std::unique_ptr<implementation>);
    friend class engine_build;
};
class network_execution {
  public:
    network_execution();
    ~network_execution();
    network_execution(network_execution &&) noexcept;
    network_execution &operator=(network_execution &&) noexcept;
    network_execution(const network_execution &) = delete;
    network_execution &operator=(const network_execution &) = delete;
    void set_input_shapes(const std::vector<input_shape> &);
    std::vector<int64_t> get_tensor_shape(const char *name);
    tensor_dtype get_tensor_dtype(const char *name) const;
    void enqueue(const std::vector<tensor_address> &, void *stream);
    void close();

  private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
    explicit network_execution(std::unique_ptr<implementation>);
    friend class engine_build;
};
inline constexpr std::array<int64_t, 16> cache_buckets{1,   8,   16,  24,  32,  48,  64,  96,
                                                       128, 160, 192, 224, 256, 320, 384, 448};
struct bucket_execution {
    int64_t history;
    network_execution execution;
};
class engine_build {
  public:
    engine_build(model_source, graph_kind, precision, int32_t sample_batch = 1);
    const build_declaration &declaration() const noexcept { return options_; }
    whisper_engine open() const;
    std::vector<input_shape> bucket_shapes(int64_t history) const;
    network_execution create_execution(const whisper_engine &,
                                       std::optional<int64_t> history = {}) const;
    std::vector<bucket_execution> create_sampled_executions(const whisper_engine &) const;

  private:
    model_source source_;
    graph_kind kind_;
    precision precision_;
    int32_t batch_;
    build_declaration options_;
};
} // namespace listenflow::fastest_whisper
