#include "cache_recovery.hpp"
#include "model_error.hpp"
#include "network_comparison.hpp"
#include "network_gpu_policy.hpp"
#include "precision_policy.hpp"
#include "sha256.hpp"
#include "weight_storage.hpp"
#include <listenflow/whisper_network.hpp>
#include <tensoracle/plugins.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fw = listenflow::fastest_whisper;
namespace fs = std::filesystem;
static void expect(bool value, const char *why) {
    if (!value)
        throw std::runtime_error(why);
}
static void write(const fs::path &path, const std::string &value) {
    std::ofstream(path, std::ios::binary) << value;
}
static fw::model_source source(const fs::path &root) {
    fw::model_source value;
    value.config = {128, 1500, 1280, 20, 5120, 32, 20, 5120, 32, 448, 51866};
    value.weights = {(root / "model-1.safetensors").string(),
                     (root / "model-2.safetensors").string()};
    value.config_file = (root / "config.json").string();
    value.tokenizer_file = (root / "tokenizer.json").string();
    value.scale_file = (root / "scales.json").string();
    value.mel_filter_file = (root / "filters.npz").string();
    value.mel_weights = (root / "mel.safetensors").string();
    value.fc1_library = (root / "fc1.so").string();
    value.no_speech = 50362;
    value.timestamp_begin = 50365;
    for (const auto &file :
         {value.config_file, value.tokenizer_file, value.scale_file, value.mel_filter_file})
        write(file, "original");
    return value;
}
static const fw::tensor_spec &spec(const fw::build_declaration &options, const std::string &name) {
    auto it =
        std::find_if(options.io.begin(), options.io.end(), [&](auto &v) { return v.name == name; });
    if (it == options.io.end())
        throw std::runtime_error("missing tensor " + name);
    return *it;
}
static void identities(const fs::path &root) {
    auto s = source(root);
    for (auto kind : {fw::graph_kind::mel, fw::graph_kind::encoder, fw::graph_kind::decoder,
                      fw::graph_kind::sampled_decoder}) {
        const auto before =
            fw::engine_build(s, kind, fw::precision::fp8, 5).declaration().graph_version;
        expect(before ==
                   fw::engine_build(s, kind, fw::precision::fp8, 5).declaration().graph_version,
               "same contents changed identity");
        std::vector<std::string> files{s.config_file};
        if (kind == fw::graph_kind::mel)
            files.push_back(s.mel_filter_file);
        else
            files.push_back(s.scale_file);
        if (kind == fw::graph_kind::decoder || kind == fw::graph_kind::sampled_decoder)
            files.push_back(s.tokenizer_file);
        for (auto &path : files) {
            write(path, "changed");
            expect(before !=
                       fw::engine_build(s, kind, fw::precision::fp8, 5).declaration().graph_version,
                   "auxiliary content omitted from identity");
            write(path, "original");
        }
        auto options = fw::engine_build(s, kind, fw::precision::fp8, 5).declaration();
        for (auto &path : options.weights)
            expect(fs::path(path).extension() == ".safetensors", "non-safetensors build weight");
        if (kind == fw::graph_kind::mel)
            expect(options.weights == std::vector<std::string>{s.mel_weights},
                   "mel reads full checkpoint");
        else
            expect(options.weights == s.weights, "shards were lost");
    }
    const auto base = fw::engine_build(s, fw::graph_kind::decoder, fw::precision::fp32)
                          .declaration()
                          .graph_version;
    auto altered = s;
    altered.no_speech = 1;
    expect(base != fw::engine_build(altered, fw::graph_kind::decoder, fw::precision::fp32)
                       .declaration()
                       .graph_version,
           "concrete token value omitted from identity");
    altered = s;
    altered.config.audio_state = 640;
    expect(base != fw::engine_build(altered, fw::graph_kind::decoder, fw::precision::fp32)
                       .declaration()
                       .graph_version,
           "concrete config omitted from identity");
    altered = s;
    altered.weights.push_back("config.json");
    try {
        fw::engine_build invalid(altered, fw::graph_kind::encoder, fw::precision::fp32);
        throw std::runtime_error("accepted JSON weight");
    } catch (const std::invalid_argument &) {
    }
}
static void layouts(const fs::path &root) {
    auto s = source(root);
    for (auto dtype : {fw::precision::fp32, fw::precision::fp16, fw::precision::fp8}) {
        fw::engine_build encoder(s, fw::graph_kind::encoder, dtype);
        auto &e = encoder.declaration();
        expect(e.workspace_limit == int64_t(4) * 1024 * 1024 * 1024 && e.optimization_level == 5,
               "product build settings changed");
        expect(spec(e, "mel").shape == std::vector<int64_t>({1, 128, 3000}),
               "encoder input layout changed");
        expect(spec(e, "encoder").dtype ==
                   (dtype == fw::precision::fp32 ? fw::tensor_dtype::fp32 : fw::tensor_dtype::fp16),
               "encoder precision changed");
        fw::engine_build decoder(s, fw::graph_kind::decoder, dtype);
        expect(spec(decoder.declaration(), "token").dtype == fw::tensor_dtype::int32,
               "greedy token type changed");
        expect(decoder.declaration().plugins.empty() == (dtype != fw::precision::fp16),
               "wrong graph loads fc1");
        fw::engine_build sampled(s, fw::graph_kind::sampled_decoder, dtype, 5);
        auto &d = sampled.declaration();
        expect(spec(d, "tokens").shape == std::vector<int64_t>({5, 1}), "sample batch lost");
        expect(spec(d, "sampled").shape == std::vector<int64_t>({5, 1, 3}),
               "sample result changed");
        expect(spec(d, "cross_keys").shape == std::vector<int64_t>({32, 1, 20, 1500, 64}),
               "cross KV no longer shared");
        for (auto bucket : fw::cache_buckets) {
            auto shapes = sampled.bucket_shapes(bucket);
            expect(shapes.size() == 3, "bucket omitted a ranged input");
            for (auto &shape : shapes) {
                if (shape.name == "mask")
                    expect(shape.dimensions == std::vector<int64_t>({1, 1, 1, bucket + 1}),
                           "mask history differs");
                else
                    expect(shape.dimensions == std::vector<int64_t>({32, 5, 20, bucket, 64}),
                           "KV history differs");
            }
        }
        expect(d.graph_version != fw::engine_build(s, fw::graph_kind::sampled_decoder, dtype, 2)
                                      .declaration()
                                      .graph_version,
               "sample batches share identity");
    }
}
static void weights_from_safetensors(const fs::path &root) {
    const auto file = root / "weights.safetensors";
    std::string header =
        R"({"matrix":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"half":{"dtype":"F16","shape":[2],"data_offsets":[24,28]},"bfloat":{"dtype":"BF16","shape":[1],"data_offsets":[28,30]}})";
    while (header.size() % 8)
        header += ' ';
    {
        std::ofstream output(file, std::ios::binary);
        uint64_t length = header.size();
        output.write(reinterpret_cast<const char *>(&length), 8);
        output << header;
        const float values[]{1, -2, 3, 4, 5, 6};
        output.write(reinterpret_cast<const char *>(values), sizeof(values));
        const uint16_t encoded[]{0x3c00, 0x0001, 0x3fc0};
        output.write(reinterpret_cast<const char *>(encoded), sizeof(encoded));
    }
    auto owned = tensoracle::open_weights(file);
    expect(fw::detail::read_weight(owned.view(), "matrix", {2, 3}) ==
               std::vector<float>({1, -2, 3, 4, 5, 6}),
           "checkpoint values changed");
    expect(fw::detail::read_weight(owned.view(), "half", {2}) == std::vector<float>({1, 0x1p-24f}),
           "F16 checkpoint conversion changed");
    expect(fw::detail::read_weight(owned.view(), "bfloat", {1}) == std::vector<float>({1.5}),
           "BF16 checkpoint conversion changed");
    bool rejected = false;
    try {
        fw::detail::read_weight(owned.view(), "matrix", {3, 2});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    expect(rejected, "wrong checkpoint shape accepted");
    fw::detail::weight_storage storage;
    auto first = storage.floating({1.25f, -2.5f}, true);
    for (int i = 0; i < 100; ++i)
        storage.floating(std::vector<float>(100, float(i)), true);
    const auto *kept = static_cast<const uint16_t *>(first.values);
    expect(kept[0] == 0x3d00 && kept[1] == 0xc100, "later constants invalidate retained storage");
    // Round-trip all finite IEEE half encodings; NaN payloads are intentionally canonicalized.
    for (unsigned h = 0; h < 65536; ++h)
        if ((h & 0x7c00) != 0x7c00)
            expect(fw::detail::to_half(fw::detail::from_half(uint16_t(h))) == h,
                   "half conversion loses finite values");
}
static void recovery() {
    // The lambdas replace only tensoracle's external open/execution boundary.
    for (auto status : {TENSORACLE_CORRUPT_ENGINE, TENSORACLE_ENGINE_CONDITION_MISMATCH,
                        TENSORACLE_ENGINE_IO_MISMATCH}) {
        int calls = 0;
        int result = fw::detail::recover_saved_engine(
            false,
            [&](bool forced, bool &) {
                ++calls;
                if (!forced)
                    throw tensoracle::error(status, "bad saved engine");
                return 7;
            },
            [](int &) {});
        expect(result == 7 && calls == 2, "saved engine not rebuilt exactly once");
    }
    for (tensoracle_status status : std::array<tensoracle_status, 4>{
             TENSORACLE_INVALID_ARGUMENT, TENSORACLE_OUT_OF_MEMORY, TENSORACLE_INTERNAL_ERROR,
             TENSORACLE_PLUGIN_LIBRARY_ERROR}) {
        int calls = 0;
        try {
            fw::detail::recover_saved_engine(
                false,
                [&](bool, bool &) -> int {
                    ++calls;
                    throw tensoracle::error(status, "not cache corruption");
                },
                [](int &) {});
        } catch (const tensoracle::error &e) {
            expect(e.status() == status && calls == 1, "unrelated failure rebuilt");
            continue;
        }
        throw std::runtime_error("unrelated failure swallowed");
    }
    int calls = 0, probes = 0;
    expect(fw::detail::recover_saved_engine(
               true,
               [&](bool force, bool &) {
                   expect(force == (calls > 0), "recovery lost force flag");
                   return ++calls;
               },
               [&](int &) {
                   if (++probes == 1)
                       throw tensoracle::error(TENSORACLE_INTERNAL_ERROR, "delayed plugin failure");
               }) == 2 &&
               probes == 2,
           "delayed plugin failure not recovered");
    calls = 0;
    bool caught_callback = false;
    try {
        fw::detail::recover_saved_engine(
            false,
            [&](bool, bool &built) -> int {
                ++calls;
                built = true;
                throw tensoracle::error(TENSORACLE_CORRUPT_ENGINE, "callback error");
            },
            [](int &) {});
    } catch (const tensoracle::error &) {
        expect(calls == 1, "callback error treated as cache corruption");
        caught_callback = true;
    }
    expect(caught_callback, "callback failure was swallowed");
    calls = 0;
    bool caught_rebuild = false;
    try {
        fw::detail::recover_saved_engine(
            false,
            [&](bool, bool &) -> int {
                ++calls;
                throw tensoracle::error(TENSORACLE_CORRUPT_ENGINE, "persistent failure");
            },
            [](int &) {});
    } catch (const tensoracle::error &) {
        expect(calls == 2, "rebuild repeated");
        caught_rebuild = true;
    }
    expect(caught_rebuild, "failed rebuild was swallowed");
    for (tensoracle_status status :
         std::array<tensoracle_status, 2>{TENSORACLE_INVALID_ARGUMENT, TENSORACLE_OUT_OF_MEMORY}) {
        calls = 0;
        int probes = 0;
        bool caught = false;
        try {
            fw::detail::recover_saved_engine(
                true, [&](bool, bool &) { return ++calls; },
                [&](int &) {
                    ++probes;
                    throw tensoracle::error(status, "probe argument/allocation failure");
                });
        } catch (const tensoracle::error &error) {
            caught = true;
            expect(error.status() == status && calls == 1 && probes == 1,
                   "probe argument/allocation failure rebuilt");
        }
        expect(caught, "probe failure was swallowed");
    }
    calls = 0;
    bool caught_new_probe = false;
    try {
        fw::detail::recover_saved_engine(
            true,
            [&](bool, bool &built) {
                built = true;
                return ++calls;
            },
            [](int &) {
                throw tensoracle::error(TENSORACLE_INTERNAL_ERROR,
                                        "newly built plugin engine failed");
            });
    } catch (const tensoracle::error &) {
        caught_new_probe = true;
        expect(calls == 1, "newly built plugin engine rebuilt");
    }
    expect(caught_new_probe, "newly built plugin failure swallowed");
}
static void gpu_test_policies() {
    struct graph_case {
        fw::graph_kind kind;
        const char *name;
        bool quantized;
    };
    const graph_case graphs[]{{fw::graph_kind::mel, "mel", false},
                              {fw::graph_kind::encoder, "encoder", true},
                              {fw::graph_kind::decoder, "decoder", true},
                              {fw::graph_kind::sampled_decoder, "sampled", true}};
    struct precision_case {
        fw::precision value;
        const char *name;
        bool fp8;
    };
    const precision_case precisions[]{{fw::precision::fp32, "fp32", false},
                                      {fw::precision::fp16, "fp16", false},
                                      {fw::precision::fp8, "fp8", true}};
    struct device_case {
        int major, minor;
        bool skip_fp8;
    };
    const device_case devices[]{{8, 6, true}, {8, 9, false}, {12, 0, false}};
    int cases = 0;
    for (const auto &graph : graphs)
        for (const auto &precision : precisions)
            for (const auto &device : devices) {
                const auto policy = fw::test::gpu_test_policy(graph.kind, precision.value,
                                                              device.major, device.minor);
                const auto context = std::string(graph.name) + "/" + precision.name + " on sm" +
                                     std::to_string(device.major) + std::to_string(device.minor);
                expect(policy.skip == (precision.fp8 && device.skip_fp8),
                       ("GPU test skip mismatch for " + context).c_str());
                expect(policy.inspect_fp8 == (precision.fp8 && graph.quantized),
                       ("FP8 structure inspection mismatch for " + context).c_str());
                ++cases;
            }
    std::cout << "GPU test skip and FP8 structure policies verified: " << cases << " cases\n";
}
int main(int argc, char **argv) {
    try {
        const fs::path root = argc > 1 ? argv[1] : "network-contract-data";
        fs::create_directories(root);
        gpu_test_policies();
        identities(root);
        layouts(root);
        recovery();
        weights_from_safetensors(root);
        {
            float expected_values[]{50000.f, -1.f, 0.1f};
            tensoracle_weight reference{};
            reference.dtype = TENSORACLE_FLOAT32;
            reference.rank = 3;
            const int64_t shape[]{1, 1, 3};
            reference.dimensions = shape;
            reference.element_count = 3;
            reference.data = expected_values;
            reference.byte_size = sizeof(expected_values);
            std::vector<char> actual(sizeof(expected_values));
            std::memcpy(actual.data(), expected_values, sizeof(expected_values));
            compare_network_output("sampled", actual, {1, 1, 3}, reference);
            float wrong_token = 50001;
            std::memcpy(actual.data(), &wrong_token, sizeof(wrong_token));
            bool caught = false;
            try {
                compare_network_output("sampled", actual, {1, 1, 3}, reference);
            } catch (const std::runtime_error &) {
                caught = true;
            }
            expect(caught, "sample comparison tolerates a different token ID");
            std::fill(actual.begin(), actual.end(), 0);
            caught = false;
            try {
                compare_network_output("mel", actual, {1, 1, 3}, reference);
            } catch (const std::runtime_error &) {
                caught = true;
            }
            expect(caught, "mel comparison accepts a graph that returns zero");
        }
        for (tensoracle_status status : std::array<tensoracle_status, 9>{
                 TENSORACLE_INVALID_ARGUMENT, TENSORACLE_INCOMPATIBLE_VERSION,
                 TENSORACLE_OUT_OF_MEMORY, TENSORACLE_INTERNAL_ERROR, TENSORACLE_CORRUPT_ENGINE,
                 TENSORACLE_ENGINE_CONDITION_MISMATCH, TENSORACLE_ENGINE_IO_MISMATCH,
                 TENSORACLE_PLUGIN_LIBRARY_ERROR, 512}) {
            bool caught = false;
            try {
                fw::detail::model_call([&]() -> void {
                    throw tensoracle::error(status, "original backend diagnostic");
                });
            } catch (const fw::model_error &error) {
                caught = true;
                expect(error.status() == status &&
                           std::string(error.what()) == "original backend diagnostic",
                       "model error loses backend status/diagnostic");
            }
            expect(caught, "backend error escaped without a model error");
        }
        fw::detail::sha256 empty_hash;
        expect(empty_hash.finish() ==
                   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
               "empty auxiliary digest is not SHA-256");
        fw::detail::sha256 hash;
        hash.update("abc", 3);
        expect(hash.finish() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
               "auxiliary digest is not SHA-256");
        bool rejected = false;
        try {
            fw::detail::check_precision(fw::precision::fp8, 8, 6);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        expect(rejected, "fp8 accepted on sm86");
        fw::detail::check_precision(fw::precision::fp32, 8, 6);
        fw::detail::check_precision(fw::precision::fp16, 8, 6);
        fw::detail::check_precision(fw::precision::fp8, 8, 9);
        fw::detail::check_precision(fw::precision::fp8, 12, 0);
        expect(fw::test::skip_fp8_gpu_test(8, 6), "FP8 GPU test must skip sm86");
        expect(!fw::test::skip_fp8_gpu_test(8, 9), "FP8 GPU test must run on sm89");
        expect(!fw::test::skip_fp8_gpu_test(12, 0), "FP8 GPU test must run on sm120");
        // Independent IEEE half edge cases: ties round to even and subnormals survive.
        expect(fw::detail::to_half(1.00048828125f) == 0x3c00, "half tie rounding changed");
        expect(fw::detail::to_half(0x1p-24f) == 1, "half subnormal lost");
        expect(fw::detail::from_half(0x8000) == 0 && std::signbit(fw::detail::from_half(0x8000)),
               "negative zero lost");
        std::cout << "network declarations, identity, recovery and half conversion passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
