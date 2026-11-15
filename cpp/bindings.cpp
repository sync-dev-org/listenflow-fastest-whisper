#include <nanobind/nanobind.h>
#include <listenflow/whisper_runtime.hpp>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <memory>
#include <vector>

namespace nb = nanobind;
namespace rt = listenflow::fastest_whisper::runtime;
namespace {
using array = nb::ndarray<nb::numpy, nb::c_contig, nb::device::cpu>;
using waveform = nb::ndarray<const float, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using features = nb::ndarray<const float, nb::numpy, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using tokens = nb::ndarray<const int32_t, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using floats = nb::ndarray<const float, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
struct metadata { lffw_config config; lffw_model_info info; };
struct bound_mel { rt::mel value; std::shared_ptr<metadata> meta; };
struct bound_audio { rt::encoded_audio value; std::shared_ptr<metadata> meta; };
struct bound_decoder { rt::decoder value; std::shared_ptr<metadata> meta; int32_t batch; };
struct bound_prefix { rt::prefix value; std::shared_ptr<metadata> meta; };
struct bound_model { rt::model value; std::shared_ptr<metadata> meta; };
struct source_input {
    lffw_config config{};
    std::vector<std::string> weights;
    std::vector<const char *> weight_pointers;
    std::string config_file, tokenizer_file, scale_file, mel_filter_file, mel_weights, fc1_library;
    std::vector<lffw_scale> scales;
    std::vector<const lffw_scale *> scale_pointers;
    std::map<std::string, float> scale_values;
    std::vector<int32_t> suppressed, begin;
    lffw_model_options options{};
    source_input(nb::dict source, const std::string &precision, int32_t device, int32_t batch,
                 std::vector<int32_t> normal, std::vector<int32_t> first)
        : suppressed(std::move(normal)), begin(std::move(first)) {
        auto c = nb::cast<nb::dict>(source["config"]); config.struct_size = sizeof(config);
#define DIM(name) config.name = nb::cast<int64_t>(c[#name])
        DIM(n_mels); DIM(audio_positions); DIM(audio_state); DIM(audio_heads); DIM(audio_ffn);
        DIM(audio_layers); DIM(text_heads); DIM(text_ffn); DIM(text_layers); DIM(text_positions); DIM(vocab_size);
#undef DIM
        weights = nb::cast<std::vector<std::string>>(source["weights"]);
        for (auto &path : weights) weight_pointers.push_back(path.c_str());
        auto string = [&](const char *key) { return source.contains(key) ? nb::cast<std::string>(source[key]) : std::string(); };
        config_file = string("config_file"); tokenizer_file = string("tokenizer_file");
        scale_file = string("scale_file"); mel_filter_file = string("mel_filter_file");
        mel_weights = string("mel_weights"); fc1_library = string("fc1_library");
        if (source.contains("scales")) scale_values = nb::cast<std::map<std::string, float>>(source["scales"]);
        for (auto &value : scale_values) scales.push_back({sizeof(lffw_scale), value.first.c_str(), value.second});
        for (auto &value : scales) scale_pointers.push_back(&value);
        options.struct_size = sizeof(options); options.abi_major = LFFW_ABI_MAJOR; options.config = &config;
        if (precision == "fp32") options.precision = LFFW_FP32;
        else if (precision == "fp16") options.precision = LFFW_FP16;
        else if (precision == "fp8") options.precision = LFFW_FP8;
        else throw std::invalid_argument("unsupported model precision");
        options.device = device; options.sampled_batch = batch;
        options.weights = weight_pointers.data(); options.weight_count = weight_pointers.size();
        options.config_file = config_file.c_str(); options.tokenizer_file = tokenizer_file.c_str();
        options.scale_file = scale_file.c_str(); options.mel_filter_file = mel_filter_file.c_str();
        options.mel_weights = mel_weights.c_str(); options.fc1_library = fc1_library.c_str();
        options.scales = scale_pointers.data(); options.scale_count = scale_pointers.size();
        auto id = [&](const char *key) { return source.contains(key) && !source[key].is_none() ? nb::cast<int32_t>(source[key]) : -1; };
        options.no_speech_token = id("no_speech_token"); options.timestamp_begin_token = id("timestamp_begin_token");
        options.suppressed_tokens = suppressed.data(); options.suppressed_token_count = suppressed.size();
        options.begin_suppressed_tokens = begin.data(); options.begin_suppressed_token_count = begin.size();
    }
};
nb::object empty(std::vector<size_t> shape, const char *dtype) {
    return nb::module_::import_("numpy").attr("empty")(nb::cast(shape), nb::arg("dtype") = dtype);
}
template <class Handle> void opened(const Handle &value) {
    if (!value) throw rt::error(LFFW_CLOSED, "native handle is closed");
}
nb::object mel_numpy(bound_mel &self) {
    opened(self.value);
    auto out = empty({1, size_t(self.meta->config.n_mels), 3000}, "float32");
    auto data = nb::cast<array>(out); auto *pointer = static_cast<float *>(data.data()); auto count = data.size();
    { nb::gil_scoped_release release; rt::copy(self.value, pointer, count); }
    return out;
}
nb::object audio_numpy(bound_audio &self) {
    opened(self.value);
    auto out = empty({size_t(self.meta->config.audio_positions), size_t(self.meta->config.audio_state)},
                     self.meta->info.encoder_dtype == LFFW_FLOAT16 ? "float16" : "float32");
    auto data = nb::cast<array>(out); auto *pointer = data.data(); auto count = data.nbytes();
    { nb::gil_scoped_release release; rt::copy(self.value, pointer, count); }
    return out;
}
template <class Self, class Convert>
nb::object array_method(Self &self, nb::object dtype, nb::object copy, Convert convert) {
    auto result = convert(self);
    if (!dtype.is_none()) result = result.attr("astype")(dtype, nb::arg("copy") = !copy.is(nb::bool_(false)));
    return result;
}
}
NB_MODULE(_native, module) {
    auto exception = nb::steal<nb::object>(PyErr_NewException("listenflow.fastest_whisper._native.NativeError", PyExc_RuntimeError, nullptr));
    module.attr("NativeError") = exception;
    nb::register_exception_translator([](const std::exception_ptr &value, void *context) {
        try { std::rethrow_exception(value); }
        catch (const rt::error &error) {
            if (error.status() == LFFW_INVALID_ARGUMENT) { PyErr_SetString(PyExc_ValueError, error.what()); return; }
            auto type = nb::borrow<nb::object>(static_cast<PyObject *>(context));
            auto instance = type(error.what()); instance.attr("status") = nb::int_(error.status());
            PyErr_SetObject(type.ptr(), instance.ptr());
        }
    }, exception.ptr());
    module.def("abi_version", [] { auto value = rt::version(); return nb::make_tuple(value.major, value.minor, value.patch); });
    module.attr("PREFIX") = nb::int_(int32_t(LFFW_PREFIX)); module.attr("GREEDY") = nb::int_(int32_t(LFFW_GREEDY)); module.attr("SAMPLED") = nb::int_(int32_t(LFFW_SAMPLED));
    nb::class_<bound_model>(module, "Model")
        .def("__init__", [](bound_model *self, nb::dict source, const std::string &precision,
                            int32_t device, int32_t batch, std::vector<int32_t> suppressed,
                            std::vector<int32_t> begin) {
            source_input input(source, precision, device, batch, std::move(suppressed), std::move(begin));
            auto meta = std::make_shared<metadata>(); meta->config = input.config;
            nb::gil_scoped_release release;
            rt::model model(input.options); meta->info = model.info();
            new (self) bound_model{std::move(model), std::move(meta)};
        }, nb::arg("source"), nb::arg("precision"), nb::arg("device"), nb::arg("sampled_batch"),
           nb::arg("suppressed_tokens"), nb::arg("begin_suppressed_tokens"))
        .def("close", [](bound_model &self) { nb::gil_scoped_release release; self.value.close(); })
        .def_prop_ro("info", [](bound_model &self) {
            lffw_model_info info;
            { nb::gil_scoped_release release; info = self.value.info(); }
            nb::dict result;
            result["encoder_dtype"] = info.encoder_dtype == LFFW_FLOAT16 ? "float16" : "float32";
            result["decoder_dtype"] = info.decoder_dtype == LFFW_FLOAT16 ? "float16" : "float32";
            nb::dict seconds, paths;
            seconds["mel"] = info.mel_open_seconds; seconds["encoder"] = info.encoder_open_seconds; seconds["decoder"] = info.decoder_open_seconds;
            paths["mel"] = info.mel_engine_path; paths["encoder"] = info.encoder_engine_path; paths["decoder"] = info.decoder_engine_path;
            if (info.sampled_batch) { seconds["sampled_decoder"] = info.sampled_open_seconds; paths["sampled_decoder"] = info.sampled_engine_path; }
            result["open_seconds"] = seconds; result["engine_paths"] = paths; return result;
        })
        .def("mel", [](bound_model &self, waveform samples) {
            auto pointer = samples.data(); auto count = samples.size();
            nb::gil_scoped_release release; return bound_mel{self.value.mel(pointer, count), self.meta};
        }, nb::arg("samples").noconvert())
        .def("log_mel_block", [](bound_model &self, waveform samples) {
            auto out = empty({size_t(self.meta->config.n_mels), 3000}, "float32");
            auto data = nb::cast<array>(out); auto pointer = static_cast<float *>(data.data()); auto count = data.size();
            { nb::gil_scoped_release release; self.value.log_mel_block(samples.data(), samples.size(), pointer, count); }
            return out;
        }, nb::arg("samples").noconvert())
        .def("encode", [](bound_model &self, bound_mel &mel) {
            opened(mel.value); nb::gil_scoped_release release;
            return bound_audio{self.value.encode(mel.value), self.meta};
        })
        .def("encode_features", [](bound_model &self, features mel) {
            if (mel.shape(0) != size_t(self.meta->config.n_mels) || mel.shape(1) != size_t(self.meta->config.audio_positions * 2))
                throw std::invalid_argument("mel must match model feature shape");
            nb::gil_scoped_release release; return bound_audio{self.value.encode_features(mel.data(), mel.size()), self.meta};
        }, nb::arg("mel").noconvert());
    nb::class_<bound_mel>(module, "Mel")
        .def("numpy", &mel_numpy)
        .def("__array__", [](bound_mel &self, nb::object dtype, nb::object copy) { return array_method(self, dtype, copy, mel_numpy); },
             nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
        .def("close", [](bound_mel &self) { nb::gil_scoped_release release; rt::close(self.value); });
    nb::class_<bound_audio>(module, "EncodedAudio")
        .def("numpy", &audio_numpy)
        .def("__array__", [](bound_audio &self, nb::object dtype, nb::object copy) { return array_method(self, dtype, copy, audio_numpy); },
             nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
        .def("decoder", [](bound_audio &self, int32_t kind, int32_t batch) {
            opened(self.value); nb::gil_scoped_release release;
            return bound_decoder{rt::create_decoder(self.value, kind, batch), self.meta, batch};
        }, nb::arg("kind"), nb::arg("batch") = 1)
        .def("close", [](bound_audio &self) { nb::gil_scoped_release release; rt::close(self.value); });
    nb::class_<bound_prefix>(module, "Prefix")
        .def("close", [](bound_prefix &self) { nb::gil_scoped_release release; rt::close(self.value); });
    nb::class_<bound_decoder>(module, "Decoder")
        .def("prefix_step", [](bound_decoder &self, bound_prefix *parent, int32_t token) {
            opened(self.value); if (parent) opened(parent->value);
            auto out = empty({size_t(self.meta->config.vocab_size)}, self.meta->info.decoder_dtype == LFFW_FLOAT16 ? "float16" : "float32");
            auto data = nb::cast<array>(out); auto pointer = data.data(); auto bytes = data.nbytes();
            rt::prefix prefix;
            { nb::gil_scoped_release release; prefix = rt::prefix_step(self.value, parent ? &parent->value : nullptr, token, pointer, bytes); }
            return nb::make_tuple(bound_prefix{std::move(prefix), self.meta}, out);
        }, nb::arg("parent").none(), nb::arg("token"))
        .def("greedy_step", [](bound_decoder &self, int32_t token, bool first, bool score) {
            opened(self.value); lffw_selection result;
            { nb::gil_scoped_release release; result = rt::greedy_step(self.value, token, first, score); }
            return nb::make_tuple(result.token, result.logprob, result.no_speech_probability);
        }, nb::arg("token"), nb::arg("first"), nb::arg("score"))
        .def("sampled_step", [](bound_decoder &self, tokens input, features suppression, float temperature,
                                floats uniform, bool timestamps) {
            opened(self.value); const auto batch = size_t(self.batch);
            if (input.size() != batch || uniform.size() != batch || suppression.shape(0) != batch ||
                suppression.shape(1) != size_t(self.meta->config.vocab_size)) throw std::invalid_argument("sampled arrays differ from prepared batch");
            std::vector<lffw_selection> values(batch); std::vector<lffw_selection *> pointers;
            for (auto &value : values) { value.struct_size = sizeof(value); pointers.push_back(&value); }
            auto token_out = empty({batch}, "int32"), score_out = empty({batch}, "float32"), speech_out = empty({batch}, "float32");
            auto token_data = nb::cast<array>(token_out), score_data = nb::cast<array>(score_out), speech_data = nb::cast<array>(speech_out);
            { nb::gil_scoped_release release; rt::sampled_step(self.value, input.data(), batch, suppression.data(), suppression.size(),
                                                              temperature, uniform.data(), timestamps, pointers.data()); }
            for (size_t row = 0; row < batch; ++row) {
                static_cast<int32_t *>(token_data.data())[row] = values[row].token;
                static_cast<float *>(score_data.data())[row] = values[row].logprob;
                static_cast<float *>(speech_data.data())[row] = values[row].no_speech_probability;
            }
            return nb::make_tuple(token_out, score_out, speech_out);
        }, nb::arg("tokens").noconvert(), nb::arg("suppression").noconvert(), nb::arg("temperature"),
           nb::arg("uniform").noconvert(), nb::arg("timestamps"))
        .def("close", [](bound_decoder &self) { nb::gil_scoped_release release; rt::close(self.value); });
}
