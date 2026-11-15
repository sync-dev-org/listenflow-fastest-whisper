#include "network_graph.hpp"
#include <algorithm>
#include <dlfcn.h>
#include <limits>
#include <map>
#include <stdexcept>

namespace listenflow::fastest_whisper::detail {
using namespace nvinfer1;
using T = ITensor *;
using S = std::vector<int64_t>;
Dims dims(const S &shape) {
    if (shape.size() > 8)
        throw std::invalid_argument("TensorRT rank exceeds 8");
    Dims d{};
    d.nbDims = int32_t(shape.size());
    std::copy(shape.begin(), shape.end(), d.d);
    return d;
}
template <class L> L *checked(L *layer) {
    if (!layer)
        throw std::runtime_error("TensorRT rejected Whisper network layer");
    return layer;
}
static T out(ILayer *layer) { return checked(checked(layer)->getOutput(0)); }
static DataType native_type(tensoracle_dtype dtype) {
    switch (dtype) {
    case TENSORACLE_FLOAT32:
        return DataType::kFLOAT;
    case TENSORACLE_FLOAT16:
        return DataType::kHALF;
    case TENSORACLE_INT32:
        return DataType::kINT32;
    case TENSORACLE_BOOL:
        return DataType::kBOOL;
    default:
        throw std::invalid_argument("unsupported model I/O dtype");
    }
}
class graph {
    INetworkDefinition &net;
    const tensoracle::weights_view &weights;
    const model_source &source;
    const model_config &c;
    precision precision_;
    bool half_;
    int32_t batch;
    weight_storage &storage;
    std::map<std::string, T> inputs;

  public:
    graph(INetworkDefinition &n, const tensoracle::weights_view &w, const model_source &s,
          precision p, int32_t b, const tensoracle::builder_options &declaration,
          weight_storage &owner)
        : net(n), weights(w), source(s), c(s.config), precision_(p), half_(p != precision::fp32),
          batch(b), storage(owner) {
        for (auto &spec : declaration.io)
            if (spec.input)
                inputs.emplace(spec.name,
                               checked(net.addInput(spec.name.c_str(), native_type(spec.dtype),
                                                    dims(spec.shape))));
    }
    T constant(std::vector<float> value, const S &shape, bool fp32 = false) {
        return out(
            net.addConstant(dims(shape), storage.floating(std::move(value), half_ && !fp32)));
    }
    T integer(std::vector<int32_t> value, const S &shape) {
        return out(net.addConstant(dims(shape), storage.integer(std::move(value))));
    }
    T scalar(float value, int rank, bool fp32 = false) {
        return constant({value}, S(rank, 1), fp32);
    }
    std::vector<float> weight(const std::string &key, const S &shape) {
        return read_weight(weights, key, shape);
    }
    T binary(T x, T y, ElementWiseOperation op) { return out(net.addElementWise(*x, *y, op)); }
    T add(T x, T y) { return binary(x, y, ElementWiseOperation::kSUM); }
    T mul(T x, T y) { return binary(x, y, ElementWiseOperation::kPROD); }
    T cast(T x, DataType type) { return out(net.addCast(*x, type)); }
    T single(T x) { return half_ ? cast(x, DataType::kFLOAT) : x; }
    T half(T x) { return half_ ? cast(x, DataType::kHALF) : x; }
    T matrix(T x, T y, bool transpose = false) {
        return out(net.addMatrixMultiply(*x, MatrixOperation::kNONE, *y,
                                         transpose ? MatrixOperation::kTRANSPOSE
                                                   : MatrixOperation::kNONE));
    }
    T unary(T x, UnaryOperation op) { return out(net.addUnary(*x, op)); }
    T reduce(T x, ReduceOperation op, uint32_t axes = 4) {
        return out(net.addReduce(*x, op, axes, true));
    }
    T softmax(T x, uint32_t axes) {
        auto *layer = checked(net.addSoftMax(*x));
        layer->setAxes(axes);
        return out(layer);
    }
    T select(T test, T x, T y) { return out(net.addSelect(*test, *x, *y)); }
    T reshape(T x, const S &shape) {
        auto *layer = checked(net.addShuffle(*x));
        layer->setReshapeDimensions(dims(shape));
        return out(layer);
    }
    T gather(T x, T indices, int axis) { return out(net.addGather(*x, *indices, axis)); }
    T concat(const std::vector<T> &tensors, int axis) {
        auto *layer = checked(net.addConcatenation(tensors.data(), int(tensors.size())));
        layer->setAxis(axis);
        return out(layer);
    }
    T slice(T x, const S &start, const S &size) {
        return out(net.addSlice(*x, dims(start), dims(size), dims(S(size.size(), 1))));
    }
    void output(T x, const char *name) {
        x->setName(name);
        net.markOutput(*x);
    }
    T qdq(T x, T scale, const std::string &name, int axis) {
        auto *q = checked(net.addQuantize(*x, *scale, DataType::kFP8));
        q->setAxis(axis);
        q->setName((name + ".Q").c_str());
        auto *dq = checked(net.addDequantize(*out(q), *scale, DataType::kHALF));
        dq->setAxis(axis);
        dq->setName((name + ".DQ").c_str());
        return out(dq);
    }
    T linear(T x, const std::string &key, bool bias = true) {
        const int64_t in = x->getDimensions().d[x->getDimensions().nbDims - 1];
        auto view = weights.at((key + ".weight").c_str());
        if (view.rank != 2 || view.dimensions[1] != in)
            throw std::invalid_argument("linear weight shape differs: " + key);
        const auto rows = view.dimensions[0];
        auto w = weight(key + ".weight", {rows, in});
        std::vector<float> transposed(w.size());
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < in; ++j)
                transposed[j * rows + i] = w[i * in + j];
        auto wt = constant(std::move(transposed), {1, in, rows});
        if (precision_ == precision::fp8) {
            auto found = source.scales.find(key);
            if (found == source.scales.end() || !std::isfinite(found->second) || found->second <= 0)
                throw std::invalid_argument("missing or invalid activation scale for " + key);
            std::vector<float> scales(rows, 0);
            for (int64_t i = 0; i < rows; ++i) {
                for (int64_t j = 0; j < in; ++j)
                    scales[i] = std::max(scales[i], std::abs(w[i * in + j]));
                scales[i] /= 448.f;
                if (!(scales[i] > 0))
                    scales[i] = 1;
            }
            wt = qdq(wt, constant(std::move(scales), {rows}, true), key + ".weight", 2);
            x = qdq(x, constant({found->second}, {}, true), key + ".input", 0);
        }
        auto *layer =
            checked(net.addMatrixMultiply(*x, MatrixOperation::kNONE, *wt, MatrixOperation::kNONE));
        if (precision_ == precision::fp8)
            layer->setName((key + ".matmul").c_str());
        auto y = out(layer);
        if (bias)
            y = add(y, constant(weight(key + ".bias", {rows}), {1, 1, rows}));
        return y;
    }
    T norm(T x, const std::string &key) {
        auto scale =
            constant(weight(key + ".weight", {c.audio_state}), {1, 1, c.audio_state}, true);
        auto bias = constant(weight(key + ".bias", {c.audio_state}), {1, 1, c.audio_state}, true);
        auto *layer = checked(net.addNormalizationV2(*single(x), *scale, *bias, 4));
        layer->setEpsilon(1e-5f);
        return half(out(layer));
    }
    T gelu(T x) {
        int rank = x->getDimensions().nbDims;
        auto inv = scalar(float(std::pow(2., -.5)), rank);
        auto one = scalar(1, rank);
        auto factor = scalar(.5f, rank);
        // Preserve the Python graph's layer creation order as well as its
        // arithmetic edges; TensorRT can fuse fp16 GELU patterns differently.
        auto scaled = mul(x, inv);
        auto erf = unary(scaled, UnaryOperation::kERF);
        auto left = mul(x, factor);
        auto right = add(one, erf);
        return mul(left, right);
    }
    T fc1(T x, const std::string &key) {
        if (precision_ != precision::fp16)
            return gelu(linear(x, key));
        auto wt = constant(weight(key + ".weight", {c.text_ffn, c.audio_state}),
                           {c.text_ffn, c.audio_state});
        auto bias = constant(weight(key + ".bias", {c.text_ffn}), {c.text_ffn});
        struct library {
            void *value;
            ~library() {
                if (value)
                    dlclose(value);
            }
        } lib{dlopen(source.fc1_library.c_str(), RTLD_NOW | RTLD_LOCAL)};
        if (!lib.value)
            throw std::runtime_error("required fp16 fc1 library cannot be loaded");
        auto digest = static_cast<const char *>(dlsym(lib.value, "tensoracle_bundle_sha256"));
        if (!digest || std::char_traits<char>::length(digest) != 64)
            throw std::runtime_error("required fp16 fc1 bundle identity is invalid");
        auto *creator = dynamic_cast<IPluginCreatorV3One *>(
            getPluginRegistry()->getCreator("LffwFp16Fc1", "1", "listenflow.fastest_whisper"));
        if (!creator)
            throw std::runtime_error("required fp16 fc1 creator is unavailable");
        PluginField field{"bundle_sha256", digest, PluginFieldType::kCHAR, 64};
        PluginFieldCollection fields{1, &field};
        std::vector<T> rows;
        for (int i = 0; i < batch; ++i) {
            T row = batch > 1 ? slice(x, {i, 0, 0}, {1, 1, c.audio_state}) : x;
            storage.plugins.emplace_back(
                checked(creator->createPlugin("fc1", &fields, TensorRTPhase::kBUILD)));
            T args[]{row, wt, bias};
            auto *layer = checked(net.addPluginV3(args, 3, nullptr, 0, *storage.plugins.back()));
            layer->setName((key + ".gelu" + (batch > 1 ? "." + std::to_string(i) : "")).c_str());
            rows.push_back(out(layer));
        }
        return batch > 1 ? concat(rows, 0) : rows.front();
    }
    T split_heads(T x, int64_t heads) {
        auto *layer = checked(net.addShuffle(*x));
        layer->setReshapeDimensions(dims({0, 0, heads, c.audio_state / heads}));
        Permutation p{{0, 2, 1, 3}};
        layer->setSecondTranspose(p);
        return out(layer);
    }
    T merge_heads(T x) {
        auto *layer = checked(net.addShuffle(*x));
        Permutation p{{0, 2, 1, 3}};
        layer->setFirstTranspose(p);
        layer->setReshapeDimensions(dims({0, 0, c.audio_state}));
        return out(layer);
    }
    std::pair<T, T> project_kv(T context, const std::string &key, int64_t heads) {
        return {split_heads(linear(context, key + ".k_proj", false), heads),
                split_heads(linear(context, key + ".v_proj"), heads)};
    }
    T attend(T x, const std::string &key, int64_t heads, T keys, T values, T mask = nullptr) {
        auto query = split_heads(linear(x, key + ".q_proj"), heads);
        auto factor = scalar(float(std::pow(double(c.audio_state / heads), -.25)), 4);
        query = mul(query, factor);
        keys = mul(keys, factor);
        auto scores = single(matrix(query, keys, true));
        if (mask)
            scores = add(scores, mask);
        return linear(merge_heads(matrix(half(softmax(scores, 8)), values)), key + ".out_proj");
    }
    T attention(T x, const std::string &key, int64_t heads) {
        auto kv = project_kv(x, key, heads);
        return attend(x, key, heads, kv.first, kv.second);
    }
    void stack(const std::vector<T> &values, const char *name) {
        std::vector<T> expanded;
        for (T x : values) {
            auto d = x->getDimensions();
            S shape{1};
            shape.insert(shape.end(), d.d, d.d + d.nbDims);
            expanded.push_back(reshape(x, shape));
        }
        output(concat(expanded, 0), name);
    }
    T cache_layer(T x, int index) { return gather(x, integer({index}, {}), 0); }
    void encoder() {
        auto x = reshape(half(inputs.at("mel")), {0, 0, 0, 1});
        for (int layer_id = 1; layer_id <= 2; ++layer_id) {
            auto key = "model.encoder.conv" + std::to_string(layer_id);
            auto wt = storage.floating(
                weight(key + ".weight",
                       {c.audio_state, layer_id == 1 ? c.n_mels : c.audio_state, 3}),
                half_);
            auto bias = storage.floating(weight(key + ".bias", {c.audio_state}), half_);
            auto *layer =
                checked(net.addConvolutionNd(*x, int(c.audio_state), dims({3, 1}), wt, bias));
            layer->setPaddingNd(dims({1, 0}));
            layer->setStrideNd(dims({layer_id == 1 ? 1 : 2, 1}));
            x = gelu(out(layer));
        }
        auto *shuffle = checked(net.addShuffle(*x));
        Permutation p{{0, 2, 1, 3}};
        shuffle->setFirstTranspose(p);
        shuffle->setReshapeDimensions(dims({0, 0, 0}));
        x = out(shuffle);
        x = add(x, constant(weight("model.encoder.embed_positions.weight",
                                   {c.audio_positions, c.audio_state}),
                            {1, c.audio_positions, c.audio_state}));
        for (int i = 0; i < c.audio_layers; ++i) {
            auto key = "model.encoder.layers." + std::to_string(i);
            auto normalized = norm(x, key + ".self_attn_layer_norm");
            x = add(x, attention(normalized, key + ".self_attn", c.audio_heads));
            normalized = norm(x, key + ".final_layer_norm");
            x = add(x, linear(gelu(linear(normalized, key + ".fc1")), key + ".fc2"));
        }
        x = norm(x, "model.encoder.layer_norm");
        output(x, "encoder");
        std::vector<T> keys, values;
        for (int i = 0; i < c.text_layers; ++i) {
            auto kv = project_kv(x, "model.decoder.layers." + std::to_string(i) + ".encoder_attn",
                                 c.text_heads);
            keys.push_back(kv.first);
            values.push_back(kv.second);
        }
        stack(keys, "cross_keys");
        stack(values, "cross_values");
    }
    T token_indices() {
        std::vector<int32_t> ids(c.vocab_size);
        for (int i = 0; i < c.vocab_size; ++i)
            ids[i] = i;
        return integer(std::move(ids), {1, 1, c.vocab_size});
    }
    T select_token(T logits, T suppression) {
        auto scores = add(single(logits), suppression);
        auto equal =
            binary(scores, reduce(scores, ReduceOperation::kMAX), ElementWiseOperation::kEQUAL);
        auto indices = token_indices();
        auto sentinel = integer({int32_t(c.vocab_size)}, {1, 1, 1});
        return reduce(select(equal, indices, sentinel), ReduceOperation::kMIN);
    }
    T silence(T logits) {
        if (!source.no_speech)
            return scalar(std::numeric_limits<float>::quiet_NaN(), 3, true);
        auto distribution = softmax(single(logits), 4);
        auto index = integer({*source.no_speech}, {1});
        return gather(distribution, index, 2);
    }
    T greedy(T logits, T suppression, T token) {
        auto token_value = cast(token, DataType::kFLOAT);
        auto scores = add(single(logits), suppression);
        auto logprob =
            unary(reduce(softmax(scores, 4), ReduceOperation::kMAX), UnaryOperation::kLOG);
        auto no_speech = silence(logits);
        return concat({token_value, logprob, no_speech}, 2);
    }
    T sample(T logits, T suppression) {
        auto scores = add(single(logits), suppression);
        if (source.timestamp_begin) {
            auto centered =
                binary(scores, reduce(scores, ReduceOperation::kMAX), ElementWiseOperation::kSUB);
            auto weights = unary(centered, UnaryOperation::kEXP);
            std::vector<int32_t> ids(c.vocab_size);
            for (int i = 0; i < c.vocab_size; ++i)
                ids[i] = i >= *source.timestamp_begin;
            auto is_timestamp =
                cast(integer(std::move(ids), {1, 1, c.vocab_size}), DataType::kBOOL);
            auto zero = scalar(0, 3, true);
            auto timestamp_weights = select(is_timestamp, weights, zero);
            auto text_weights = select(is_timestamp, zero, weights);
            auto timestamp_mass = reduce(timestamp_weights, ReduceOperation::kSUM);
            auto text_maximum = reduce(text_weights, ReduceOperation::kMAX);
            auto wins = binary(timestamp_mass, text_maximum, ElementWiseOperation::kGREATER);
            wins = binary(wins, inputs.at("timestamps"), ElementWiseOperation::kAND);
            auto text_only = select(is_timestamp, scores,
                                    scalar(-std::numeric_limits<float>::infinity(), 3, true));
            scores = select(wins, text_only, scores);
        }
        auto centered =
            binary(scores, reduce(scores, ReduceOperation::kMAX), ElementWiseOperation::kSUB);
        auto scaled = binary(centered, inputs.at("temperature"), ElementWiseOperation::kDIV);
        auto weights = unary(scaled, UnaryOperation::kEXP);
        auto cumulative = out(net.addCumulative(*weights, *integer({2}, {}),
                                                CumulativeOperation::kSUM, false, false));
        auto total = slice(cumulative, {0, 0, c.vocab_size - 1}, {batch, 1, 1});
        auto eligible =
            binary(cumulative, mul(inputs.at("uniform"), total), ElementWiseOperation::kGREATER);
        auto indices = token_indices();
        auto sentinel = integer({int32_t(c.vocab_size - 1)}, {1, 1, 1});
        auto token = reduce(select(eligible, indices, sentinel), ReduceOperation::kMIN);
        auto *gathered = checked(net.addGatherV2(*centered, *token, GatherMode::kELEMENT));
        gathered->setGatherAxis(2);
        auto normalizer =
            unary(reduce(unary(centered, UnaryOperation::kEXP), ReduceOperation::kSUM),
                  UnaryOperation::kLOG);
        auto logprob = binary(out(gathered), normalizer, ElementWiseOperation::kSUB);
        auto no_speech =
            source.no_speech
                ? silence(logits)
                : constant(std::vector<float>(batch, std::numeric_limits<float>::quiet_NaN()),
                           {batch, 1, 1}, true);
        auto token_value = cast(token, DataType::kFLOAT);
        return concat({token_value, logprob, no_speech}, 2);
    }
    void decoder(bool sampled) {
        auto token_table =
            constant(weight("model.decoder.embed_tokens.weight", {c.vocab_size, c.audio_state}),
                     {c.vocab_size, c.audio_state});
        auto position_table = constant(
            weight("model.decoder.embed_positions.weight", {c.text_positions, c.audio_state}),
            {c.text_positions, c.audio_state});
        auto token_embedding = gather(token_table, inputs.at("tokens"), 0);
        auto position_embedding = gather(position_table, inputs.at("positions"), 0);
        auto x = add(token_embedding, position_embedding);
        std::vector<T> keys, values;
        for (int i = 0; i < c.text_layers; ++i) {
            auto key = "model.decoder.layers." + std::to_string(i);
            auto normalized = norm(x, key + ".self_attn_layer_norm");
            auto kv = project_kv(normalized, key + ".self_attn", c.text_heads);
            auto k = concat({cache_layer(inputs.at("self_keys"), i), kv.first}, 2);
            auto v = concat({cache_layer(inputs.at("self_values"), i), kv.second}, 2);
            keys.push_back(k);
            values.push_back(v);
            x = add(x,
                    attend(normalized, key + ".self_attn", c.text_heads, k, v, inputs.at("mask")));
            normalized = norm(x, key + ".encoder_attn_layer_norm");
            auto cross_keys = cache_layer(inputs.at("cross_keys"), i);
            auto cross_values = cache_layer(inputs.at("cross_values"), i);
            x = add(x, attend(normalized, key + ".encoder_attn", c.text_heads, cross_keys,
                              cross_values));
            normalized = norm(x, key + ".final_layer_norm");
            x = add(x, linear(fc1(normalized, key + ".fc1"), key + ".fc2"));
        }
        x = norm(x, "model.decoder.layer_norm");
        auto logits = matrix(x, reshape(token_table, {1, c.vocab_size, c.audio_state}), true);
        output(logits, "logits");
        if (sampled)
            output(sample(logits, inputs.at("suppression")), "sampled");
        else {
            auto token = select_token(logits, inputs.at("suppression"));
            output(token, "token");
            output(greedy(logits, inputs.at("suppression"), token), "greedy");
        }
        stack(keys, "present_keys");
        stack(values, "present_values");
    }
    void mel() {
        // Same n=20*n1+n2, k=k1+20*k2 factorization as the fp32 product.
        constexpr int frames = 3000, fft = 400, hop = 160;
        const double pi = std::acos(-1.);
        std::vector<int32_t> indices(frames * fft);
        for (int f = 0; f < frames; ++f)
            for (int n = 0; n < fft; ++n)
                indices[f * fft + n] = f * hop + n;
        auto gathered = gather(inputs.at("audio"), integer(std::move(indices), {frames, fft}), 0);
        std::vector<float> window(fft);
        for (int n = 0; n < fft; ++n)
            window[n] = float(.5 - .5 * std::cos(2 * pi * n / fft));
        auto signal =
            reshape(mul(gathered, constant(std::move(window), {1, fft}, true)), {frames, 20, 20});
        std::vector<float> cosine(400), sine(400), twiddle_cos(400), twiddle_sin(400);
        for (int i = 0; i < 20; ++i)
            for (int j = 0; j < 20; ++j) {
                double a = -2 * pi * i * j / 20;
                cosine[i * 20 + j] = float(std::cos(a));
                sine[i * 20 + j] = float(std::sin(a));
                a = -2 * pi * i * j / fft;
                twiddle_cos[i * 20 + j] = float(std::cos(a));
                twiddle_sin[i * 20 + j] = float(std::sin(a));
            }
        auto cos = constant(std::move(cosine), {1, 20, 20}, true);
        auto sin = constant(std::move(sine), {1, 20, 20}, true);
        auto real = matrix(cos, signal);
        auto imag = matrix(sin, signal);
        auto tc = constant(std::move(twiddle_cos), {1, 20, 20}, true);
        auto ts = constant(std::move(twiddle_sin), {1, 20, 20}, true);
        auto real_cos = mul(real, tc);
        auto imag_sin = mul(imag, ts);
        auto r = binary(real_cos, imag_sin, ElementWiseOperation::kSUB);
        auto real_sin = mul(real, ts);
        auto imag_cos = mul(imag, tc);
        auto im = add(real_sin, imag_cos);
        auto real_cosine = matrix(r, cos, true);
        auto imag_sine = matrix(im, sin, true);
        real = binary(real_cosine, imag_sine, ElementWiseOperation::kSUB);
        auto real_sine = matrix(r, sin, true);
        auto imag_cosine = matrix(im, cos, true);
        imag = add(real_sine, imag_cosine);
        auto positive = [&](T x) {
            auto *shuffle = checked(net.addShuffle(*x));
            Permutation p{{0, 2, 1}};
            shuffle->setFirstTranspose(p);
            shuffle->setReshapeDimensions(dims({frames, fft}));
            return slice(out(shuffle), {0, 0}, {frames, 201});
        };
        real = positive(real);
        imag = positive(imag);
        auto real_power = mul(real, real);
        auto imag_power = mul(imag, imag);
        auto power = add(real_power, imag_power);
        auto filter = constant(weight("mel_" + std::to_string(c.n_mels), {c.n_mels, 201}),
                               {c.n_mels, 201}, true);
        auto value = matrix(filter, power, true);
        value = binary(value, scalar(1e-10f, 2, true), ElementWiseOperation::kMAX);
        auto log_value = unary(value, UnaryOperation::kLOG);
        auto log_scale = scalar(float(1 / std::log(10.)), 2, true);
        auto log = mul(log_value, log_scale);
        output(log, "log_mel");
        auto peak = reduce(log, ReduceOperation::kMAX, 3);
        auto range = scalar(8, 2, true);
        auto floor = binary(peak, range, ElementWiseOperation::kSUB);
        log = binary(log, floor, ElementWiseOperation::kMAX);
        auto shifted = add(log, scalar(4, 2, true));
        auto scale = scalar(.25f, 2, true);
        output(mul(shifted, scale), "mel");
    }
};
void define_network(INetworkDefinition &net, const tensoracle::weights_view &weights,
                    const model_source &source, graph_kind kind, precision dtype, int32_t batch,
                    const tensoracle::builder_options &declaration, weight_storage &storage) {
    graph g(net, weights, source, kind == graph_kind::mel ? precision::fp32 : dtype, batch,
            declaration, storage);
    switch (kind) {
    case graph_kind::mel:
        g.mel();
        break;
    case graph_kind::encoder:
        g.encoder();
        break;
    case graph_kind::decoder:
        g.decoder(false);
        break;
    case graph_kind::sampled_decoder:
        g.decoder(true);
        break;
    }
}
} // namespace listenflow::fastest_whisper::detail
