#pragma once
#include "weight_storage.hpp"
#include <NvInfer.h>
#include <listenflow/whisper_network.hpp>
#include <memory>

namespace listenflow::fastest_whisper::detail {
nvinfer1::Dims dims(const std::vector<int64_t> &shape);
void define_network(nvinfer1::INetworkDefinition &net, const tensoracle::weights_view &weights,
                    const model_source &source, graph_kind kind, precision dtype, int32_t batch,
                    const tensoracle::builder_options &declaration, weight_storage &storage);
} // namespace listenflow::fastest_whisper::detail
