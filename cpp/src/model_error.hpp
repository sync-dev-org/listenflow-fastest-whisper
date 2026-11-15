#pragma once
#include <listenflow/whisper_network.hpp>
#include <tensoracle/tensoracle.hpp>
#include <utility>

namespace listenflow::fastest_whisper::detail {
// Distinguish driver failures owned by the network wrapper from backend errors
// so the inference C ABI can retain its existing CUDA and OOM statuses.
class cuda_error : public model_error {
  public:
    using model_error::model_error;
};
template <class F> decltype(auto) model_call(F &&function) {
    try {
        return std::forward<F>(function)();
    } catch (const tensoracle::error &error) {
        throw model_error(error.status(), error.what());
    }
}
} // namespace listenflow::fastest_whisper::detail
