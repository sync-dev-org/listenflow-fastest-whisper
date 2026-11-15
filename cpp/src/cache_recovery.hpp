#pragma once
#include <tensoracle/engine.hpp>
#include <utility>

namespace listenflow::fastest_whisper::detail {
inline bool saved_engine_status(tensoracle_status status) {
    return status == TENSORACLE_CORRUPT_ENGINE || status == TENSORACLE_ENGINE_CONDITION_MISMATCH ||
           status == TENSORACLE_ENGINE_IO_MISMATCH;
}
// Open sets built at entry to the network callback, before user/model code can
// throw. A callback failure can never trigger saved-engine recovery.
template <class Open, class Probe>
auto recover_saved_engine(bool plugins, Open &&open, Probe &&probe) {
    bool built = false, probing = false;
    try {
        auto engine = open(false, built);
        probing = true;
        if (plugins)
            probe(engine);
        return engine;
    } catch (const tensoracle::error &failure) {
        const bool delayed = plugins && probing && failure.status() == TENSORACLE_INTERNAL_ERROR;
        if (built || (!delayed && !saved_engine_status(failure.status())))
            throw;
    }
    auto engine = open(true, built);
    if (plugins)
        probe(engine);
    return engine;
}
} // namespace listenflow::fastest_whisper::detail
