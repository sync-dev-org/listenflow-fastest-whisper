#include <listenflow/whisper_network.hpp>
#include <type_traits>

namespace fw = listenflow::fastest_whisper;
static_assert(std::is_move_constructible_v<fw::whisper_engine>);
static_assert(std::is_move_constructible_v<fw::network_execution>);
static_assert(std::is_same_v<decltype(std::declval<const fw::engine_build &>().declaration()),
                             const fw::build_declaration &>);
int main() {
    fw::whisper_engine engine;
    fw::whisper_engine moved = std::move(engine);
    fw::network_execution execution;
    fw::network_execution moved_execution = std::move(execution);
    moved.close();
    moved_execution.close();
    fw::model_error error(101, "condition mismatch");
    return error.status() == 101 ? 0 : 1;
}
