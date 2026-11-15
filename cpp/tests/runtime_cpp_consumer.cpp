#include <listenflow/whisper_runtime.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace rt = listenflow::fastest_whisper::runtime;
static_assert(!std::is_copy_constructible_v<rt::model>);
static_assert(std::is_nothrow_move_constructible_v<rt::model>);
static_assert(std::is_nothrow_destructible_v<rt::model>);
static_assert(!std::is_copy_constructible_v<rt::prefix>);

/* SP-LFFASTESTWHISPER-031, SP-LFFASTESTWHISPER-033: the C++ consumer uses
 * only the model header, with owned diagnostics and fixed-width statuses. */
int main() {
    lffw_model_options options{};
    options.struct_size = sizeof(options);
    options.abi_major = LFFW_ABI_MAJOR + 1;
    try {
        rt::model model(options);
        return 1;
    } catch (const rt::error &error) {
        if (error.status() != LFFW_ABI_MISMATCH || std::string(error.what()).empty())
            return 2;
    }
    lffw_version version{};
    version.struct_size = sizeof(version);
    lffw_error *diagnostic = nullptr;
    auto status = lffw_get_version(&version, &diagnostic);
    rt::check(status, diagnostic);
    return version.major == LFFW_ABI_MAJOR ? 0 : 3;
}
