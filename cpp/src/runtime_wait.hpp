#pragma once
#include <cerrno>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <system_error>
#include <time.h>
#include <unistd.h>
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
#include "runtime_diagnostics.hpp"
#endif

namespace listenflow::fastest_whisper::runtime_detail {
struct wait_hooks {
    std::function<unsigned long()> get_slack;
    std::function<void(unsigned long)> set_slack;
    std::function<void(long)> sleep_ns;
};
class wait_restore_error : public std::runtime_error {
  public:
    std::exception_ptr primary;
    wait_restore_error(std::exception_ptr failure, std::string message)
        : std::runtime_error(std::move(message)), primary(failure) {}
};
inline wait_hooks system_wait_hooks() {
    return {
        [] {
            auto result = syscall(SYS_prctl, PR_GET_TIMERSLACK, 0ul, 0ul, 0ul, 0ul);
            if (result == -1) throw std::system_error(errno, std::generic_category(), "PR_GET_TIMERSLACK");
            return static_cast<unsigned long>(result);
        },
        [](unsigned long value) {
            if (syscall(SYS_prctl, PR_SET_TIMERSLACK, value, 0ul, 0ul, 0ul) == -1)
                throw std::system_error(errno, std::generic_category(), "PR_SET_TIMERSLACK");
        },
        [](long ns) {
            timespec remaining{0, ns};
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
            while (true) {
                diagnostic::sleep observed(uint64_t(remaining.tv_sec) * 1000000000 + uint64_t(remaining.tv_nsec));
                if (nanosleep(&remaining, &remaining) == 0) break;
                if (errno != EINTR) throw std::system_error(errno, std::generic_category(), "nanosleep");
            }
#else
            while (nanosleep(&remaining, &remaining) == -1) {
                if (errno != EINTR) throw std::system_error(errno, std::generic_category(), "nanosleep");
            }
#endif
        }};
}
template <class Query>
void poll_completion(Query query, const wait_hooks &hooks) {
    auto previous = hooks.get_slack();
    try {
        hooks.set_slack(1);
        while (!query()) hooks.sleep_ns(50000);
    } catch (...) {
        auto primary = std::current_exception();
        try { hooks.set_slack(previous); }
        catch (const std::exception &restore) {
            std::string message = "completion failed";
            try { std::rethrow_exception(primary); }
            catch (const std::exception &failure) { message = failure.what(); }
            catch (...) {}
            throw wait_restore_error(primary, message + "; timer slack restoration failed: " + restore.what());
        }
        std::rethrow_exception(primary);
    }
    hooks.set_slack(previous);
}
}
