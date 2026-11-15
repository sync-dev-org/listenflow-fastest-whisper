#include "runtime_wait.hpp"
#include <array>
#include <stdexcept>
#include <vector>

using namespace listenflow::fastest_whisper::runtime_detail;
static void expect(bool value, const char *why) { if (!value) throw std::runtime_error(why); }
int main() {
    // SP-LFFASTESTWHISPER-017: not-ready polling requests 50 us sleeps and
    // scopes a one-nanosecond slack, preserving values wider than 32 bits.
    unsigned long slack = (1ul << 34) + 123;
    std::vector<unsigned long> settings;
    std::vector<long> sleeps;
    int queries = 0;
    wait_hooks hooks;
    hooks.get_slack = [&] { return slack; };
    hooks.set_slack = [&](unsigned long value) { settings.push_back(value); slack = value; };
    hooks.sleep_ns = [&](long value) { sleeps.push_back(value); };
    poll_completion([&] { return ++queries == 3; }, hooks);
    expect(queries == 3, "completion was not queried until ready");
    expect(sleeps == std::vector<long>{50000, 50000}, "poll sleep differs from 50 us");
    expect(settings == std::vector<unsigned long>{1, (1ul << 34) + 123}, "slack not restored");
    settings.clear();
    try {
        poll_completion([]() -> bool { throw std::runtime_error("CUDA query failed"); }, hooks);
        return 1;
    } catch (const std::runtime_error &error) {
        expect(std::string(error.what()).find("CUDA query failed") != std::string::npos,
               "completion error lost");
    }
    expect(settings == std::vector<unsigned long>{1, (1ul << 34) + 123}, "failure leaked slack");
    hooks.set_slack = [&](unsigned long value) {
        if (value != 1) throw std::runtime_error("restore failed");
    };
    try {
        poll_completion([]() -> bool { throw std::runtime_error("primary query"); }, hooks);
        return 2;
    } catch (const std::runtime_error &error) {
        expect(std::string(error.what()).find("primary query") != std::string::npos &&
               std::string(error.what()).find("restore failed") != std::string::npos,
               "primary and restoration failures were not retained");
    }
    // The production OS boundary must retain the calling thread's actual slack.
    auto system = system_wait_hooks();
    const auto before = system.get_slack();
    system.set_slack((1ul << 34) + 123);
    poll_completion([] { return true; }, system);
    expect(system.get_slack() == (1ul << 34) + 123, "real Linux slack was truncated");
    system.set_slack(before);
    poll_completion([] { return true; }, system);
    expect(system.get_slack() == before, "real Linux slack changed after completion");

    // SP-LFFASTESTWHISPER-017: OS failures keep errno and restore slack even
    // if setting it took effect before the boundary reported an interruption.
    for (const auto *stage : {"read", "set", "sleep", "restore"}) {
        bool changed = false;
        unsigned long actual = before;
        wait_hooks failing{
            [&] {
                if (std::string(stage) == "read")
                    throw std::system_error(EACCES, std::generic_category(), "read slack");
                return actual;
            },
            [&](unsigned long value) {
                actual = value;
                if (value == 1) changed = true;
                if ((value == 1 && std::string(stage) == "set") ||
                    (value != 1 && std::string(stage) == "restore"))
                    throw std::system_error(EACCES, std::generic_category(), "set slack");
            },
            [](long) { throw std::system_error(EACCES, std::generic_category(), "sleep"); }};
        try {
            poll_completion([&] { return std::string(stage) != "sleep"; }, failing);
            return 3;
        } catch (const std::system_error &error) {
            expect(error.code().value() == EACCES, "OS error number was lost");
        }
        expect(actual == before && changed == (std::string(stage) != "read"),
               "OS failure leaked a changed timer slack");
    }
}
