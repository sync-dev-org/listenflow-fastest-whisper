// Isolate each native test exactly as Python's GPU conftest isolates its children.
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace {
volatile sig_atomic_t interrupted = 0;
void interrupt(int signal) { interrupted = signal; }
void require(bool value, std::string const& message) { if (!value) throw std::runtime_error(message); }
struct Temporary {
    fs::path path;
    explicit Temporary(fs::path const& parent) {
        fs::create_directories(parent);
        std::string pattern = (parent / "run-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end()); writable.push_back('\0');
        require(mkdtemp(writable.data()), "cannot create test directory"); path = writable.data();
    }
    ~Temporary() {
        std::error_code error; fs::remove_all(path, error);
        if (error) std::cerr << "cannot remove test directory " << path << ": " << error.message() << '\n';
    }
    Temporary(Temporary const&) = delete;
};
std::vector<char> bytes(fs::path const& path)
{
    std::ifstream input(path, std::ios::binary); require(bool(input), "cannot open " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}
std::string text(fs::path const& path)
{
    auto value = bytes(path); return {value.begin(), value.end()};
}
void run(std::vector<std::string> const& arguments, fs::path const& root,
    std::string const& label, std::string const& observer = "", int seconds = 120)
{
    auto log = root / (label + ".log");
    pid_t child = fork(); require(child >= 0, "cannot fork fixture");
    if (!child) {
        setpgid(0, 0);
        auto cache = (root / "cache").string();
        if (setenv("XDG_CACHE_HOME", cache.c_str(), 1) != 0) _exit(126);
        if (!observer.empty()) {
            if (setenv("LD_PRELOAD", observer.c_str(), 1) != 0 ||
                setenv("LFFW_QKV_DRIVER_LOG", (root / "driver.log").c_str(), 1) != 0) _exit(126);
        }
        int file = open(log.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
        if (file < 0) _exit(126);
        if (dup2(file, STDOUT_FILENO) < 0 || dup2(file, STDERR_FILENO) < 0) _exit(126);
        close(file);
        std::vector<char*> argv;
        for (auto const& value : arguments) argv.push_back(const_cast<char*>(value.c_str()));
        argv.push_back(nullptr); execv(argv[0], argv.data()); _exit(127);
    }
    setpgid(child, child);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    int status{};
    for (;;) {
        auto result = waitpid(child, &status, WNOHANG);
        if (result == child) break;
        if (result < 0 && errno == EINTR) continue;
        if (result < 0 || interrupted || std::chrono::steady_clock::now() >= deadline) {
            kill(-child, SIGKILL); while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            throw std::runtime_error(label + " timed out or waitpid failed");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << label << ": " << text(log);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, label + " fixture failed");
}
void cleanup_test(fs::path const& parent)
{
    auto* initial = std::getenv("XDG_CACHE_HOME");
    bool was_set = initial != nullptr;
    std::string original = initial ? initial : "";
    for (bool timeout : {false, true}) {
        fs::path path;
        {
            Temporary temporary(parent); path = temporary.path; bool failed = false;
            try {
                if (timeout) run({"/bin/sleep", "10"}, path, "timeout", "", 1);
                else run({"/bin/false"}, path, "failure");
            } catch (std::runtime_error const&) { failed = true; }
            require(failed, "negative cleanup probe did not fail");
            require(fs::exists(path / (timeout ? "timeout.log" : "failure.log")), "probe did not execute child");
        }
        require(!fs::exists(path), "test directory remained after child failure or timeout");
    }
    auto* final = std::getenv("XDG_CACHE_HOME");
    require((final != nullptr) == was_set && std::string(final ? final : "") == original,
        "test supervisor changed its caller's cache environment");
}
} // namespace
int main(int argc, char** argv)
{
    try {
        struct sigaction action{}; action.sa_handler = interrupt; sigemptyset(&action.sa_mask);
        require(sigaction(SIGTERM, &action, nullptr) == 0 && sigaction(SIGINT, &action, nullptr) == 0,
            "cannot install test cleanup signal handlers");
        if (argc == 3 && std::string(argv[1]) == "--cleanup-self-test") { cleanup_test(argv[2]); return 0; }
        require(argc == 9 || argc == 10, "usage: scenario RUNTIME KIND SCENARIO BUNDLE PLUGIN LEGACY OBSERVER TEMP_ROOT [REAL_WEIGHTS]");
        std::string fixture = argv[1], kind = argv[2], scenario = argv[3];
        fs::path removed;
        {
            Temporary temporary(argv[8]); auto root = temporary.path; removed = root;
            std::vector<std::string> prepare{fixture,"prepare",kind,root.string()};
            if (argc == 10 && *argv[9]) prepare.emplace_back(argv[9]);
            run(prepare, root, "prepare");
            auto invoke = [&](std::string const& mode, char const* library, bool observe = false) {
                run({fixture,mode,kind,root.string(),argv[4],library}, root, mode, observe ? argv[7] : "");
            };
            if (scenario == "data") {
                // Preparing and validating safetensors/cases is CPU-only.
            } else if (scenario == "legacy") {
                invoke("legacy-build", argv[6]); invoke("legacy-reject", argv[5]);
            } else if (scenario == "reject") invoke("reject", argv[5]);
            else if (scenario == "rounding") invoke("rounding", argv[5]);
            else {
                require(scenario == "values" || scenario == "reuse" || scenario == "contexts", "unknown scenario");
                invoke("build", argv[5]);
                if (scenario == "values" && kind == "bias") {
                    invoke("serialized", argv[5]);
                    require(bytes(root / "build.output") == bytes(root / "serialized.output"), "raw and cached graph outputs differ");
                }
                if (scenario == "reuse") {
                    auto engine = text(root / "build.path"); auto saved = bytes(engine);
                    invoke("reuse", argv[5]);
                    require(text(root / "reuse.path") == engine, "reused engine path changed");
                    require(bytes(engine) == saved, "reused engine bytes changed");
                    require(bytes(root / "reuse.output") == bytes(root / "build.output"), "reused constants or outputs changed");
                }
                if (scenario == "contexts") invoke("contexts", argv[5], true);
            }
        }
        require(!fs::exists(removed), "test temporary directory was not removed");
        std::cout << "temporary directory removed\n"; return 0;
    } catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
