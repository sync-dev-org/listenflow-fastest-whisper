#include <listenflow/whisper_runtime_diagnostics.h>
#include <cassert>
#include <cstring>
#include <thread>

// SP-LFFASTESTWHISPER-001, SP-LFFASTESTWHISPER-017: diagnostics preserve the
// real C ABI result, observe their own thread and release snapshots safely.
int main() {
    lffw_diagnostic_options options{sizeof(options), LFFW_DIAGNOSTIC_HOST};
    lffw_diagnostic_session *session = nullptr;
    assert(lffw_diagnostic_create(&options, &session, nullptr) == LFFW_SUCCESS);
    assert(lffw_diagnostic_begin(session, nullptr) == LFFW_SUCCESS);
    lffw_diagnostic_session *nested = nullptr;
    assert(lffw_diagnostic_create(&options, &nested, nullptr) == LFFW_SUCCESS);
    assert(lffw_diagnostic_begin(nested, nullptr) == LFFW_BUSY);
    std::thread other([] {
        lffw_version value{sizeof(value), 0, 0, 0};
        assert(lffw_get_version(&value, nullptr) == LFFW_SUCCESS);
        assert(value.major == LFFW_ABI_MAJOR);
    });
    other.join();
    lffw_version value{sizeof(value), 0, 0, 0};
    assert(lffw_get_version(&value, nullptr) == LFFW_SUCCESS);
    assert(value.major == LFFW_ABI_MAJOR);
    assert(lffw_diagnostic_end(session, nullptr) == LFFW_SUCCESS);
    lffw_diagnostic_counts counts{sizeof(counts), 0};
    assert(lffw_diagnostic_get_counts(session, &counts, nullptr) == LFFW_SUCCESS);
    assert(counts.operation_count == 1);
    lffw_diagnostic_operation operation{};
    operation.struct_size = sizeof(operation);
    assert(lffw_diagnostic_get_operation(session, 0, &operation, nullptr) == LFFW_SUCCESS);
    assert(std::strcmp(operation.name, "lffw_get_version") == 0);
    assert(operation.status == LFFW_SUCCESS && operation.end_ns >= operation.start_ns);
    assert(lffw_diagnostic_get_operation(session, 1, &operation, nullptr) == LFFW_INVALID_ARGUMENT);
    assert(lffw_diagnostic_destroy(session, nullptr) == LFFW_SUCCESS);
    assert(lffw_diagnostic_destroy(nested, nullptr) == LFFW_SUCCESS);
    assert(lffw_diagnostic_destroy(nullptr, nullptr) == LFFW_SUCCESS);
}
