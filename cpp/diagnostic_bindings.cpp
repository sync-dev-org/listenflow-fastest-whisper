#include <nanobind/nanobind.h>
#include <listenflow/whisper_runtime.hpp>
#include <listenflow/whisper_runtime_diagnostics.h>

namespace nb = nanobind;
namespace rt = listenflow::fastest_whisper::runtime;
namespace {
struct session {
    lffw_diagnostic_session *value = nullptr;
    explicit session(uint32_t flags) {
        lffw_diagnostic_options options{sizeof(options), flags};
        rt::invoke(lffw_diagnostic_create, &options, &value);
    }
    ~session() { (void)lffw_diagnostic_destroy(value, nullptr); }
    void begin() { rt::invoke(lffw_diagnostic_begin, value); }
    void end() { rt::invoke(lffw_diagnostic_end, value); }
    void close() { rt::invoke(lffw_diagnostic_destroy, value); value = nullptr; }
    nb::list records() const {
        lffw_diagnostic_counts counts{sizeof(counts), 0};
        rt::invoke(lffw_diagnostic_get_counts, value, &counts);
        nb::list results;
        for (size_t i = 0; i < counts.operation_count; ++i) {
            lffw_diagnostic_operation operation{}; operation.struct_size = sizeof(operation);
            rt::invoke(lffw_diagnostic_get_operation, value, i, &operation);
            nb::dict row;
            row["name"] = operation.name; row["sequence"] = operation.sequence;
            row["status"] = operation.status; row["start_ns"] = operation.start_ns; row["end_ns"] = operation.end_ns;
            row["decoder_kind"] = operation.decoder_kind;
            row["input_token"] = operation.token_valid & 1 ? nb::cast(operation.input_token) : nb::none();
            row["output_token"] = operation.token_valid & 2 ? nb::cast(operation.output_token) : nb::none();
            nb::list spans, sleeps;
            for (size_t j = 0; j < operation.span_count; ++j) {
                lffw_diagnostic_span span{}; span.struct_size = sizeof(span);
                rt::invoke(lffw_diagnostic_get_span, value, i, j, &span);
                nb::dict item;
                item["phase"] = span.phase; item["graph"] = span.graph; item["tensor"] = span.tensor;
                item["start_ns"] = span.start_ns; item["end_ns"] = span.end_ns; item["cpu_ns"] = span.cpu_ns;
                item["gpu_ms"] = span.gpu_time_valid ? nb::cast(span.gpu_ms) : nb::none();
                item["graph_capture"] = bool(span.graph_capture); item["graph_replay"] = bool(span.graph_replay);
                item["bucket"] = span.bucket; item["bytes"] = span.bytes; item["query_count"] = span.query_count;
                spans.append(item);
            }
            for (size_t j = 0; j < operation.sleep_count; ++j) {
                lffw_diagnostic_sleep sleep{}; sleep.struct_size = sizeof(sleep);
                rt::invoke(lffw_diagnostic_get_sleep, value, i, j, &sleep);
                nb::dict item;
                item["start_ns"] = sleep.start_ns; item["end_ns"] = sleep.end_ns;
                item["requested_ns"] = sleep.requested_ns;
                sleeps.append(item);
            }
            row["spans"] = spans; row["sleeps"] = sleeps; results.append(row);
        }
        return results;
    }
};
}
NB_MODULE(_lffw_diagnostics, module) {
    nb::class_<session>(module, "Session")
        .def(nb::init<uint32_t>(), nb::arg("flags") = 7)
        .def("begin", &session::begin).def("end", &session::end).def("close", &session::close)
        .def("__enter__", [](session &self) -> session & { self.begin(); return self; }, nb::rv_policy::reference_internal)
        .def("__exit__", [](session &self, nb::object, nb::object, nb::object) { self.end(); },
             nb::arg("exc_type").none(), nb::arg("exc_value").none(), nb::arg("traceback").none())
        .def_prop_ro("records", &session::records);
    module.attr("HOST") = nb::int_(1); module.attr("EVENTS") = nb::int_(2); module.attr("WAIT") = nb::int_(4);
}
