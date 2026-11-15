#pragma once
#include "runtime_cache.hpp"
#include <map>
#include <optional>

namespace listenflow::fastest_whisper::runtime_detail {
inline size_t dtype_bytes(tensor_dtype type) {
    switch (type) {
    case tensor_dtype::fp16: return 2;
    case tensor_dtype::boolean: return 1;
    default: return 4;
    }
}
struct graph_entry {
    network_execution execution;
    CUgraphExec graph = nullptr;
    std::vector<void *> addresses;
    explicit graph_entry(network_execution value) : execution(std::move(value)) {}
    ~graph_entry() { try { close(); } catch (...) {} }
    void close() {
        if (graph) { cuda_check(cuGraphExecDestroy(graph), "cuGraphExecDestroy"); graph = nullptr; }
        execution.close();
    }
    graph_entry(const graph_entry &) = delete;
};
class executor {
    CUcontext context_;
    buffer_pool &pool_;
    gpu_queue &queue_;
    std::map<std::string, buffer> buffers_;
    std::map<std::string, std::unique_ptr<allocation>> hosts_;
    std::map<size_t, std::unique_ptr<graph_entry>> graphs_;
    std::optional<network_execution> fixed_;
  public:
    engine_build build;
    whisper_engine engine;
    std::string path;
    // Content currently held in the fixed suppression input: none, normal,
    // begin, or zero. This describes one actual buffer, not a decoding policy.
    int suppression_content = -1;
    executor(CUcontext context, buffer_pool &pool, gpu_queue &queue,
             model_source source, graph_kind kind, precision dtype, int32_t batch)
        : context_(context), pool_(pool), queue_(queue), build(std::move(source), kind, dtype, batch),
          engine(build.open()), path(engine.path()) {
        if (kind == graph_kind::sampled_decoder) {
            for (auto &entry : build.create_sampled_executions(engine))
                graphs_.emplace(size_t(entry.history), std::make_unique<graph_entry>(std::move(entry.execution)));
        } else if (kind == graph_kind::mel || kind == graph_kind::encoder)
            fixed_.emplace(build.create_execution(engine));
    }
    ~executor() { try { close(); } catch (...) {} }
    const tensor_spec &spec(const std::string &name) const {
        for (const auto &value : build.declaration().io) if (value.name == name) return value;
        throw std::invalid_argument("unknown model tensor: " + name);
    }
    size_t capacity(const std::string &name) const {
        const auto &value = spec(name);
        auto shape = value.input ? value.maximum : value.shape;
        size_t count = 1;
        for (auto dimension : shape) {
            // Only decoder present_* outputs have a ranged output dimension.
            auto size = dimension < 0 ? size_t(cache_buckets.back() + 1) : size_t(dimension);
            count = multiply(count, size);
        }
        return multiply(count, dtype_bytes(value.dtype));
    }
    buffer storage(const std::string &name) {
        auto &value = buffers_[name];
        if (!value) value = pool_.acquire(capacity(name));
        return value;
    }
    allocation &host(const std::string &name, size_t bytes) {
        auto &value = hosts_[name];
        if (!value || value->bytes < bytes) value = std::make_unique<allocation>(context_, bytes, true);
        return *value;
    }
    void upload(const std::string &name, const void *data, size_t bytes) {
        require(bytes <= capacity(name), "tensor exceeds declared input capacity");
        auto target = storage(name);
        auto &staging = host(name, capacity(name));
        if (bytes) std::memcpy(staging.pointer(), data, bytes);
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
        diagnostic::gpu_span observed(queue_.diagnostic_events, queue_.stream, "h2d", build.declaration().name.c_str(), 0, name.c_str(), bytes);
#endif
        queue_.pending = true;
        cuda_check(cuMemcpyHtoDAsync(target->address, staging.pointer(), bytes, queue_.stream), "cuMemcpyHtoDAsync");
    }
    void download(const std::string &name, CUdeviceptr source, size_t bytes) {
        auto &staging = host(name, bytes);
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
        diagnostic::gpu_span observed(queue_.diagnostic_events, queue_.stream, "d2h", build.declaration().name.c_str(), 0, name.c_str(), bytes);
#endif
        queue_.pending = true;
        cuda_check(cuMemcpyDtoHAsync(staging.pointer(), source, bytes, queue_.stream), "cuMemcpyDtoHAsync");
    }
    void copy_host(const std::string &name, void *out, size_t bytes) {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
        diagnostic::span observed("host_result", build.declaration().name.c_str(), 0, name.c_str(), bytes);
#endif
        std::memcpy(out, hosts_.at(name)->pointer(), bytes);
    }
    void run(const std::map<std::string, CUdeviceptr> &overrides = {}, std::optional<size_t> history = {}) {
        std::vector<tensor_address> addresses;
        std::vector<void *> signature;
        for (const auto &value : build.declaration().io) {
            auto found = overrides.find(value.name);
            auto address = found == overrides.end() ? storage(value.name)->address : found->second;
            addresses.push_back({value.name, reinterpret_cast<void *>(address)});
            signature.push_back(reinterpret_cast<void *>(address));
        }
        queue_.pending = true;
        if (!history) {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
            diagnostic::gpu_span observed(queue_.diagnostic_events, queue_.stream, "enqueue", build.declaration().name.c_str());
#endif
            fixed_->enqueue(addresses, reinterpret_cast<void *>(queue_.stream));
            return;
        }
        auto &entry = graphs_[*history];
        if (entry && entry->graph && entry->addresses != signature) { entry->close(); entry.reset(); }
        if (!entry) entry = std::make_unique<graph_entry>(build.create_execution(engine, int64_t(*history)));
        if (!entry->graph) {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
            diagnostic::span observed("graph_capture", build.declaration().name.c_str(), *history);
            if (observed.owner) observed.owner->spans[observed.index].value.graph_capture = 1;
#endif
            entry->execution.enqueue(addresses, reinterpret_cast<void *>(queue_.stream));
            queue_.finish();
            cuda_check(cuStreamBeginCapture(queue_.stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), "cuStreamBeginCapture");
            CUgraph captured = nullptr;
            try {
                entry->execution.enqueue(addresses, reinterpret_cast<void *>(queue_.stream));
                cuda_check(cuStreamEndCapture(queue_.stream, &captured), "cuStreamEndCapture");
                cuda_check(cuGraphInstantiate(&entry->graph, captured, 0), "cuGraphInstantiate");
                cuda_check(cuGraphDestroy(captured), "cuGraphDestroy"); captured = nullptr;
                entry->addresses = signature;
            } catch (...) {
                if (!captured) (void)cuStreamEndCapture(queue_.stream, &captured);
                if (captured) (void)cuGraphDestroy(captured);
                entry.reset(); throw;
            }
        }
        queue_.pending = true;
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
        diagnostic::gpu_span observed(queue_.diagnostic_events, queue_.stream, "enqueue", build.declaration().name.c_str(), *history, "", 0, true);
#endif
        cuda_check(cuGraphLaunch(entry->graph, queue_.stream), "cuGraphLaunch");
    }
    void close() {
        // Outstanding work is drained by the model before entering here.
        cleanup_errors errors;
        for (auto entry = graphs_.begin(); entry != graphs_.end();) {
            if (!entry->second) { entry = graphs_.erase(entry); continue; }
            bool released = false;
            errors.attempt([&] { entry->second->close(); released = true; });
            if (released) entry = graphs_.erase(entry);
            else ++entry;
        }
        if (graphs_.empty()) fixed_.reset();
        for (auto host = hosts_.begin(); host != hosts_.end();) {
            if (!host->second) { host = hosts_.erase(host); continue; }
            bool released = false;
            errors.attempt([&] { host->second->close(); released = true; });
            if (released) host = hosts_.erase(host);
            else ++host;
        }
        // Preserve the existing host-before-device cleanup order, including
        // primary/additional failures. A failed scratch release retains engine.
        if (graphs_.empty()) errors.attempt([&] { engine.close(); buffers_.clear(); });
        errors.rethrow();
    }
    bool has_graphs() const { return !graphs_.empty(); }
};
}
