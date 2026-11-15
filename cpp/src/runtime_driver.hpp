#pragma once
#include "runtime_wait.hpp"
#include <listenflow/whisper_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
#include "runtime_diagnostic_events.hpp"
#endif

namespace listenflow::fastest_whisper::runtime_detail {
class failure : public std::runtime_error {
  public:
    lffw_status status;
    failure(lffw_status code, std::string message) : std::runtime_error(std::move(message)), status(code) {}
};
class cleanup_error : public std::runtime_error {
  public:
    std::exception_ptr primary;
    cleanup_error(std::exception_ptr original, std::string message)
        : std::runtime_error(std::move(message)), primary(original) {}
};
class cleanup_errors {
    std::exception_ptr primary_;
    std::string message_;
  public:
    template <class Operation> void attempt(Operation operation) {
        try { operation(); }
        catch (...) {
            auto error = std::current_exception();
            std::string message = "unknown cleanup failure";
            try { std::rethrow_exception(error); }
            catch (const std::exception &value) { message = value.what(); }
            catch (...) {}
            if (!primary_) { primary_ = error; message_ = std::move(message); }
            else message_ += "; cleanup also failed: " + message;
        }
    }
    void rethrow() const {
        if (primary_) throw cleanup_error(primary_, message_);
    }
};
inline void require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}
inline void cuda_check(CUresult result, const char *operation) {
    if (result != CUDA_SUCCESS)
        throw failure(result == CUDA_ERROR_OUT_OF_MEMORY ? LFFW_OUT_OF_MEMORY : LFFW_CUDA_ERROR,
                      std::string(operation) + " failed: CUDA status " + std::to_string(result));
}
class context_guard {
    bool pushed_ = false;
  public:
    explicit context_guard(CUcontext context) {
        if (context) { cuda_check(cuCtxPushCurrent(context), "cuCtxPushCurrent"); pushed_ = true; }
    }
    ~context_guard() { if (pushed_) { CUcontext previous{}; (void)cuCtxPopCurrent(&previous); } }
    context_guard(const context_guard &) = delete;
};
struct allocation {
    CUcontext context;
    CUdeviceptr address = 0;
    size_t bytes;
    bool pinned;
    allocation(CUcontext owner, size_t capacity, bool host = false)
        : context(owner), bytes(std::max(size_t(1), capacity)), pinned(host) {
        context_guard active(context);
        if (pinned) {
            void *result = nullptr;
            cuda_check(cuMemHostAlloc(&result, bytes, 0), "cuMemHostAlloc");
            address = reinterpret_cast<CUdeviceptr>(result);
        } else cuda_check(cuMemAlloc(&address, bytes), "cuMemAlloc");
    }
    allocation(const allocation &) = delete;
    ~allocation() { try { close(); } catch (...) {} }
    void close() {
        if (!address) return;
        context_guard active(context);
        cuda_check(pinned ? cuMemFreeHost(reinterpret_cast<void *>(address)) : cuMemFree(address),
                   pinned ? "cuMemFreeHost" : "cuMemFree");
        address = 0;
    }
    void *pointer() const { return reinterpret_cast<void *>(address); }
};
using buffer = std::shared_ptr<allocation>;
class buffer_pool {
    CUcontext context_;
    std::vector<buffer> entries_;
  public:
    explicit buffer_pool(CUcontext context) : context_(context) {}
    buffer acquire(size_t bytes) {
        for (auto &entry : entries_)
            if (entry.use_count() == 1 && entry->address && entry->bytes >= bytes) return entry;
        auto result = std::make_shared<allocation>(context_, bytes);
        entries_.push_back(result);
        return result;
    }
    void close() {
        cleanup_errors errors;
        for (auto entry = entries_.begin(); entry != entries_.end();) {
            bool released = false;
            errors.attempt([&] { (*entry)->close(); released = true; });
            if (released) entry = entries_.erase(entry);
            else ++entry;
        }
        errors.rethrow();
    }
};
struct gpu_queue {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
    diagnostic::event_log diagnostic_events;
#endif
    CUstream stream = nullptr;
    CUevent completion = nullptr;
    bool pending = false;
    explicit gpu_queue() {
        cuda_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        auto result = cuEventCreate(&completion, CU_EVENT_DISABLE_TIMING);
        if (result != CUDA_SUCCESS) {
            (void)cuStreamDestroy(stream); stream = nullptr;
            cuda_check(result, "cuEventCreate");
        }
    }
    ~gpu_queue() { try { close(); } catch (...) {} }
    void finish() {
        if (!pending) return;
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
        {
        diagnostic::span observed("wait");
#endif
        cuda_check(cuEventRecord(completion, stream), "cuEventRecord");
        poll_completion([&] {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
            diagnostic::query();
#endif
            auto status = cuEventQuery(completion);
            if (status == CUDA_ERROR_NOT_READY) return false;
            cuda_check(status, "cuEventQuery"); return true;
        }, system_wait_hooks());
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
        }
        diagnostic_events.complete();
#endif
        pending = false;
    }
    void drain() {
        if (!pending) return;
        try { finish(); }
        catch (...) {
            auto primary = std::current_exception();
            // Only an already failed event/OS wait uses this safety fallback.
            try { cuda_check(cuStreamSynchronize(stream), "failure cleanup cuStreamSynchronize"); }
            catch (const std::exception &cleanup) {
                std::string message = "GPU completion failed";
                try { std::rethrow_exception(primary); }
                catch (const std::exception &original) { message = original.what(); }
                catch (...) {}
                throw cleanup_error(primary, message + "; cleanup also failed: " + cleanup.what());
            }
            pending = false;
            std::rethrow_exception(primary);
        }
    }
    void close() {
        cleanup_errors errors;
        errors.attempt([&] { drain(); });
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
        errors.attempt([&] {
            diagnostic_events.complete();
            if (!diagnostic_events.empty()) throw failure(LFFW_CUDA_ERROR, "diagnostic timing event release failed");
        });
#endif
        if (completion) errors.attempt([&] {
            cuda_check(cuEventDestroy(completion), "cuEventDestroy"); completion = nullptr;
        });
        if (stream) errors.attempt([&] {
            cuda_check(cuStreamDestroy(stream), "cuStreamDestroy"); stream = nullptr;
        });
        pending = false;
        errors.rethrow();
    }
    void zero(CUdeviceptr target, size_t bytes) {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
        diagnostic::gpu_span observed(diagnostic_events, stream, "zero", "", 0, "", bytes);
#endif
        pending = true; cuda_check(cuMemsetD8Async(target, 0, bytes, stream), "cuMemsetD8Async");
    }
    void copy(CUdeviceptr target, CUdeviceptr source, size_t bytes) {
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
        diagnostic::gpu_span observed(diagnostic_events, stream, "d2d", "", 0, "", bytes);
#endif
        pending = true; cuda_check(cuMemcpyDtoDAsync(target, source, bytes, stream), "cuMemcpyDtoDAsync");
    }
    void copy_rows(CUdeviceptr target, size_t target_pitch, CUdeviceptr source,
                   size_t source_pitch, size_t width, size_t rows) {
        if (!width || !rows) return;
#if defined(LFFW_RUNTIME_DIAGNOSTICS)
        diagnostic::gpu_span observed(diagnostic_events, stream, "d2d", "", 0, "", width * rows);
#endif
        CUDA_MEMCPY2D spec{};
        spec.srcMemoryType = CU_MEMORYTYPE_DEVICE; spec.srcDevice = source; spec.srcPitch = source_pitch;
        spec.dstMemoryType = CU_MEMORYTYPE_DEVICE; spec.dstDevice = target; spec.dstPitch = target_pitch;
        spec.WidthInBytes = width; spec.Height = rows;
        pending = true; cuda_check(cuMemcpy2DAsync(&spec, stream), "cuMemcpy2DAsync");
    }
};
inline size_t multiply(size_t a, size_t b) {
    require(!a || b <= SIZE_MAX / a, "tensor size overflows size_t"); return a * b;
}
}
