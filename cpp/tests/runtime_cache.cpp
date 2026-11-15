#include "runtime_cache.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace listenflow::fastest_whisper::runtime_detail;
static size_t copied = 0;
extern "C" CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr target, unsigned char value, size_t bytes, CUstream) {
    std::memset(reinterpret_cast<void *>(target), value, bytes); return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *copy, CUstream) {
    copied += copy->WidthInBytes * copy->Height;
    for (size_t row = 0; row < copy->Height; ++row)
        std::memmove(reinterpret_cast<char *>(copy->dstDevice) + row * copy->dstPitch,
                     reinterpret_cast<const char *>(copy->srcDevice) + row * copy->srcPitch,
                     copy->WidthInBytes);
    return CUDA_SUCCESS;
}
static void expect(bool condition, const char *why) { if (!condition) throw std::runtime_error(why); }
int main() {
    // SP-LFFASTESTWHISPER-001, SP-LFFASTESTWHISPER-002: every row retains its
    // history and the appended token at every bucket boundary. CUDA I/O only
    // is observed here; padding/compaction/append are the production routines.
    constexpr size_t rows = 6, width = 3;
    for (size_t history = 1; history <= 448; ++history) {
        auto bucket = cache_bucket(history);
        std::vector<int> present(rows * (bucket + 1) * width, -99);
        std::vector<int> target(rows * 449 * width, -7);
        std::vector<int> compact(rows * (history + 1) * width, -8);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < history; ++column)
                for (size_t i = 0; i < width; ++i)
                    present[(row * (bucket + 1) + column) * width + i] = int(row * 10000 + column * 10 + i);
            for (size_t i = 0; i < width; ++i)
                present[(row * (bucket + 1) + bucket) * width + i] = int(row * 10000 + history * 10 + i);
        }
        cache_copies copy{
            [](auto target, auto pitch, auto source, auto source_pitch, auto bytes, auto count) {
                CUDA_MEMCPY2D spec{}; spec.dstDevice = target; spec.dstPitch = pitch;
                spec.srcDevice = source; spec.srcPitch = source_pitch; spec.WidthInBytes = bytes; spec.Height = count;
                cuda_check(cuMemcpy2DAsync(&spec, nullptr), "cache copy");
            },
            [](auto target, auto bytes) { cuda_check(cuMemsetD8Async(target, 0, bytes, nullptr), "cache zero"); }};
        compact_cache(copy, reinterpret_cast<CUdeviceptr>(compact.data()),
                      reinterpret_cast<CUdeviceptr>(present.data()), history, bucket, rows, width * sizeof(int));
        for (size_t row = 0; row < rows; ++row)
            for (size_t column = 0; column <= history; ++column)
                for (size_t i = 0; i < width; ++i)
                    expect(compact[(row * (history + 1) + column) * width + i] == int(row * 10000 + column * 10 + i),
                           "compacted row differs from the unpadded history");
        if (history == 448) continue;
        auto next = cache_bucket(history + 1);
        for (size_t row = 0; row < rows; ++row)
            std::memcpy(target.data() + row * bucket * width, present.data() + row * (bucket + 1) * width,
                        history * width * sizeof(int));
        copied = 0;
        advance_cache(copy, reinterpret_cast<CUdeviceptr>(target.data()),
                      reinterpret_cast<CUdeviceptr>(present.data()), history, bucket, rows, width * sizeof(int));
        expect(copied == rows * width * sizeof(int) * (next == bucket ? 1 : history + 1),
               "mutable cache copied the whole history between bucket changes");
        for (size_t row = 0; row < rows; ++row)
            for (size_t column = 0; column <= history; ++column)
                for (size_t i = 0; i < width; ++i)
                    expect(target[(row * next + column) * width + i] == int(row * 10000 + column * 10 + i),
                           "mutable cache lost a row at a bucket boundary");
    }
    try { cache_bucket(449); return 1; } catch (const std::invalid_argument &) {}
}
