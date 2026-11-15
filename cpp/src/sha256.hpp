#pragma once
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace listenflow::fastest_whisper::detail {
// Streaming SHA-256 avoids a hashing dependency and never loads weight files.
class sha256 {
    std::array<uint32_t, 8> h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<uint8_t, 64> block_{};
    size_t used_ = 0;
    uint64_t bytes_ = 0;
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void compress() {
        static constexpr uint32_t k[]{
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        uint32_t w[64]{};
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 4; ++j)
                w[i] = (w[i] << 8) | block_[4 * i + j];
        for (int i = 16; i < 64; ++i) {
            auto x = w[i - 15], y = w[i - 2];
            w[i] = w[i - 16] + (rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)) + w[i - 7] +
                   (rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10));
        }
        auto a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            auto t1 =
                h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            auto t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }

  public:
    void update(const void *data, size_t count) {
        auto p = static_cast<const uint8_t *>(data);
        bytes_ += count;
        while (count--) {
            block_[used_++] = *p++;
            if (used_ == 64) {
                compress();
                used_ = 0;
            }
        }
    }
    void field(const std::string &value) {
        uint8_t size[8];
        for (int i = 0; i < 8; ++i)
            size[i] = uint64_t(value.size()) >> (8 * i);
        update(size, 8);
        update(value.data(), value.size());
    }
    std::string finish() {
        uint64_t bits = bytes_ * 8;
        uint8_t end = 128, zero = 0;
        update(&end, 1);
        while (used_ != 56)
            update(&zero, 1);
        uint8_t length[8];
        for (int i = 0; i < 8; ++i)
            length[7 - i] = bits >> (8 * i);
        update(length, 8);
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (auto h : h_)
            out << std::setw(8) << h;
        return out.str();
    }
};
inline std::string file_digest(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::invalid_argument("cannot read auxiliary input: " + path);
    sha256 hash;
    std::array<char, 32768> buffer{};
    while (input) {
        input.read(buffer.data(), buffer.size());
        hash.update(buffer.data(), size_t(input.gcount()));
    }
    if (!input.eof())
        throw std::runtime_error("failed reading auxiliary input: " + path);
    return hash.finish();
}
} // namespace listenflow::fastest_whisper::detail
