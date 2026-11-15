#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace lffw_test {
inline std::uint16_t half_bits(double value)
{
    _Float16 half = static_cast<_Float16>(value);
    std::uint16_t bits{}; std::memcpy(&bits, &half, sizeof(bits)); return bits;
}
inline double half_value(std::uint16_t bits)
{
    _Float16 half{}; std::memcpy(&half, &bits, sizeof(bits)); return static_cast<double>(half);
}
inline double round_half(double value) { return half_value(half_bits(value)); }
inline double gelu(double dot, double bias)
{
    double x = round_half(round_half(dot) + bias);
    double scaled = round_half(x * round_half(1 / std::sqrt(2.0)));
    return round_half(round_half(x * .5) * round_half(1 + round_half(std::erf(scaled))));
}
struct Projection {
    static constexpr int width = 1280;
    int rows;
    std::vector<std::uint16_t> weight, bias, v_bias, x;
    explicit Projection(bool fc1) : rows(fc1 ? 5120 : 3840), weight(rows * width),
        bias(fc1 ? rows : width), v_bias(fc1 ? 0 : width), x(width) {
        std::mt19937 random(3318); std::normal_distribution<double> normal;
        std::uniform_real_distribution<double> uniform(-.5, .5);
        for (auto& v : weight) v = half_bits(normal(random) / std::sqrt(width));
        for (auto& v : bias) v = half_bits(uniform(random));
        for (auto& v : v_bias) v = half_bits(uniform(random));
        for (auto& v : x) v = half_bits(normal(random));
    }
    std::vector<double> reference(bool fc1) const {
        std::vector<double> result(rows);
        for (int row = 0; row < rows; ++row) {
            double dot = 0;
            for (int column = 0; column < width; ++column)
                dot += half_value(x[column]) * half_value(weight[row * width + column]);
            if (fc1) result[row] = gelu(dot, half_value(bias[row]));
            else result[row] = round_half(dot + (row < width ? half_value(bias[row]) : 0) +
                (row >= 2 * width ? half_value(v_bias[row - 2 * width]) : 0));
        }
        return result;
    }
};
} // namespace lffw_test
