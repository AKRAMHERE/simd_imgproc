#include "simdip/kernel.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace simdip {

double Kernel::sumAbs() const {
    double s = 0;
    for (float v : w) s += std::fabs(static_cast<double>(v));
    return s;
}

double Kernel::sum() const {
    double s = 0;
    for (float v : w) s += static_cast<double>(v);
    return s;
}

void Kernel::validate() const {
    if (size < 1 || size > kMaxKernelSize || size % 2 == 0)
        throw std::invalid_argument("kernel size must be odd and in 1..63");
    if (w.size() != static_cast<std::size_t>(size) * size)
        throw std::invalid_argument("kernel weight count does not match its size");
    for (float v : w)
        if (!std::isfinite(v)) throw std::invalid_argument("kernel contains NaN or infinity");
    // The AVX2 path converts float sums to int32; keep |sum| (<= 255 * L1) far below 2^31
    // so that float->int conversion can never overflow and all implementations agree.
    if (sumAbs() * 255.0 > 1.0e9) throw std::invalid_argument("kernel weights too large (255 * sum|w| must be <= 1e9)");
}

namespace kernels {

static Kernel make(int size) {
    Kernel k;
    k.size = size;
    k.w.assign(static_cast<std::size_t>(size) * size, 0.0f);
    return k;
}

Kernel identity(int size) {
    Kernel k = make(size);
    k.w[static_cast<std::size_t>(size / 2) * size + size / 2] = 1.0f;
    return k;
}

Kernel box(int size) {
    Kernel k = make(size);
    const float v = 1.0f / static_cast<float>(size * size);
    for (float& x : k.w) x = v;
    return k;
}

Kernel gaussian(int size, double sigma) {
    Kernel k = make(size);
    const int r = size / 2;
    double total = 0;
    std::vector<double> g(k.w.size());
    for (int y = -r; y <= r; ++y)
        for (int x = -r; x <= r; ++x) {
            const double v = std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
            g[static_cast<std::size_t>(y + r) * size + (x + r)] = v;
            total += v;
        }
    for (std::size_t i = 0; i < g.size(); ++i) k.w[i] = static_cast<float>(g[i] / total);
    return k;
}

Kernel binomial(int size) {
    // Row n of Pascal's triangle, n = size-1: 1 2 1 / 1 4 6 4 1 / ... outer product, divided by (2^n)^2.
    std::vector<double> row(static_cast<std::size_t>(size), 0.0);
    row[0] = 1;
    for (int n = 1; n < size; ++n)
        for (int i = n; i >= 1; --i) row[static_cast<std::size_t>(i)] += row[static_cast<std::size_t>(i - 1)];
    const double norm = std::pow(2.0, 2.0 * (size - 1));
    Kernel k = make(size);
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            k.w[static_cast<std::size_t>(y) * size + x] =
                static_cast<float>(row[static_cast<std::size_t>(y)] * row[static_cast<std::size_t>(x)] / norm);
    return k;
}

Kernel sharpen3() {
    Kernel k = make(3);
    k.w = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    return k;
}

Kernel emboss3() {
    Kernel k = make(3);
    k.w = {-2, -1, 0, -1, 1, 1, 0, 1, 2};
    return k;
}

Kernel randomKernel(int size, unsigned seed) {
    Kernel k = make(size);
    std::uint32_t s = seed * 2654435761u + 977u;
    for (float& x : k.w) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        x = (static_cast<float>(s & 0xFFFF) / 32767.5f - 1.0f) / static_cast<float>(size);   // in [-1,1]/size
    }
    return k;
}

}  // namespace kernels
}  // namespace simdip
