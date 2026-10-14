// reference.cpp - deliberately simple implementations used as the ground truth in tests.
// No optimisation: every pixel access goes through a function that clamps the coordinates.
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "simdip/ops.hpp"

namespace simdip {
namespace reference {

namespace {
inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline int at(const Image& img, int x, int y, int c = 0) {
    x = clampi(x, 0, img.width() - 1);
    y = clampi(y, 0, img.height() - 1);
    return img.row(y)[x * img.channels() + c];
}
}  // namespace

void rgbToGray(const Image& src, Image& dst) {
    for (int y = 0; y < src.height(); ++y)
        for (int x = 0; x < src.width(); ++x) {
            const int r = at(src, x, y, 0), g = at(src, x, y, 1), b = at(src, x, y, 2);
            dst.row(y)[x] = static_cast<std::uint8_t>((77 * r + 150 * g + 29 * b + 128) / 256);
        }
}

void sobel(const Image& src, Image& dst) {
    // Written from the definition with the two 3x3 kernels (correlation, replicate border).
    static const int kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static const int ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    for (int y = 0; y < src.height(); ++y)
        for (int x = 0; x < src.width(); ++x) {
            int gx = 0, gy = 0;
            for (int j = -1; j <= 1; ++j)
                for (int i = -1; i <= 1; ++i) {
                    const int v = at(src, x + i, y + j);
                    gx += kx[j + 1][i + 1] * v;
                    gy += ky[j + 1][i + 1] * v;
                }
            int m = std::abs(gx) + std::abs(gy);
            dst.row(y)[x] = static_cast<std::uint8_t>(m > 255 ? 255 : m);
        }
}

std::vector<float> convolveFmaFloat(const Image& src, const Kernel& K) {
    const int k = K.size, r = k / 2, w = src.width(), h = src.height();
    std::vector<float> out(static_cast<std::size_t>(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < k; ++i)
                    acc = std::fmaf(K.at(j, i), static_cast<float>(at(src, x - r + i, y - r + j)), acc);
            out[static_cast<std::size_t>(y) * w + x] = acc;
        }
    return out;
}

std::vector<double> convolveDouble(const Image& src, const Kernel& K) {
    const int k = K.size, r = k / 2, w = src.width(), h = src.height();
    std::vector<double> out(static_cast<std::size_t>(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double acc = 0.0;
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < k; ++i)
                    acc += static_cast<double>(K.at(j, i)) * static_cast<double>(at(src, x - r + i, y - r + j));
            out[static_cast<std::size_t>(y) * w + x] = acc;
        }
    return out;
}

std::uint8_t roundClamp(double v) {
    const double r = std::nearbyint(v);   // ties to even in the default rounding mode
    return static_cast<std::uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
}

void convolve(const Image& src, Image& dst, const Kernel& K) {
    const std::vector<float> s = convolveFmaFloat(src, K);
    for (int y = 0; y < src.height(); ++y)
        for (int x = 0; x < src.width(); ++x)
            dst.row(y)[x] = roundClamp(static_cast<double>(s[static_cast<std::size_t>(y) * src.width() + x]));
}

}  // namespace reference
}  // namespace simdip
