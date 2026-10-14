#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "internal.hpp"
namespace simdip {
namespace autovec {
#include "scalar_impl.inc"

// ---- Rewrite of the convolution so the compiler is ABLE to vectorize it ------------------------
// The natural loop above computes one output pixel at a time with a sum over k*k taps. A strict
// floating-point sum may not be reordered by the compiler (that would change the rounding), so the
// auto-vectorizer cannot turn the inner reduction into SIMD without -ffast-math. This version keeps
// the SAME per-pixel order of additions (ky outer, kx inner) but makes the x loop the innermost one:
//     for each tap: acc[x] += w * pad[x + kx]      for all x       <- trivially vectorizable
// Each acc[x] still receives its additions in exactly the same order, so the result is bit-identical.
void convolveRowsRowAcc(const Image& src, Image& dst, const Kernel& K, int y0, int y1) {
    const int w = src.width(), h = src.height(), k = K.size, r = k / 2;
    const float* __restrict wt = K.w.data();
    std::vector<float> padStore(static_cast<std::size_t>(k) * (w + 2 * r));
    std::vector<float> acc(static_cast<std::size_t>(w));
    const int pl = w + 2 * r;
    for (int y = y0; y < y1; ++y) {
        // float copies of the k source rows with replicated left/right borders
        for (int j = 0; j < k; ++j) {
            const std::uint8_t* s = src.row(clampInt(y - r + j, 0, h - 1));
            float* __restrict d = padStore.data() + static_cast<std::size_t>(j) * pl;
            for (int x = 0; x < pl; ++x) d[x] = static_cast<float>(s[clampInt(x - r, 0, w - 1)]);
        }
        float* __restrict a = acc.data();
        for (int x = 0; x < w; ++x) a[x] = 0.0f;
        for (int j = 0; j < k; ++j) {
            const float* __restrict row = padStore.data() + static_cast<std::size_t>(j) * pl;
            for (int i = 0; i < k; ++i) {
                const float wv = wt[j * k + i];
                const float* __restrict s = row + i;
                for (int x = 0; x < w; ++x) a[x] += wv * s[x];
            }
        }
        std::uint8_t* __restrict q = dst.row(y);
        for (int x = 0; x < w; ++x) q[x] = roundClampF(a[x]);
    }
}

}  // namespace autovec
}  // namespace simdip
