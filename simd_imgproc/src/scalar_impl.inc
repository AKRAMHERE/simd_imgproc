// scalar_impl.inc - the plain C++ implementation. It is #included twice, inside different namespaces
// and compiled with different flags (see Makefile):
//     scalar_novec.cpp    -fno-tree-vectorize -fno-tree-slp-vectorize   -> namespace simdip::scalar
//     scalar_autovec.cpp  (nothing special: -O3 vectorizes if it can)   -> namespace simdip::autovec
// Same source, two compilations: the difference between them is exactly what the compiler's
// auto-vectorizer buys. Both are compiled with -ffp-contract=off so that "a*b + c" stays a multiply
// followed by an add (no hidden fused multiply-add), which keeps the two bit-identical.
//
// Conventions: row pointers are `const uint8_t* __restrict` (the output never overlaps the input;
// the public API checks that); borders replicate the edge pixel.

// (The needed standard headers - <cmath>, <cstdint>, <cstdlib> - are included by the .cpp files
//  before the namespace is opened; never #include a header inside a namespace.)

namespace {

inline int clampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline std::uint8_t sobelPixel(const std::uint8_t* a, const std::uint8_t* b, const std::uint8_t* c,
                               int xm, int x, int xp) {
    // a, b, c = rows y-1, y, y+1 ; xm, x, xp = columns x-1, x, x+1 (already clamped by the caller)
    const int gx = (a[xp] - a[xm]) + 2 * (b[xp] - b[xm]) + (c[xp] - c[xm]);
    const int gy = (c[xm] + 2 * c[x] + c[xp]) - (a[xm] + 2 * a[x] + a[xp]);
    const int m = std::abs(gx) + std::abs(gy);
    return static_cast<std::uint8_t>(m > 255 ? 255 : m);
}

inline std::uint8_t roundClampF(float v) {
    const long r = std::lrintf(v);   // round to nearest, ties to even (default rounding mode)
    return static_cast<std::uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
}

}  // namespace

void rgbToGrayRows(const Image& src, Image& dst, int y0, int y1) {
    const int w = src.width();
    for (int y = y0; y < y1; ++y) {
        const std::uint8_t* __restrict p = src.row(y);
        std::uint8_t* __restrict q = dst.row(y);
        for (int x = 0; x < w; ++x) {
            const unsigned r = p[3 * x], g = p[3 * x + 1], b = p[3 * x + 2];
            q[x] = static_cast<std::uint8_t>((77u * r + 150u * g + 29u * b + 128u) >> 8);
        }
    }
}

void sobelRows(const Image& src, Image& dst, int y0, int y1) {
    const int w = src.width(), h = src.height();
    for (int y = y0; y < y1; ++y) {
        const std::uint8_t* __restrict a = src.row(y > 0 ? y - 1 : 0);
        const std::uint8_t* __restrict b = src.row(y);
        const std::uint8_t* __restrict c = src.row(y < h - 1 ? y + 1 : h - 1);
        std::uint8_t* __restrict q = dst.row(y);
        if (w == 1) {
            q[0] = sobelPixel(a, b, c, 0, 0, 0);
            continue;
        }
        q[0] = sobelPixel(a, b, c, 0, 0, 1);
        for (int x = 1; x < w - 1; ++x) q[x] = sobelPixel(a, b, c, x - 1, x, x + 1);
        q[w - 1] = sobelPixel(a, b, c, w - 2, w - 1, w - 1);
    }
}

void convolveRows(const Image& src, Image& dst, const Kernel& K, int y0, int y1) {
    const int w = src.width(), h = src.height(), k = K.size, r = k / 2;
    const float* __restrict wt = K.w.data();
    const std::uint8_t* rows[kMaxKernelSize];
    for (int y = y0; y < y1; ++y) {
        for (int j = 0; j < k; ++j) rows[j] = src.row(clampInt(y - r + j, 0, h - 1));
        std::uint8_t* __restrict q = dst.row(y);
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            if (x >= r && x < w - r) {                       // interior: no clamping needed
                for (int j = 0; j < k; ++j) {
                    const std::uint8_t* s = rows[j] + (x - r);
                    for (int i = 0; i < k; ++i) acc += wt[j * k + i] * static_cast<float>(s[i]);
                }
            } else {                                         // left/right border: clamp the column
                for (int j = 0; j < k; ++j)
                    for (int i = 0; i < k; ++i)
                        acc += wt[j * k + i] * static_cast<float>(rows[j][clampInt(x - r + i, 0, w - 1)]);
            }
            q[x] = roundClampF(acc);
        }
    }
}
