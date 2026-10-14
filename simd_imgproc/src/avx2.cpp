// avx2.cpp - hand-vectorized kernels using AVX2 (+FMA) intrinsics.
// This file is compiled with -mavx2 -mfma (see Makefile) and is only ever CALLED after a run-time check
// that the CPU supports those instructions (dispatch.cpp). Nothing in here may run at program start-up
// (no global objects with constructors, only constexpr tables), otherwise a machine without AVX2 could
// crash before main().
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "internal.hpp"

namespace simdip {
namespace avx2 {

namespace {

inline int clampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Scalar versions used for the few pixels that the vector loops do not cover (borders, tails).
inline std::uint8_t sobelPixel(const std::uint8_t* a, const std::uint8_t* b, const std::uint8_t* c,
                               int xm, int x, int xp) {
    const int gx = (a[xp] - a[xm]) + 2 * (b[xp] - b[xm]) + (c[xp] - c[xm]);
    const int gy = (c[xm] + 2 * c[x] + c[xp]) - (a[xm] + 2 * a[x] + a[xp]);
    const int m = std::abs(gx) + std::abs(gy);
    return static_cast<std::uint8_t>(m > 255 ? 255 : m);
}

// =============================================================================================
// 1. RGB -> gray
// =============================================================================================
// Memory holds R0 G0 B0 R1 G1 B1 ... (interleaved). 16 pixels are 48 bytes = three 16-byte loads
// v0 = bytes 0..15, v1 = bytes 16..31, v2 = bytes 32..47. The R bytes sit at 0,3,6,...,45, the G bytes
// at 1,4,...,46, the B bytes at 2,5,...,47. PSHUFB (_mm_shuffle_epi8) picks bytes by index within ONE
// register and writes zero where the index has its top bit set. So for each channel we take, from each
// of the three registers, the bytes that belong to it, and OR the three partial results together.
struct DeinterleaveMasks {
    // m[channel][source register][destination byte] = index into that source register, or 0x80 = "write 0"
    std::uint8_t m[3][3][16];
};

constexpr DeinterleaveMasks makeMasks() {
    DeinterleaveMasks d{};
    for (int c = 0; c < 3; ++c)
        for (int s = 0; s < 3; ++s)
            for (int i = 0; i < 16; ++i) d.m[c][s][i] = 0x80;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 16; ++i) {
            const int g = 3 * i + c;              // byte position of channel c of pixel i in the 48 bytes
            d.m[c][g / 16][i] = static_cast<std::uint8_t>(g % 16);
        }
    return d;
}
alignas(16) constexpr DeinterleaveMasks kMasks = makeMasks();

}  // namespace

void rgbToGrayRows(const Image& src, Image& dst, int y0, int y1) {
    const int w = src.width();
    __m128i mask[3][3];
    for (int c = 0; c < 3; ++c)
        for (int s = 0; s < 3; ++s) mask[c][s] = _mm_load_si128(reinterpret_cast<const __m128i*>(kMasks.m[c][s]));
    const __m256i c77 = _mm256_set1_epi16(77), c150 = _mm256_set1_epi16(150), c29 = _mm256_set1_epi16(29),
                  c128 = _mm256_set1_epi16(128);

    for (int y = y0; y < y1; ++y) {
        const std::uint8_t* p = src.row(y);
        std::uint8_t* q = dst.row(y);
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            const std::uint8_t* s = p + 3 * x;      // reads s[0..47], inside the row because x+16 <= w
            const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
            const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 16));
            const __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 32));
            const __m128i r8 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(v0, mask[0][0]), _mm_shuffle_epi8(v1, mask[0][1])),
                                            _mm_shuffle_epi8(v2, mask[0][2]));
            const __m128i g8 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(v0, mask[1][0]), _mm_shuffle_epi8(v1, mask[1][1])),
                                            _mm_shuffle_epi8(v2, mask[1][2]));
            const __m128i b8 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(v0, mask[2][0]), _mm_shuffle_epi8(v1, mask[2][1])),
                                            _mm_shuffle_epi8(v2, mask[2][2]));
            // widen 16 bytes -> 16 x uint16. 77*255 + 150*255 + 29*255 + 128 = 65408 < 65536: no overflow
            // in 16 bits, so the multiply-adds can be done with plain 16-bit instructions.
            __m256i acc = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(r8), c77);
            acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(g8), c150));
            acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(b8), c29));
            acc = _mm256_add_epi16(acc, c128);
            acc = _mm256_srli_epi16(acc, 8);        // logical shift: values are 0..255 now
            const __m128i packed = _mm_packus_epi16(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q + x), packed);
        }
        for (; x < w; ++x) {                         // 0..15 leftover pixels
            const unsigned r = p[3 * x], g = p[3 * x + 1], b = p[3 * x + 2];
            q[x] = static_cast<std::uint8_t>((77u * r + 150u * g + 29u * b + 128u) >> 8);
        }
    }
}

// =============================================================================================
// 2. Sobel
// =============================================================================================
namespace {
inline __m256i load16(const std::uint8_t* p) {   // 16 bytes -> 16 x int16 (zero extended)
    return _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}
}  // namespace

void sobelRows(const Image& src, Image& dst, int y0, int y1) {
    const int w = src.width(), h = src.height();
    for (int y = y0; y < y1; ++y) {
        const std::uint8_t* a = src.row(y > 0 ? y - 1 : 0);
        const std::uint8_t* b = src.row(y);
        const std::uint8_t* c = src.row(y < h - 1 ? y + 1 : h - 1);
        std::uint8_t* q = dst.row(y);
        if (w == 1) {
            q[0] = sobelPixel(a, b, c, 0, 0, 0);
            continue;
        }
        q[0] = sobelPixel(a, b, c, 0, 0, 1);
        int x = 1;
        // 16 output pixels per iteration. The loads at x-1, x, x+1 read bytes up to x+16, so we need
        // x + 16 <= w - 1.  Everything is 16-bit: |Gx|,|Gy| <= 1020, |Gx|+|Gy| <= 2040 < 32768.
        for (; x + 16 <= w - 1; x += 16) {
            const __m256i a0 = load16(a + x - 1), a1 = load16(a + x), a2 = load16(a + x + 1);
            const __m256i b0 = load16(b + x - 1), b2 = load16(b + x + 1);
            const __m256i c0 = load16(c + x - 1), c1 = load16(c + x), c2 = load16(c + x + 1);
            const __m256i dxb = _mm256_sub_epi16(b2, b0);
            const __m256i gx = _mm256_add_epi16(_mm256_add_epi16(_mm256_sub_epi16(a2, a0), _mm256_sub_epi16(c2, c0)),
                                                _mm256_add_epi16(dxb, dxb));
            const __m256i top = _mm256_add_epi16(_mm256_add_epi16(a0, a2), _mm256_add_epi16(a1, a1));
            const __m256i bot = _mm256_add_epi16(_mm256_add_epi16(c0, c2), _mm256_add_epi16(c1, c1));
            const __m256i gy = _mm256_sub_epi16(bot, top);
            const __m256i mag = _mm256_add_epi16(_mm256_abs_epi16(gx), _mm256_abs_epi16(gy));
            // packus saturates int16 -> uint8, i.e. min(mag, 255) since mag >= 0.
            // It works inside each 128-bit lane, so the 16 results end up as qwords [0, 0', 1, 1'];
            // permute4x64 with 0x08 moves qwords 0 and 2 to the bottom.
            const __m256i pk = _mm256_permute4x64_epi64(_mm256_packus_epi16(mag, mag), 0x08);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q + x), _mm256_castsi256_si128(pk));
        }
        for (; x < w - 1; ++x) q[x] = sobelPixel(a, b, c, x - 1, x, x + 1);
        q[w - 1] = sobelPixel(a, b, c, w - 2, w - 1, w - 1);
    }
}

// =============================================================================================
// 3. Convolution (float, FMA)
// =============================================================================================
namespace {

// Turns 4 vectors of 8 int32 (pixels 0-7, 8-15, 16-23, 24-31) into 32 saturated bytes in pixel order.
inline __m256i packInt32To32Bytes(__m256i c0, __m256i c1, __m256i c2, __m256i c3) {
    const __m256i p01 = _mm256_packs_epi32(c0, c1);    // int32 -> int16 (saturating), per 128-bit lane
    const __m256i p23 = _mm256_packs_epi32(c2, c3);
    const __m256i p = _mm256_packus_epi16(p01, p23);   // int16 -> uint8 (saturating) => clamp to 0..255
    // dwords now are [px0-3, px8-11, px16-19, px24-27 | px4-7, px12-15, px20-23, px28-31]
    return _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

struct FreeDeleter {
    void operator()(void* p) const noexcept { std::free(p); }
};

// Replicated-border float copy of one source row: out[j] = src[clamp(j - r, 0, w-1)], j in [0, len).
void convertRowPadded(const std::uint8_t* s, int w, int r, int len, float* out) {
    int j = 0;
    const float left = static_cast<float>(s[0]);
    for (; j < r && j < len; ++j) out[j] = left;
    // middle part j in [r, r+w): vectorised u8 -> float conversion, 8 pixels at a time
    int x = 0;
    for (; x + 8 <= w && r + x + 8 <= len; x += 8) {
        const __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + x));
        _mm256_storeu_ps(out + r + x, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8)));
    }
    for (j = r + x; j < len; ++j) out[j] = static_cast<float>(s[clampInt(j - r, 0, w - 1)]);
}

// NV = number of 8-float vectors per block (block = 8*NV output pixels). 8 independent accumulators hide the
// 4-cycle FMA latency on CPUs that can start two FMAs per cycle; 4 is the ablation.
template <int NV>
void convolveRowsImpl(const Image& src, Image& dst, const Kernel& K, int y0, int y1) {
    constexpr int kBlock = 8 * NV;
    const int w = src.width(), h = src.height(), k = K.size, r = k / 2;
    const float* wt = K.w.data();
    const int wRound = (w + kBlock - 1) / kBlock * kBlock;
    const int RL = wRound + 2 * r;               // length of one padded float row

    // k slots of padded float rows; slotRow[s] = which source row currently lives in slot s
    std::unique_ptr<float, FreeDeleter> ring(static_cast<float*>(alignedAlloc(sizeof(float) * static_cast<std::size_t>(RL) * k, 64)));
    int slotRow[kMaxKernelSize];
    for (int s = 0; s < k; ++s) slotRow[s] = -1;

    for (int y = y0; y < y1; ++y) {
        int ids[kMaxKernelSize];
        const float* rowp[kMaxKernelSize];
        for (int j = 0; j < k; ++j) ids[j] = clampInt(y - r + j, 0, h - 1);
        for (int j = 0; j < k; ++j) {
            int slot = -1;
            for (int s = 0; s < k; ++s)
                if (slotRow[s] == ids[j]) { slot = s; break; }
            if (slot < 0) {                       // not cached: reuse a slot that no needed row lives in
                for (int s = 0; s < k && slot < 0; ++s) {
                    bool needed = false;
                    for (int t = 0; t < k; ++t) needed = needed || (slotRow[s] == ids[t]);
                    if (!needed) slot = s;
                }
                convertRowPadded(src.row(ids[j]), w, r, RL, ring.get() + static_cast<std::size_t>(slot) * RL);
                slotRow[slot] = ids[j];
            }
            rowp[j] = ring.get() + static_cast<std::size_t>(slot) * RL;
        }

        std::uint8_t* q = dst.row(y);
        for (int x = 0; x < w; x += kBlock) {
            __m256 acc[NV];
            for (int v = 0; v < NV; ++v) acc[v] = _mm256_setzero_ps();
            for (int j = 0; j < k; ++j) {
                const float* p = rowp[j] + x;
                const float* wk = wt + j * k;
                for (int i = 0; i < k; ++i) {
                    const __m256 wv = _mm256_broadcast_ss(wk + i);
                    for (int v = 0; v < NV; ++v) acc[v] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p + i + 8 * v), acc[v]);
                }
            }
            // round to nearest even (MXCSR default), pack with saturation, store
            __m256i out[NV / 4];
            for (int g = 0; g < NV / 4; ++g)
                out[g] = packInt32To32Bytes(_mm256_cvtps_epi32(acc[4 * g]), _mm256_cvtps_epi32(acc[4 * g + 1]),
                                            _mm256_cvtps_epi32(acc[4 * g + 2]), _mm256_cvtps_epi32(acc[4 * g + 3]));
            if (x + kBlock <= w) {
                for (int g = 0; g < NV / 4; ++g) _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + x + 32 * g), out[g]);
            } else {                              // last, partial block: never write past the row
                alignas(32) std::uint8_t tmp[kBlock];
                for (int g = 0; g < NV / 4; ++g) _mm256_store_si256(reinterpret_cast<__m256i*>(tmp + 32 * g), out[g]);
                std::memcpy(q + x, tmp, static_cast<std::size_t>(w - x));
            }
        }
    }
}

}  // namespace

void convolveRows(const Image& src, Image& dst, const Kernel& K, int y0, int y1) { convolveRowsImpl<8>(src, dst, K, y0, y1); }
void convolveRowsAcc4(const Image& src, Image& dst, const Kernel& K, int y0, int y1) { convolveRowsImpl<4>(src, dst, K, y0, y1); }

// ---- ablation: convert u8 -> float again for every tap (what a first attempt usually looks like) ----
void convolveRowsPerTap(const Image& src, Image& dst, const Kernel& K, int y0, int y1) {
    constexpr int NV = 8, kBlock = 8 * NV;
    const int w = src.width(), h = src.height(), k = K.size, r = k / 2;
    const float* wt = K.w.data();
    const int wRound = (w + kBlock - 1) / kBlock * kBlock;
    const int RL = wRound + 2 * r + 8;            // +8: the 8-byte loads below may read a few bytes past the last tap
    // one padded u8 row per needed source row, rebuilt for every output row (kept simple on purpose)
    std::vector<std::uint8_t> pad(static_cast<std::size_t>(RL) * k);
    for (int y = y0; y < y1; ++y) {
        for (int j = 0; j < k; ++j) {
            const std::uint8_t* s = src.row(clampInt(y - r + j, 0, h - 1));
            std::uint8_t* d = pad.data() + static_cast<std::size_t>(j) * RL;
            std::memset(d, s[0], static_cast<std::size_t>(r));                       // replicate left border
            std::memcpy(d + r, s, static_cast<std::size_t>(w));                      // the row itself
            std::memset(d + r + w, s[w - 1], static_cast<std::size_t>(RL - r - w)); // replicate right border + slack
        }
        std::uint8_t* q = dst.row(y);
        for (int x = 0; x < w; x += kBlock) {
            __m256 acc[NV];
            for (int v = 0; v < NV; ++v) acc[v] = _mm256_setzero_ps();
            for (int j = 0; j < k; ++j) {
                const std::uint8_t* p = pad.data() + static_cast<std::size_t>(j) * RL + x;
                for (int i = 0; i < k; ++i) {
                    const __m256 wv = _mm256_broadcast_ss(wt + j * k + i);
                    for (int v = 0; v < NV; ++v) {
                        const __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + i + 8 * v));
                        acc[v] = _mm256_fmadd_ps(wv, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8)), acc[v]);
                    }
                }
            }
            __m256i out[NV / 4];
            for (int g = 0; g < NV / 4; ++g)
                out[g] = packInt32To32Bytes(_mm256_cvtps_epi32(acc[4 * g]), _mm256_cvtps_epi32(acc[4 * g + 1]),
                                            _mm256_cvtps_epi32(acc[4 * g + 2]), _mm256_cvtps_epi32(acc[4 * g + 3]));
            alignas(32) std::uint8_t tmp[kBlock];
            for (int g = 0; g < NV / 4; ++g) _mm256_store_si256(reinterpret_cast<__m256i*>(tmp + 32 * g), out[g]);
            std::memcpy(q + x, tmp, static_cast<std::size_t>(std::min(kBlock, w - x)));
        }
    }
}

}  // namespace avx2
}  // namespace simdip
