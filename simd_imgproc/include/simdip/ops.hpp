// ops.hpp - the public API: three image operations, each available as
//   Impl::Scalar   plain C++ loops, compiled with auto-vectorization switched OFF
//   Impl::AutoVec  the same C++ source compiled with the compiler's auto-vectorizer ON
//   Impl::Avx2     hand-written AVX2 intrinsics (needs a CPU with AVX2 + FMA, checked at run time)
// and each can be run on 1 thread or split into horizontal bands over `threads` OpenMP threads.
//
// All operations treat the image border by REPLICATING the outermost pixels
// ("clamp to edge"): a pixel outside the image has the value of the nearest pixel inside.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "simdip/image.hpp"
#include "simdip/kernel.hpp"

namespace simdip {

enum class Impl { Scalar, AutoVec, Avx2 };

const char* name(Impl impl);
Impl parseImpl(const std::string& s);   // "scalar" | "autovec" | "avx2"; throws on anything else
bool avx2Supported();                   // CPU has AVX2 and FMA (run-time check)
int maxThreads();                       // what OpenMP would use by default (1 if built without OpenMP)

struct Exec {
    Impl impl = Impl::Scalar;
    int threads = 1;                    // 1 = run on the calling thread, no OpenMP region
};

// RGB (3 channels) -> gray (1 channel):  Y = (77 R + 150 G + 29 B + 128) >> 8.
// Integer arithmetic only; results are identical for every Impl and thread count.
void rgbToGray(const Image& src, Image& dst, Exec ex);

// Sobel edge strength of a gray image: min(255, |Gx| + |Gy|).  Integer, exact for every Impl.
void sobel(const Image& src, Image& dst, Exec ex);

// dst = clamp(round_half_even(sum_{ky,kx} w[ky][kx] * src[y+ky-r][x+kx-r]), 0, 255), float arithmetic.
// Accumulation order is fixed (ky outer, kx inner), so Impl::Avx2 is bit-identical to the
// FMA reference and Impl::Scalar/AutoVec are bit-identical to each other (see docs, numerics).
void convolve(const Image& src, Image& dst, const Kernel& k, Exec ex);

// `dst` must already have the right size and channel count (1 channel except as input to rgbToGray).
// All functions throw std::invalid_argument on mismatched sizes/channels, on src and dst sharing memory,
// or (convolve) on an unusable kernel. They never read or write outside the pixel bytes of a row.

// ------------------------------------------------------------------------------------------
// Slower or alternative variants kept for the experiments in the guide (not for production use).
namespace ablation {
// Scalar-source convolution rewritten so the compiler CAN vectorize it (accumulate whole rows).
void convolveAutoVecRowAcc(const Image& src, Image& dst, const Kernel& k, int threads);
// AVX2 convolution that converts u8->float again for every tap instead of once per row.
void convolveAvx2PerTapConvert(const Image& src, Image& dst, const Kernel& k, int threads);
// AVX2 convolution with 4 accumulators (32 px per block) instead of 8 (64 px per block).
void convolveAvx2Acc4(const Image& src, Image& dst, const Kernel& k, int threads);
// Multi-threaded memcpy of the pixel rows: the memory-bandwidth yardstick for the benchmarks.
void copyRows(const Image& src, Image& dst, int threads);
}  // namespace ablation

// ------------------------------------------------------------------------------------------
// Reference implementations: written for obviousness, not speed. Used by the tests.
namespace reference {
void rgbToGray(const Image& src, Image& dst);
void sobel(const Image& src, Image& dst);
// Convolution in float with explicit fused multiply-add in the same order as the AVX2 kernel:
// returns the un-rounded float sums (height*width values, row-major).
std::vector<float> convolveFmaFloat(const Image& src, const Kernel& k);
// Same sums in double precision (order identical), the "ground truth" for the error bound.
std::vector<double> convolveDouble(const Image& src, const Kernel& k);
// Round half to even and clamp to 0..255 (the same rule the kernels use).
std::uint8_t roundClamp(double v);
void convolve(const Image& src, Image& dst, const Kernel& k);   // convolveFmaFloat + roundClamp
}  // namespace reference

}  // namespace simdip
