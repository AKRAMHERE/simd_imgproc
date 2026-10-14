// dispatch.cpp - the public API: argument checks, choice of implementation, and splitting the image into
// horizontal bands for OpenMP.
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "internal.hpp"
#include "simdip/ops.hpp"

namespace simdip {

const char* name(Impl impl) {
    switch (impl) {
        case Impl::Scalar: return "scalar";
        case Impl::AutoVec: return "autovec";
        case Impl::Avx2: return "avx2";
    }
    return "?";
}

Impl parseImpl(const std::string& s) {
    if (s == "scalar") return Impl::Scalar;
    if (s == "autovec") return Impl::AutoVec;
    if (s == "avx2") return Impl::Avx2;
    throw std::invalid_argument("unknown implementation '" + s + "' (use scalar, autovec or avx2)");
}

bool avx2Supported() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

int maxThreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

namespace {

// Calls f(y0, y1) for contiguous bands of rows. With threads <= 1 (or no OpenMP) it is one plain call.
// Band t of n covers rows [h*t/n, h*(t+1)/n): sizes differ by at most one row, each thread works on
// memory that is contiguous, and the same thread count always gives the same split.
template <class F>
void forBands(int h, int threads, F f) {
#ifdef _OPENMP
    if (threads > 1 && h > 1) {
#pragma omp parallel num_threads(threads)
        {
            const int t = omp_get_thread_num(), n = omp_get_num_threads();
            const int y0 = static_cast<int>(static_cast<long long>(h) * t / n);
            const int y1 = static_cast<int>(static_cast<long long>(h) * (t + 1) / n);
            if (y0 < y1) f(y0, y1);
        }
        return;
    }
#else
    (void)threads;
#endif
    f(0, h);
}

void checkPair(const Image& src, const Image& dst, int srcChannels, const char* op) {
    const std::string o(op);
    if (src.empty() || dst.empty()) throw std::invalid_argument(o + ": empty image");
    if (src.channels() != srcChannels) throw std::invalid_argument(o + ": source must have " + std::to_string(srcChannels) + " channel(s)");
    if (dst.channels() != 1) throw std::invalid_argument(o + ": destination must have 1 channel");
    if (src.width() != dst.width() || src.height() != dst.height()) throw std::invalid_argument(o + ": size mismatch");
    const std::uint8_t* s0 = src.data();
    const std::uint8_t* s1 = src.data() + src.stride() * static_cast<std::size_t>(src.height());
    const std::uint8_t* d0 = dst.data();
    const std::uint8_t* d1 = dst.data() + dst.stride() * static_cast<std::size_t>(dst.height());
    if (s0 < d1 && d0 < s1) throw std::invalid_argument(o + ": source and destination memory overlap (in-place is not supported)");
}

void requireAvx2(Impl impl) {
    if (impl == Impl::Avx2 && !avx2Supported()) throw std::runtime_error("this CPU does not support AVX2+FMA");
}

}  // namespace

void rgbToGray(const Image& src, Image& dst, Exec ex) {
    checkPair(src, dst, 3, "rgbToGray");
    requireAvx2(ex.impl);
    switch (ex.impl) {
        case Impl::Scalar: forBands(src.height(), ex.threads, [&](int a, int b) { scalar::rgbToGrayRows(src, dst, a, b); }); break;
        case Impl::AutoVec: forBands(src.height(), ex.threads, [&](int a, int b) { autovec::rgbToGrayRows(src, dst, a, b); }); break;
        case Impl::Avx2: forBands(src.height(), ex.threads, [&](int a, int b) { avx2::rgbToGrayRows(src, dst, a, b); }); break;
    }
}

void sobel(const Image& src, Image& dst, Exec ex) {
    checkPair(src, dst, 1, "sobel");
    requireAvx2(ex.impl);
    switch (ex.impl) {
        case Impl::Scalar: forBands(src.height(), ex.threads, [&](int a, int b) { scalar::sobelRows(src, dst, a, b); }); break;
        case Impl::AutoVec: forBands(src.height(), ex.threads, [&](int a, int b) { autovec::sobelRows(src, dst, a, b); }); break;
        case Impl::Avx2: forBands(src.height(), ex.threads, [&](int a, int b) { avx2::sobelRows(src, dst, a, b); }); break;
    }
}

void convolve(const Image& src, Image& dst, const Kernel& k, Exec ex) {
    checkPair(src, dst, 1, "convolve");
    k.validate();
    requireAvx2(ex.impl);
    switch (ex.impl) {
        case Impl::Scalar: forBands(src.height(), ex.threads, [&](int a, int b) { scalar::convolveRows(src, dst, k, a, b); }); break;
        case Impl::AutoVec: forBands(src.height(), ex.threads, [&](int a, int b) { autovec::convolveRows(src, dst, k, a, b); }); break;
        case Impl::Avx2: forBands(src.height(), ex.threads, [&](int a, int b) { avx2::convolveRows(src, dst, k, a, b); }); break;
    }
}

namespace ablation {

void convolveAutoVecRowAcc(const Image& src, Image& dst, const Kernel& k, int threads) {
    checkPair(src, dst, 1, "convolveAutoVecRowAcc");
    k.validate();
    forBands(src.height(), threads, [&](int a, int b) { autovec::convolveRowsRowAcc(src, dst, k, a, b); });
}

void convolveAvx2PerTapConvert(const Image& src, Image& dst, const Kernel& k, int threads) {
    checkPair(src, dst, 1, "convolveAvx2PerTapConvert");
    k.validate();
    requireAvx2(Impl::Avx2);
    forBands(src.height(), threads, [&](int a, int b) { avx2::convolveRowsPerTap(src, dst, k, a, b); });
}

void convolveAvx2Acc4(const Image& src, Image& dst, const Kernel& k, int threads) {
    checkPair(src, dst, 1, "convolveAvx2Acc4");
    k.validate();
    requireAvx2(Impl::Avx2);
    forBands(src.height(), threads, [&](int a, int b) { avx2::convolveRowsAcc4(src, dst, k, a, b); });
}

void copyRows(const Image& src, Image& dst, int threads) {
    if (src.width() != dst.width() || src.height() != dst.height() || src.channels() != dst.channels())
        throw std::invalid_argument("copyRows: shape mismatch");
    forBands(src.height(), threads, [&](int a, int b) {
        for (int y = a; y < b; ++y) std::memcpy(dst.row(y), src.row(y), src.rowBytes());
    });
}

}  // namespace ablation
}  // namespace simdip
