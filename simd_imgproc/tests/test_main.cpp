// Tests for the SIMD image-processing library. No framework: a test is a function registered with TEST(name);
// CHECK(cond) counts and records failures. Exit code 0 only if every check passed.
//
// Layers (see the guide, chapter "How we know it is correct"):
//   1. hand-computed examples                      (are the formulas right?)
//   2. exhaustive / property tests                 (are they right for ALL inputs / do invariances hold?)
//   3. equivalence of every implementation vs the simple reference, over many sizes, strides, threads
//   4. numerical error bounds                      (is float32 accurate enough, and where can ties differ?)
//   5. robustness: padding untouched, bad arguments rejected; out-of-bounds access is caught by `make asan-test`
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "simdip/ops.hpp"

using namespace simdip;

// ------------------------------------------------------------------------------------------ mini framework
static long g_checks = 0, g_failures = 0;
static const char* g_current = "";
struct TestCase { const char* name; void (*fn)(); };
static std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; }
struct Registrar { Registrar(const char* n, void (*f)()) { registry().push_back({n, f}); } };

#define TEST(name)                                           \
    static void test_##name();                               \
    static Registrar reg_##name(#name, test_##name);         \
    static void test_##name()

#define CHECK(cond)                                                                              \
    do {                                                                                         \
        ++g_checks;                                                                              \
        if (!(cond)) {                                                                           \
            ++g_failures;                                                                        \
            if (g_failures <= 40) std::printf("  FAIL [%s] %s:%d: %s\n", g_current, __FILE__, __LINE__, #cond); \
        }                                                                                        \
    } while (0)

#define CHECK_MSG(cond, ...)                                                                     \
    do {                                                                                         \
        ++g_checks;                                                                              \
        if (!(cond)) {                                                                           \
            ++g_failures;                                                                        \
            if (g_failures <= 40) {                                                              \
                std::printf("  FAIL [%s] %s:%d: %s -- ", g_current, __FILE__, __LINE__, #cond);  \
                std::printf(__VA_ARGS__);                                                        \
                std::printf("\n");                                                               \
            }                                                                                    \
        }                                                                                        \
    } while (0)

#define CHECK_THROWS(expr)                                                                       \
    do {                                                                                         \
        ++g_checks;                                                                              \
        bool threw_ = false;                                                                     \
        try { expr; } catch (const std::exception&) { threw_ = true; }                           \
        if (!threw_) {                                                                           \
            ++g_failures;                                                                        \
            if (g_failures <= 40) std::printf("  FAIL [%s] %s:%d: expected an exception from %s\n", g_current, __FILE__, __LINE__, #expr); \
        }                                                                                        \
    } while (0)

// ------------------------------------------------------------------------------------------ helpers
static std::vector<Impl> impls() {
    std::vector<Impl> v = {Impl::Scalar, Impl::AutoVec};
    if (avx2Supported()) v.push_back(Impl::Avx2);
    return v;
}

// Sizes that sit around every vector-width boundary used by the kernels (16, 32, 64) plus degenerate shapes.
static const int kEdgeSizes[] = {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 18, 19, 31, 32, 33, 47, 63, 64, 65, 66, 100, 129};

static std::string describe(Impl i, int threads, int w, int h) {
    char b[96];
    std::snprintf(b, sizeof b, "impl=%s threads=%d size=%dx%d", name(i), threads, w, h);
    return b;
}

static bool firstDiff(const Image& a, const Image& b, int* ox, int* oy) {
    for (int y = 0; y < a.height(); ++y)
        for (int x = 0; x < a.width() * a.channels(); ++x)
            if (a.row(y)[x] != b.row(y)[x]) { *ox = x; *oy = y; return true; }
    return false;
}

// Compares and reports the first differing pixel (once per call).
static void checkSame(const Image& got, const Image& want, const std::string& what) {
    int x = 0, y = 0;
    const bool diff = firstDiff(got, want, &x, &y);
    CHECK_MSG(!diff, "%s: first difference at (%d,%d): got %d want %d", what.c_str(), x, y,
              diff ? got.row(y)[x] : 0, diff ? want.row(y)[x] : 0);
}

static Image transpose(const Image& s) {
    Image t(s.height(), s.width(), 1);
    for (int y = 0; y < s.height(); ++y)
        for (int x = 0; x < s.width(); ++x) t.row(x)[y] = s.row(y)[x];
    return t;
}

// ============================================================================================ 1. image / IO
TEST(image_rows_are_aligned_and_stride_policy) {
    for (int w : {1, 5, 63, 64, 65, 1000, 4096, 4095}) {
        for (int c : {1, 3}) {
            Image img(w, 3, c);
            CHECK(img.stride() >= img.rowBytes());
            CHECK(img.stride() % 64 == 0);
            CHECK(img.stride() % 4096 != 0);                  // never an exact multiple of the page size
            for (int y = 0; y < 3; ++y) CHECK(reinterpret_cast<std::uintptr_t>(img.row(y)) % 64 == 0);
        }
    }
    Image forced(4096, 2, 1, 4096);                           // explicit strides are honoured (used by an experiment)
    CHECK(forced.stride() == 4096);
    CHECK_THROWS(Image(10, 10, 1, 5));                        // stride < row
    CHECK_THROWS(Image(0, 10, 1));
    CHECK_THROWS(Image(10, 10, 2));
}

TEST(image_copy_is_deep) {
    Image a = makeSyntheticImage(37, 11, 1, 1);
    Image b = a;
    CHECK(a.samePixels(b));
    b.row(3)[5] ^= 0xFF;
    CHECK(!a.samePixels(b));
    CHECK(a.data() != b.data());
}

TEST(pnm_round_trip_and_comments) {
    const char* path = "/tmp/simdip_test_rt.ppm";
    Image rgb = makeSyntheticImage(21, 9, 3, 5);
    writePNM(rgb, path);
    Image back = readPNM(path);
    CHECK(back.channels() == 3 && back.width() == 21 && back.height() == 9);
    CHECK(rgb.samePixels(back));
    Image g = makeSyntheticImage(13, 4, 1, 6);
    writePNM(g, "/tmp/simdip_test_rt.pgm");
    CHECK(g.samePixels(readPNM("/tmp/simdip_test_rt.pgm")));
    // a header with a comment line
    FILE* f = std::fopen("/tmp/simdip_test_c.pgm", "wb");
    std::fputs("P5\n# a comment\n2 2\n255\n", f);
    const unsigned char px[4] = {1, 2, 3, 4};
    std::fwrite(px, 1, 4, f);
    std::fclose(f);
    Image c = readPNM("/tmp/simdip_test_c.pgm");
    CHECK(c.width() == 2 && c.row(1)[1] == 4);
    CHECK_THROWS(readPNM("/tmp/does_not_exist.pgm"));
}

// ============================================================================================ 2. grayscale
TEST(gray_known_values) {
    Image rgb(4, 1, 3);
    const unsigned char px[4][3] = {{255, 255, 255}, {0, 0, 0}, {255, 0, 0}, {0, 0, 255}};
    for (int i = 0; i < 4; ++i)
        for (int c = 0; c < 3; ++c) rgb.row(0)[3 * i + c] = px[i][c];
    // (77*255+128)>>8 = 77 ;  (29*255+128)>>8 = 29 ;  white = (65280+128)>>8 = 255
    const int want[4] = {255, 0, 77, 29};
    for (Impl im : impls()) {
        Image g(4, 1, 1);
        rgbToGray(rgb, g, {im, 1});
        for (int i = 0; i < 4; ++i) CHECK_MSG(g.row(0)[i] == want[i], "%s pixel %d got %d", name(im), i, g.row(0)[i]);
    }
    // pure green: (150*255+128)>>8 = 38378>>8 = 149
    Image green(1, 1, 3);
    green.row(0)[1] = 255;
    Image g1(1, 1, 1);
    rgbToGray(green, g1, {Impl::Scalar, 1});
    CHECK(g1.row(0)[0] == 149);
}

TEST(gray_exhaustive_all_16777216_colours) {
    // One image of 4096 x 4096 pixels in which every (R,G,B) triple occurs exactly once: proves the 16-bit
    // arithmetic of the AVX2 kernel (and the scalar code) correct for ALL possible inputs.
    Image rgb(4096, 4096, 3);
    for (int y = 0; y < 4096; ++y) {
        std::uint8_t* p = rgb.row(y);
        for (int x = 0; x < 4096; ++x) {
            const unsigned v = static_cast<unsigned>(y) * 4096u + static_cast<unsigned>(x);   // 24-bit counter
            p[3 * x] = static_cast<std::uint8_t>(v >> 16);
            p[3 * x + 1] = static_cast<std::uint8_t>(v >> 8);
            p[3 * x + 2] = static_cast<std::uint8_t>(v);
        }
    }
    Image g(4096, 4096, 1);
    for (Impl im : impls()) {
        rgbToGray(rgb, g, {im, 2});
        long bad = 0;
        for (int y = 0; y < 4096 && bad < 5; ++y)
            for (int x = 0; x < 4096; ++x) {
                const unsigned r = rgb.row(y)[3 * x], gg = rgb.row(y)[3 * x + 1], b = rgb.row(y)[3 * x + 2];
                if (g.row(y)[x] != ((77 * r + 150 * gg + 29 * b + 128) >> 8)) ++bad;
            }
        CHECK_MSG(bad == 0, "%s: %ld wrong pixels", name(im), bad);
    }
}

TEST(gray_matches_reference_edge_sizes_strides_threads) {
    for (int w : kEdgeSizes)
        for (int h : {1, 2, 5, 17}) {
            for (std::size_t extra : {std::size_t(0), std::size_t(1), std::size_t(7)}) {   // odd strides: rows not aligned
                Image src(w, h, 3, Image::defaultStride(w, 3) + extra);
                Image tmp = makeSyntheticImage(w, h, 3, 100 + w * 31 + h, static_cast<int>((w + h) % 3));
                for (int y = 0; y < h; ++y) std::memcpy(src.row(y), tmp.row(y), src.rowBytes());
                Image want(w, h, 1);
                reference::rgbToGray(src, want);
                for (Impl im : impls())
                    for (int t : {1, 2, 3, 8}) {
                        Image got(w, h, 1, Image::defaultStride(w, 1) + extra);
                        rgbToGray(src, got, {im, t});
                        checkSame(got, want, "gray " + describe(im, t, w, h));
                    }
            }
        }
}

// ============================================================================================ 3. Sobel
TEST(sobel_hand_computed_examples) {
    Image img(3, 3, 1);
    const int v[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x) img.row(y)[x] = static_cast<std::uint8_t>(v[y][x]);
    for (Impl im : impls()) {
        Image o(3, 3, 1);
        sobel(img, o, {im, 1});
        // centre: Gx = (3-1)+2(6-4)+(9-7) = 8 ; Gy = (7+16+9)-(1+4+3) = 24 ; |8|+|24| = 32
        CHECK_MSG(o.row(1)[1] == 32, "%s centre got %d", name(im), o.row(1)[1]);
        // corner (0,0), replicate border: rows (1 1 2 / 1 1 2 / 4 4 5): Gx = 1+2+1 = 4 ; Gy = (4+8+5)-(1+2+2) = 12 ; 16
        CHECK_MSG(o.row(0)[0] == 16, "%s corner got %d", name(im), o.row(0)[0]);
    }
    // vertical step edge 0 | 100: Gx = 4*100 = 400 -> saturates to 255 next to the edge, 0 elsewhere
    Image step(40, 6, 1);
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 40; ++x) step.row(y)[x] = x < 20 ? 0 : 100;
    for (Impl im : impls()) {
        Image o(40, 6, 1);
        sobel(step, o, {im, 1});
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 40; ++x) {
                const int want = (x == 19 || x == 20) ? 255 : 0;
                CHECK_MSG(o.row(y)[x] == want, "%s step (%d,%d) got %d", name(im), x, y, o.row(y)[x]);
            }
    }
    // a gentler step of 20: Gx = 80 at the two columns next to the edge
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 40; ++x) step.row(y)[x] = x < 20 ? 10 : 30;
    for (Impl im : impls()) {
        Image o(40, 6, 1);
        sobel(step, o, {im, 1});
        CHECK(o.row(3)[19] == 80 && o.row(3)[20] == 80 && o.row(3)[18] == 0 && o.row(3)[21] == 0);
    }
}

TEST(sobel_constant_image_is_zero) {
    for (int w : {1, 5, 18, 33, 64})
        for (Impl im : impls())
            for (int val : {0, 77, 255}) {
                Image c(w, 7, 1);
                c.fillPixels(static_cast<std::uint8_t>(val));
                Image o(w, 7, 1);
                o.fillAll(9);
                sobel(c, o, {im, 1});
                Image zero(w, 7, 1);
                checkSame(o, zero, "constant");
            }
}

TEST(sobel_matches_reference_edge_sizes_kinds_threads) {
    for (int w : kEdgeSizes)
        for (int h : {1, 2, 3, 6, 19}) {
            for (int kind : {0, 1, 2}) {      // photo-like, noise, 0/255 noise (saturates everywhere)
                Image src = makeSyntheticImage(w, h, 1, 7 * w + h + kind, kind);
                Image want(w, h, 1);
                reference::sobel(src, want);
                for (Impl im : impls())
                    for (int t : {1, 2, 5}) {
                        Image got(w, h, 1);
                        sobel(src, got, {im, t});
                        checkSame(got, want, "sobel " + describe(im, t, w, h) + " kind=" + std::to_string(kind));
                    }
            }
        }
}

TEST(sobel_symmetries_hold) {
    // Properties that follow from the maths and need no reference implementation:
    //  (a) transposing the input transposes the output (|Gx|+|Gy| is symmetric in x and y)
    //  (b) inverting the picture (255 - v) gives the same edge strength (gradients change sign, |.| does not)
    //  (c) adding a constant (without clipping) gives the same output (a gradient is a difference)
    for (Impl im : impls()) {
        Image a = makeSyntheticImage(70, 45, 1, 11, 0);
        Image sa(70, 45, 1), sta(45, 70, 1);
        sobel(a, sa, {im, 1});
        sobel(transpose(a), sta, {im, 1});
        checkSame(sta, transpose(sa), std::string("transpose ") + name(im));

        Image inv = a;
        for (int y = 0; y < 45; ++y)
            for (int x = 0; x < 70; ++x) inv.row(y)[x] = static_cast<std::uint8_t>(255 - a.row(y)[x]);
        Image si(70, 45, 1);
        sobel(inv, si, {im, 1});
        checkSame(si, sa, std::string("invert ") + name(im));

        Image lo = makeSyntheticImage(70, 45, 1, 12, 1);
        for (int y = 0; y < 45; ++y)
            for (int x = 0; x < 70; ++x) lo.row(y)[x] = static_cast<std::uint8_t>(lo.row(y)[x] / 2);   // 0..127
        Image sh = lo;
        for (int y = 0; y < 45; ++y)
            for (int x = 0; x < 70; ++x) sh.row(y)[x] = static_cast<std::uint8_t>(lo.row(y)[x] + 100); // 100..227
        Image s1(70, 45, 1), s2(70, 45, 1);
        sobel(lo, s1, {im, 1});
        sobel(sh, s2, {im, 1});
        checkSame(s2, s1, std::string("shift ") + name(im));
    }
}

// ============================================================================================ 4. convolution
static Kernel deltaKernel(int size, int ky, int kx) {
    Kernel k = kernels::identity(size);
    std::fill(k.w.begin(), k.w.end(), 0.0f);
    k.w[static_cast<std::size_t>(ky) * size + kx] = 1.0f;
    return k;
}

TEST(conv_identity_returns_the_input) {
    for (int k : {1, 3, 5, 9, 15})
        for (int w : {1, 3, 17, 64, 100})
            for (Impl im : impls()) {
                Image src = makeSyntheticImage(w, 13, 1, 3 * w + k, 1);
                Image o(w, 13, 1);
                convolve(src, o, kernels::identity(k), {im, 2});
                checkSame(o, src, "identity " + describe(im, 2, w, 13) + " k=" + std::to_string(k));
            }
}

TEST(conv_delta_kernel_shifts_the_image) {
    // weight 1 at (ky,kx) = (0,2) of a 3x3 kernel picks the pixel at (x+1, y-1): a shift with edge replication
    Image src = makeSyntheticImage(50, 21, 1, 9, 1);
    for (Impl im : impls()) {
        Image o(50, 21, 1);
        convolve(src, o, deltaKernel(3, 0, 2), {im, 1});
        for (int y = 0; y < 21; ++y)
            for (int x = 0; x < 50; ++x) {
                const int sx = std::min(x + 1, 49), sy = std::max(y - 1, 0);
                CHECK_MSG(o.row(y)[x] == src.row(sy)[sx], "%s (%d,%d)", name(im), x, y);
            }
    }
}

TEST(conv_blur_of_constant_image_is_constant_and_brightness_is_kept) {
    for (Impl im : impls())
        for (int k : {3, 5, 7})
            for (int val : {0, 1, 100, 254, 255}) {
                Image c(70, 9, 1);
                c.fillPixels(static_cast<std::uint8_t>(val));
                Image o(70, 9, 1);
                convolve(c, o, kernels::box(k), {im, 1});
                bool all = true;
                for (int y = 0; y < 9; ++y)
                    for (int x = 0; x < 70; ++x) all = all && (o.row(y)[x] == val);
                CHECK_MSG(all, "%s box k=%d value=%d", name(im), k, val);
            }
}

TEST(conv_binomial_kernels_are_exact_in_every_implementation) {
    // Binomial weights are multiples of 2^-2(k-1); for k <= 9 every product and partial sum is exactly representable
    // in float32, so the float result must equal the exact (double) result, ties-to-even included. This checks
    // rounding, ties and clamping with no tolerance at all, for scalar, autovec and AVX2 alike.
    for (int k : {3, 5, 7, 9}) {
        Kernel kern = kernels::binomial(k);
        for (int kind : {0, 1, 2}) {
            Image src = makeSyntheticImage(83, 29, 1, 50 + k + kind, kind);
            const std::vector<double> exact = reference::convolveDouble(src, kern);
            for (Impl im : impls()) {
                Image o(83, 29, 1);
                convolve(src, o, kern, {im, 3});
                long bad = 0;
                for (int y = 0; y < 29; ++y)
                    for (int x = 0; x < 83; ++x)
                        if (o.row(y)[x] != reference::roundClamp(exact[static_cast<std::size_t>(y) * 83 + x])) ++bad;
                CHECK_MSG(bad == 0, "%s binomial k=%d kind=%d: %ld wrong", name(im), k, kind, bad);
            }
        }
    }
}

TEST(conv_avx2_is_bit_identical_to_the_fma_reference) {
    if (!avx2Supported()) return;
    for (int k : {1, 3, 5, 7, 11, 15, 31}) {
        for (int w : {1, 2, 7, 31, 32, 33, 63, 64, 65, 70, 130}) {
            for (int h : {1, 2, 4, 9, 20}) {
                Image src = makeSyntheticImage(w, h, 1, 17 * w + h + k, (w + h + k) % 3);
                Kernel kern = kernels::randomKernel(k, 5 * k + w + h);
                if (k >= 15 && static_cast<long>(w) * h > 20 * 70) continue;   // keep the O(k^2) reference affordable
                Image want(w, h, 1);
                reference::convolve(src, want, kern);
                for (int t : {1, 2, 4}) {
                    Image got(w, h, 1);
                    convolve(src, got, kern, {Impl::Avx2, t});
                    checkSame(got, want, "conv avx2 " + describe(Impl::Avx2, t, w, h) + " k=" + std::to_string(k));
                }
                Image a4(w, h, 1), pt(w, h, 1);
                ablation::convolveAvx2Acc4(src, a4, kern, 1);
                ablation::convolveAvx2PerTapConvert(src, pt, kern, 1);
                checkSame(a4, want, "conv acc4 w=" + std::to_string(w) + " k=" + std::to_string(k));
                checkSame(pt, want, "conv pertap w=" + std::to_string(w) + " k=" + std::to_string(k));
            }
        }
    }
}

TEST(conv_scalar_autovec_and_rowacc_are_bit_identical) {
    // Same operations in the same order (multiply, then add; no FMA) => identical bits, however the compiler vectorizes.
    for (int k : {3, 5, 9, 13}) {
        for (int w : {1, 5, 16, 33, 77}) {
            for (int h : {1, 3, 12}) {
                Image src = makeSyntheticImage(w, h, 1, 31 * w + h + k, 1);
                Kernel kern = kernels::randomKernel(k, 99 + k + w);
                Image s(w, h, 1), a(w, h, 1), r(w, h, 1);
                convolve(src, s, kern, {Impl::Scalar, 1});
                convolve(src, a, kern, {Impl::AutoVec, 1});
                ablation::convolveAutoVecRowAcc(src, r, kern, 1);
                checkSame(a, s, "autovec vs scalar k=" + std::to_string(k));
                checkSame(r, s, "rowacc vs scalar k=" + std::to_string(k));
            }
        }
    }
}

TEST(conv_threads_do_not_change_the_result) {
    Image src = makeSyntheticImage(211, 97, 1, 77, 0);
    Kernel kern = kernels::gaussian(7, 1.5);
    for (Impl im : impls()) {
        Image one(211, 97, 1);
        convolve(src, one, kern, {im, 1});
        for (int t : {2, 3, 4, 7, 16, 200}) {      // 200 > number of rows: some threads get no band
            Image many(211, 97, 1);
            convolve(src, many, kern, {im, t});
            checkSame(many, one, std::string("threads ") + name(im) + " t=" + std::to_string(t));
        }
    }
}

// ============================================================================================ 5. numerics
TEST(conv_float_error_is_within_the_theoretical_bound) {
    // For a dot product of n terms evaluated with fused multiply-adds in float32 (unit roundoff u = 2^-24),
    // |computed - exact| <= gamma_n * sum|w_i x_i|, gamma_n = n u / (1 - n u). We verify it on random data
    // (using the un-rounded float sums of the FMA reference) and report the worst observed ratio.
    const double u = std::ldexp(1.0, -24);
    double worstRatio = 0;
    for (int k : {3, 5, 9, 15}) {
        for (int kind : {0, 1, 2}) {
            Image src = makeSyntheticImage(61, 37, 1, 123 + k + kind, kind);
            Kernel kern = kernels::randomKernel(k, 7 + k + kind);
            const std::vector<float> f = reference::convolveFmaFloat(src, kern);
            const std::vector<double> d = reference::convolveDouble(src, kern);
            const double n = static_cast<double>(k) * k, gamma = n * u / (1.0 - n * u);
            const double bound = gamma * 255.0 * kern.sumAbs();      // since x <= 255: sum|w x| <= 255 sum|w|
            for (std::size_t i = 0; i < f.size(); ++i) {
                const double err = std::fabs(static_cast<double>(f[i]) - d[i]);
                CHECK_MSG(err <= bound, "k=%d err=%g bound=%g", k, err, bound);
                worstRatio = std::max(worstRatio, err / bound);
            }
        }
    }
    CHECK(worstRatio < 1.0);
    std::printf("      worst observed error / theoretical bound = %.3f\n", worstRatio);
}

TEST(conv_scalar_differs_from_exact_rounding_only_next_to_ties) {
    // The scalar/autovec path adds a rounding error after each multiply AND each add (2n roundings, no FMA), so
    // it may round a value to the other side of a tie (x.5) compared with exact arithmetic. Check that this
    // happens only when the exact value lies within the error bound of a tie, and never by more than 1 level.
    const double u = std::ldexp(1.0, -24);
    long mismatches = 0, total = 0;
    for (int k : {3, 5, 9, 15}) {
        Image src = makeSyntheticImage(90, 40, 1, 321 + k, 0);
        Kernel kern = kernels::randomKernel(k, 700 + k);
        const std::vector<double> exact = reference::convolveDouble(src, kern);
        Image o(90, 40, 1);
        convolve(src, o, kern, {Impl::Scalar, 1});
        const double n = 2.0 * k * k, gamma = n * u / (1.0 - n * u);
        const double bound = gamma * 255.0 * kern.sumAbs();
        for (int y = 0; y < 40; ++y)
            for (int x = 0; x < 90; ++x) {
                const double e = exact[static_cast<std::size_t>(y) * 90 + x];
                const int want = reference::roundClamp(e), got = o.row(y)[x];
                ++total;
                if (got != want) {
                    ++mismatches;
                    CHECK(std::abs(got - want) == 1);
                    const double distToTie = std::fabs(std::fabs(e - std::floor(e)) - 0.5);
                    CHECK_MSG(distToTie <= bound, "k=%d exact=%.10f dist-to-tie=%g bound=%g", k, e, distToTie, bound);
                }
            }
    }
    std::printf("      scalar vs exact rounding: %ld of %ld pixels differ (%.4f %%)\n", mismatches, total, 100.0 * mismatches / total);
    CHECK(mismatches * 100 < total);   // far below 1 %
}

TEST(kernel_factories_and_validation) {
    CHECK(std::fabs(kernels::box(5).sum() - 1.0) < 1e-6);
    CHECK(std::fabs(kernels::gaussian(7, 1.2).sum() - 1.0) < 1e-6);
    CHECK(std::fabs(kernels::binomial(5).sum() - 1.0) < 1e-12);
    CHECK(kernels::sharpen3().sum() == 1.0);
    CHECK(kernels::identity(7).at(3, 3) == 1.0f);
    // binomial 3x3 must be [1 2 1; 2 4 2; 1 2 1] / 16
    const Kernel b3 = kernels::binomial(3);
    CHECK(b3.at(0, 0) == 1.0f / 16 && b3.at(1, 1) == 4.0f / 16 && b3.at(0, 1) == 2.0f / 16);
    Kernel even = kernels::box(3);
    even.size = 4;
    CHECK_THROWS(even.validate());
    Kernel nanK = kernels::box(3);
    nanK.w[4] = std::nanf("");
    CHECK_THROWS(nanK.validate());
    Kernel huge = kernels::box(3);
    for (float& v : huge.w) v = 1e9f;
    CHECK_THROWS(huge.validate());
    Kernel wrong = kernels::box(3);
    wrong.w.pop_back();
    CHECK_THROWS(wrong.validate());
}

// ============================================================================================ 6. robustness
TEST(outputs_never_touch_the_row_padding) {
    for (Impl im : impls())
        for (int w : {5, 33, 70, 129}) {
            const std::size_t stride = Image::defaultStride(w, 1) + 64;
            Image rgb = makeSyntheticImage(w, 11, 3, 4, 1);
            Image src = makeSyntheticImage(w, 11, 1, 5, 1);
            auto paddingIntact = [&](const Image& d) {
                for (int y = 0; y < d.height(); ++y)
                    for (std::size_t i = d.rowBytes(); i < d.stride(); ++i)
                        if (d.row(y)[i] != 0xAB) return false;
                return true;
            };
            Image d1(w, 11, 1, stride), d2(w, 11, 1, stride), d3(w, 11, 1, stride), d4(w, 11, 1, stride);
            d1.fillAll(0xAB); d2.fillAll(0xAB); d3.fillAll(0xAB); d4.fillAll(0xAB);
            rgbToGray(rgb, d1, {im, 2});
            sobel(src, d2, {im, 2});
            convolve(src, d3, kernels::gaussian(5, 1.0), {im, 2});
            convolve(src, d4, kernels::box(15), {im, 3});
            CHECK_MSG(paddingIntact(d1), "gray %s w=%d", name(im), w);
            CHECK_MSG(paddingIntact(d2), "sobel %s w=%d", name(im), w);
            CHECK_MSG(paddingIntact(d3), "conv5 %s w=%d", name(im), w);
            CHECK_MSG(paddingIntact(d4), "conv15 %s w=%d", name(im), w);
        }
}

TEST(tight_images_without_any_padding) {
    // Images whose stride equals the row size exactly and whose allocation ends right after the last row.
    // Results must still be right, and under `make asan-test` any load or store past the end is reported.
    for (int w : {1, 2, 3, 15, 16, 17, 18, 31, 32, 33, 34, 49, 50, 63, 64, 65, 97, 130})
        for (int h : {1, 2, 9}) {
            Image rgb(w, h, 3, static_cast<std::size_t>(w) * 3), src(w, h, 1, static_cast<std::size_t>(w));
            Image t1 = makeSyntheticImage(w, h, 3, w + h, 1), t2 = makeSyntheticImage(w, h, 1, w * 3 + h, 1);
            for (int y = 0; y < h; ++y) {
                std::memcpy(rgb.row(y), t1.row(y), rgb.rowBytes());
                std::memcpy(src.row(y), t2.row(y), src.rowBytes());
            }
            Image wantG(w, h, 1), wantS(w, h, 1), wantC(w, h, 1);
            reference::rgbToGray(rgb, wantG);
            reference::sobel(src, wantS);
            const Kernel k5 = kernels::binomial(5), k9 = kernels::binomial(9);
            reference::convolve(src, wantC, k5);
            for (Impl im : impls())
                for (int t : {1, 3}) {
                    Image g(w, h, 1, static_cast<std::size_t>(w)), s(w, h, 1, static_cast<std::size_t>(w)),
                          c(w, h, 1, static_cast<std::size_t>(w)), c9(w, h, 1, static_cast<std::size_t>(w));
                    rgbToGray(rgb, g, {im, t});
                    sobel(src, s, {im, t});
                    convolve(src, c, k5, {im, t});
                    convolve(src, c9, k9, {im, t});
                    checkSame(g, wantG, "tight gray " + describe(im, t, w, h));
                    checkSame(s, wantS, "tight sobel " + describe(im, t, w, h));
                    checkSame(c, wantC, "tight conv5 " + describe(im, t, w, h));
                }
        }
}

TEST(bad_arguments_are_rejected) {
    Image rgb = makeSyntheticImage(20, 20, 3, 1), gray = makeSyntheticImage(20, 20, 1, 2), small(10, 20, 1), out(20, 20, 1);
    const Exec ex{Impl::Scalar, 1};
    CHECK_THROWS(rgbToGray(gray, out, ex));                 // source must be RGB
    CHECK_THROWS(sobel(rgb, out, ex));                      // source must be gray
    CHECK_THROWS(sobel(gray, small, ex));                   // size mismatch
    CHECK_THROWS(sobel(gray, gray, ex));                    // in place
    CHECK_THROWS(convolve(gray, gray, kernels::box(3), ex));
    CHECK_THROWS(rgbToGray(rgb, rgb, ex));
    Image rgbOut(20, 20, 3);
    CHECK_THROWS(rgbToGray(rgb, rgbOut, ex));               // destination must be 1 channel
    Kernel even = kernels::box(3);
    even.size = 2;
    CHECK_THROWS(convolve(gray, out, even, ex));
    CHECK_THROWS(parseImpl("sse"));
    Image empty;
    CHECK_THROWS(sobel(empty, out, ex));
    sobel(gray, out, ex);                                   // and a valid call still works
}

TEST(large_image_matches_scalar_implementation) {
    // Too big for the naive reference; the scalar implementation (itself verified above) is the yardstick.
    Image src = makeSyntheticImage(2051, 1031, 1, 8, 0);
    Image rgb = makeSyntheticImage(2051, 1031, 3, 9, 0);
    Kernel kern = kernels::gaussian(9, 2.0);
    Image gs(2051, 1031, 1), ss(2051, 1031, 1), cs(2051, 1031, 1);
    rgbToGray(rgb, gs, {Impl::Scalar, 1});
    sobel(src, ss, {Impl::Scalar, 1});
    convolve(src, cs, kern, {Impl::Scalar, 1});
    for (Impl im : impls())
        for (int t : {1, 2}) {
            Image g(2051, 1031, 1), s(2051, 1031, 1), c(2051, 1031, 1);
            rgbToGray(rgb, g, {im, t});
            sobel(src, s, {im, t});
            convolve(src, c, kern, {im, t});
            checkSame(g, gs, std::string("big gray ") + name(im));
            checkSame(s, ss, std::string("big sobel ") + name(im));
            if (im == Impl::Avx2) {
                // Avx2 uses FMA, scalar does not: allow a +-1 difference on the rare ties, and count them
                long diff = 0;
                for (int y = 0; y < 1031; ++y)
                    for (int x = 0; x < 2051; ++x) {
                        const int d = std::abs(c.row(y)[x] - cs.row(y)[x]);
                        CHECK(d <= 1);
                        diff += d != 0;
                    }
                CHECK(diff * 1000 < 2051L * 1031);          // < 0.1 %
            } else {
                checkSame(c, cs, std::string("big conv ") + name(im));
            }
        }
}

// ============================================================================================ main
int main(int argc, char** argv) {
    std::printf("avx2+fma available: %s\n", avx2Supported() ? "yes" : "NO (AVX2 paths are skipped)");
    const char* filter = argc > 1 ? argv[1] : nullptr;
    for (const TestCase& t : registry()) {
        if (filter && !std::strstr(t.name, filter)) continue;
        g_current = t.name;
        const long before = g_failures;
        t.fn();
        std::printf("[%s] %s\n", g_failures == before ? " ok " : "FAIL", t.name);
    }
    std::printf("\n%ld checks, %ld failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
