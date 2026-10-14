// simdip_bench - benchmark harness.
//
//   simdip_bench --op gray,sobel,conv,copy --impl scalar,autovec,avx2 --sizes 512,1024,2048,4096
//                --ksizes 3,5,7,9,11 --threads 1,2,4 --reps 9 --min-ms 20 --csv out.csv --label myrun
//                [--stride BYTES] [--aspect N] [--ablation]      (aspect N: height = width / N)
//
// Method (the same for every configuration):
//   * images are allocated and touched BEFORE timing (no page-fault cost inside the measurement),
//   * 2 warm-up calls, then one call to calibrate how many iterations make a sample last >= --min-ms,
//   * `reps` samples, each = average time per call over that many iterations; we report the MEDIAN
//     (robust against a noisy neighbour on a shared machine) plus min and max,
//   * throughput = output pixels / median time; "GB/s" counts the minimum traffic (every source byte read
//     once, every destination byte written once), so it is a lower bound of what the memory system moved.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "simdip/ops.hpp"

using namespace simdip;
using Clock = std::chrono::steady_clock;

static std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(tok);
    return out;
}
static std::vector<int> splitInts(const std::string& s) {
    std::vector<int> v;
    for (const auto& t : splitCsv(s)) v.push_back(std::atoi(t.c_str()));
    return v;
}

static std::string cpuModel() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
    return "unknown";
}

struct Options {
    std::vector<std::string> ops = {"gray", "sobel", "conv"};
    std::vector<std::string> impls = {"scalar", "autovec", "avx2"};
    std::vector<int> sizes = {512, 1024, 2048};
    std::vector<int> ksizes = {3, 5, 7};
    std::vector<int> threads = {1};
    int reps = 9;
    double minMs = 20.0;
    std::size_t stride = 0;
    bool ablation = false;
    std::string csv, label = "run";
    int aspect = 1;   // height = width / aspect ... (square by default)
};

struct Stats { double median, min, max; int iters; };

static Stats measure(const std::function<void()>& f, int reps, double minMs) {
    f();
    f();   // warm-up: caches, branch predictors, OpenMP thread pool
    const auto t0 = Clock::now();
    f();
    const double one = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    const int iters = std::max(1, std::min(2000, static_cast<int>(std::ceil(minMs / std::max(one, 1e-6)))));
    std::vector<double> s;
    for (int r = 0; r < reps; ++r) {
        const auto a = Clock::now();
        for (int i = 0; i < iters; ++i) f();
        s.push_back(std::chrono::duration<double, std::milli>(Clock::now() - a).count() / iters);
    }
    std::sort(s.begin(), s.end());
    return {s[s.size() / 2], s.front(), s.back(), iters};
}

int main(int argc, char** argv) {
    try {
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
                return argv[++i];
            };
            if (a == "--op") o.ops = splitCsv(next());
            else if (a == "--impl") o.impls = splitCsv(next());
            else if (a == "--sizes") o.sizes = splitInts(next());
            else if (a == "--ksizes") o.ksizes = splitInts(next());
            else if (a == "--threads") o.threads = splitInts(next());
            else if (a == "--reps") o.reps = std::atoi(next().c_str());
            else if (a == "--min-ms") o.minMs = std::atof(next().c_str());
            else if (a == "--stride") o.stride = static_cast<std::size_t>(std::atol(next().c_str()));
            else if (a == "--ablation") o.ablation = true;
            else if (a == "--aspect") o.aspect = std::atoi(next().c_str());
            else if (a == "--csv") o.csv = next();
            else if (a == "--label") o.label = next();
            else throw std::invalid_argument("unknown option " + a);
        }
#ifdef _OPENMP
        omp_set_dynamic(0);
#endif
        std::printf("# cpu: %s\n# avx2+fma: %s   OpenMP max threads: %d\n# compiler: %s  optimized build: %s\n",
                    cpuModel().c_str(), avx2Supported() ? "yes" : "no", maxThreads(), __VERSION__,
#ifdef __OPTIMIZE__
                    "yes"
#else
                    "NO (timings are meaningless)"
#endif
        );
        const char* bind = std::getenv("OMP_PROC_BIND");
        const char* places = std::getenv("OMP_PLACES");
        std::printf("# OMP_PROC_BIND=%s OMP_PLACES=%s\n", bind ? bind : "(unset)", places ? places : "(unset)");

        std::FILE* csv = nullptr;
        if (!o.csv.empty()) {
            csv = std::fopen(o.csv.c_str(), "w");
            if (!csv) throw std::runtime_error("cannot write " + o.csv);
            std::fprintf(csv, "label,op,impl,threads,width,height,stride,ksize,iters,reps,median_ms,min_ms,max_ms,mpix_s,gb_s,gflops\n");
        }
        std::printf("%-6s %-14s %3s %6s %6s %9s %9s %9s %9s %8s\n", "op", "impl", "thr", "size", "ksize", "median_ms", "min_ms",
                    "Mpix/s", "GB/s", "GFLOP/s");

        for (int size : o.sizes) {
            const int w = size, h = size / o.aspect;
            const std::size_t gstride = (o.stride >= static_cast<std::size_t>(w)) ? o.stride : 0;
            Image rgb = makeSyntheticImage(w, h, 3, 1, 0);
            Image gray = makeSyntheticImage(w, h, 1, 2, 0);
            Image grayS(w, h, 1, gstride);                    // gray source with the requested stride
            for (int y = 0; y < h; ++y) std::memcpy(grayS.row(y), gray.row(y), grayS.rowBytes());
            Image dst(w, h, 1, gstride);
            for (int threads : o.threads) {
                for (const std::string& op : o.ops) {
                    // list of (label, function) variants to time for this op
                    struct Variant { std::string name; std::function<void()> fn; int ksize; };
                    std::vector<Variant> vars;
                    auto addImpls = [&](int ks, const std::function<void(Exec)>& call) {
                        for (const std::string& in : o.impls) {
                            const Impl im = parseImpl(in);
                            if (im == Impl::Avx2 && !avx2Supported()) continue;
                            vars.push_back({in, [=] { call({im, threads}); }, ks});
                        }
                    };
                    if (op == "gray") addImpls(0, [&](Exec e) { rgbToGray(rgb, dst, e); });
                    else if (op == "sobel") addImpls(0, [&](Exec e) { sobel(grayS, dst, e); });
                    else if (op == "copy") vars.push_back({"memcpy", [&] { ablation::copyRows(grayS, dst, threads); }, 0});
                    else if (op == "conv") {
                        for (int ks : o.ksizes) {
                            const Kernel kern = kernels::gaussian(ks, std::max(0.8, ks / 3.0));
                            addImpls(ks, [&grayS, &dst, kern](Exec e) { convolve(grayS, dst, kern, e); });
                            if (o.ablation) {
                                if (avx2Supported()) {
                                    vars.push_back({"avx2_acc4", [&grayS, &dst, kern, threads] { ablation::convolveAvx2Acc4(grayS, dst, kern, threads); }, ks});
                                    vars.push_back({"avx2_pertap", [&grayS, &dst, kern, threads] { ablation::convolveAvx2PerTapConvert(grayS, dst, kern, threads); }, ks});
                                }
                                vars.push_back({"autovec_rowacc", [&grayS, &dst, kern, threads] { ablation::convolveAutoVecRowAcc(grayS, dst, kern, threads); }, ks});
                            }
                        }
                    } else throw std::invalid_argument("unknown op " + op);

                    for (const Variant& v : vars) {
                        const Stats s = measure(v.fn, o.reps, o.minMs);
                        const double px = static_cast<double>(w) * h;
                        const double mpix = px / (s.median * 1e3);
                        const double bytesPerPx = op == "gray" ? 4.0 : 2.0;
                        const double gbs = px * bytesPerPx / (s.median * 1e6);
                        const double gflops = op == "conv" ? px * 2.0 * v.ksize * v.ksize / (s.median * 1e6) : 0.0;
                        std::printf("%-6s %-14s %3d %6d %6d %9.3f %9.3f %9.1f %9.2f %8.2f\n", op.c_str(), v.name.c_str(), threads, size,
                                    v.ksize, s.median, s.min, mpix, gbs, gflops);
                        if (csv)
                            std::fprintf(csv, "%s,%s,%s,%d,%d,%d,%zu,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.4f,%.4f\n", o.label.c_str(), op.c_str(),
                                         v.name.c_str(), threads, w, h, dst.stride(), v.ksize, s.iters, o.reps, s.median, s.min, s.max,
                                         mpix, gbs, gflops);
                    }
                }
            }
        }
        if (csv) std::fclose(csv);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
