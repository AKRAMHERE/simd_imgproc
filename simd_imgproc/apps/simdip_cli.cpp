// simdip - command line front end.
//   simdip info
//   simdip gray  in.ppm out.pgm [--impl scalar|autovec|avx2] [--threads N]
//   simdip sobel in.pgm out.pgm [--impl ...] [--threads N]
//   simdip conv  in.pgm out.pgm --kernel box|gauss|binomial|sharpen|emboss|identity [--size K] [--sigma S] [--impl ...] [--threads N]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "simdip/ops.hpp"

using namespace simdip;

static void usage() {
    std::puts(
        "usage:\n"
        "  simdip info\n"
        "  simdip gray  in.ppm out.pgm [--impl scalar|autovec|avx2] [--threads N]\n"
        "  simdip sobel in.pgm out.pgm [--impl ...] [--threads N]\n"
        "  simdip conv  in.pgm out.pgm --kernel box|gauss|binomial|sharpen|emboss|identity [--size K] [--sigma S]\n"
        "                               [--impl ...] [--threads N]");
}

static std::string cpuModel() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
    return "unknown";
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 2; }
        const std::string cmd = argv[1];
        if (cmd == "info") {
            std::printf("cpu:        %s\n", cpuModel().c_str());
            std::printf("avx2+fma:   %s\n", avx2Supported() ? "yes" : "no");
            std::printf("max threads (OpenMP): %d\n", maxThreads());
            return 0;
        }
        if (argc < 4) { usage(); return 2; }
        const std::string in = argv[2], out = argv[3];
        Exec ex;
        ex.impl = avx2Supported() ? Impl::Avx2 : Impl::AutoVec;
        std::string kernel = "box";
        int size = 3;
        double sigma = 1.0;
        for (int i = 4; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
                return argv[++i];
            };
            if (a == "--impl") ex.impl = parseImpl(next());
            else if (a == "--threads") ex.threads = std::atoi(next().c_str());
            else if (a == "--kernel") kernel = next();
            else if (a == "--size") size = std::atoi(next().c_str());
            else if (a == "--sigma") sigma = std::atof(next().c_str());
            else throw std::invalid_argument("unknown option " + a);
        }
        const Image src = readPNM(in);
        Image dst(src.width(), src.height(), 1);
        if (cmd == "gray") {
            rgbToGray(src, dst, ex);
        } else if (cmd == "sobel") {
            sobel(src, dst, ex);
        } else if (cmd == "conv") {
            Kernel k;
            if (kernel == "box") k = kernels::box(size);
            else if (kernel == "gauss") k = kernels::gaussian(size, sigma);
            else if (kernel == "binomial") k = kernels::binomial(size);
            else if (kernel == "sharpen") k = kernels::sharpen3();
            else if (kernel == "emboss") k = kernels::emboss3();
            else if (kernel == "identity") k = kernels::identity(size);
            else throw std::invalid_argument("unknown kernel " + kernel);
            convolve(src, dst, k, ex);
        } else {
            usage();
            return 2;
        }
        writePNM(dst, out);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
