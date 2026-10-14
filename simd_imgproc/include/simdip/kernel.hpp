// kernel.hpp - a square, odd-sized 2D filter kernel with float weights.
#pragma once

#include <vector>

namespace simdip {

constexpr int kMaxKernelSize = 63;

struct Kernel {
    int size = 1;            // odd, 1..kMaxKernelSize; the radius is size/2
    std::vector<float> w;    // size*size weights, row-major: w[ky*size + kx]

    int radius() const { return size / 2; }
    float at(int ky, int kx) const { return w[static_cast<std::size_t>(ky) * size + kx]; }
    double sumAbs() const;   // L1 norm: sum of |w|
    double sum() const;      // sum of w (1.0 for a brightness-preserving filter)
    void validate() const;   // throws std::invalid_argument if the kernel cannot be used
};

namespace kernels {
Kernel identity(int size);                       // 1 in the middle, 0 elsewhere
Kernel box(int size);                            // every weight 1/size^2 (blur)
Kernel gaussian(int size, double sigma);         // sampled Gaussian, normalised to sum 1
Kernel binomial(int size);                       // rows of Pascal's triangle, exact in binary float
Kernel sharpen3();                               // [0 -1 0; -1 5 -1; 0 -1 0]
Kernel emboss3();                                // diagonal edge-ish filter with a negative side
Kernel randomKernel(int size, unsigned seed);    // random weights in [-1,1], used by tests
}  // namespace kernels

}  // namespace simdip
