// internal.hpp - row-range kernels. Every implementation exposes the same three functions that process
// the output rows [y0, y1) only; dispatch.cpp splits the image into bands and calls them (one band per
// OpenMP thread). The functions assume the arguments were validated by the public API.
#pragma once

#include "simdip/image.hpp"
#include "simdip/kernel.hpp"

namespace simdip {

#define SIMDIP_DECLARE_ROW_KERNELS                                                     \
    void rgbToGrayRows(const Image& src, Image& dst, int y0, int y1);                  \
    void sobelRows(const Image& src, Image& dst, int y0, int y1);                      \
    void convolveRows(const Image& src, Image& dst, const Kernel& k, int y0, int y1);

namespace scalar {
SIMDIP_DECLARE_ROW_KERNELS
}
namespace autovec {
SIMDIP_DECLARE_ROW_KERNELS
void convolveRowsRowAcc(const Image& src, Image& dst, const Kernel& k, int y0, int y1);
}
namespace avx2 {
SIMDIP_DECLARE_ROW_KERNELS
void convolveRowsAcc4(const Image& src, Image& dst, const Kernel& k, int y0, int y1);
void convolveRowsPerTap(const Image& src, Image& dst, const Kernel& k, int y0, int y1);
}

}  // namespace simdip
