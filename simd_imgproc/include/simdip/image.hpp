// image.hpp - an 8-bit image with 64-byte aligned rows, and PNM (PGM/PPM) file I/O.
//
// Memory layout (row-major, interleaved channels):
//
//   pixel (x, y, channel c)  lives at   data + y * stride + x * channels + c
//
// `stride` is the distance in BYTES between the starts of two consecutive rows.
// It is >= width * channels. The extra bytes at the end of a row are "padding": they
// are never read as pixels and the kernels never write to them (a test checks this).
//
// Why a separate stride at all?
//   1. Every row starts on a 64-byte boundary (one cache line, and enough for AVX-512 too).
//   2. A stride that is an exact multiple of 4096 bytes is bad for kernels that read several
//      rows at once (see docs, chapter "Cache-aware access"): the rows land in the same L1 sets.
//      defaultStride() avoids that case. The benchmark can force any stride to measure the effect.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace simdip {

// Allocate `bytes` bytes aligned to `alignment` (power of two). Never returns nullptr (throws).
void* alignedAlloc(std::size_t bytes, std::size_t alignment = 64);
struct AlignedFree {
    void operator()(void* p) const noexcept;
};

class Image {
public:
    Image() = default;
    // channels: 1 (gray) or 3 (RGB). stride = 0 means "use defaultStride()".
    Image(int width, int height, int channels = 1, std::size_t stride = 0);
    Image(const Image& other);             // deep copy (keeps the stride of `other`)
    Image& operator=(const Image& other);
    Image(Image&&) noexcept = default;
    Image& operator=(Image&&) noexcept = default;

    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    std::size_t stride() const { return stride_; }
    std::size_t rowBytes() const { return static_cast<std::size_t>(width_) * channels_; }
    bool empty() const { return data_ == nullptr; }

    std::uint8_t* data() { return data_.get(); }
    const std::uint8_t* data() const { return data_.get(); }
    std::uint8_t* row(int y) { return data_.get() + static_cast<std::size_t>(y) * stride_; }
    const std::uint8_t* row(int y) const { return data_.get() + static_cast<std::size_t>(y) * stride_; }

    // Set every byte, including the row padding.
    void fillAll(std::uint8_t v);
    // Set only the pixel bytes of each row (not the padding).
    void fillPixels(std::uint8_t v);
    // True when the pixel bytes are equal (padding is ignored). Dimensions must match.
    bool samePixels(const Image& o) const;

    static std::size_t defaultStride(int width, int channels);

private:
    int width_ = 0, height_ = 0, channels_ = 0;
    std::size_t stride_ = 0;
    std::unique_ptr<std::uint8_t, AlignedFree> data_;
};

// ---- file I/O: binary PGM (P5, gray) and PPM (P6, RGB), maxval 255 only.
Image readPNM(const std::string& path);
void writePNM(const Image& img, const std::string& path);

// ---- deterministic test images. kind: 0 = smooth gradients + texture + noise ("photo-like"),
//      1 = uniform random noise, 2 = random 0/255 values (worst case for saturation),
//      3 = checkerboard of 0/255, 4 = constant 128.
Image makeSyntheticImage(int width, int height, int channels, std::uint32_t seed, int kind = 0);

}  // namespace simdip
