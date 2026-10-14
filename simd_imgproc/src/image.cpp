#include "simdip/image.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <stdexcept>

namespace simdip {

void* alignedAlloc(std::size_t bytes, std::size_t alignment) {
    // posix_memalign takes the EXACT size (no rounding up to the alignment), so a sanitizer sees the true
    // allocation end and flags a vector load that reads even one byte past the last row.
    void* p = nullptr;
    if (posix_memalign(&p, alignment, bytes == 0 ? 1 : bytes) != 0 || !p) throw std::bad_alloc();
    return p;
}

void AlignedFree::operator()(void* p) const noexcept { std::free(p); }

std::size_t Image::defaultStride(int width, int channels) {
    std::size_t s = (static_cast<std::size_t>(width) * channels + 63) / 64 * 64;
    if (s % 4096 == 0) s += 64;   // avoid exact multiples of the page size (L1 set aliasing)
    return s;
}

Image::Image(int width, int height, int channels, std::size_t stride)
    : width_(width), height_(height), channels_(channels) {
    if (width < 1 || height < 1) throw std::invalid_argument("Image: width and height must be >= 1");
    if (channels != 1 && channels != 3) throw std::invalid_argument("Image: channels must be 1 or 3");
    const std::size_t rb = rowBytes();
    stride_ = stride == 0 ? defaultStride(width, channels) : stride;
    if (stride_ < rb) throw std::invalid_argument("Image: stride smaller than a row");
    data_.reset(static_cast<std::uint8_t*>(alignedAlloc(stride_ * static_cast<std::size_t>(height))));
    std::memset(data_.get(), 0, stride_ * static_cast<std::size_t>(height));   // touch the pages now
}

Image::Image(const Image& o)
    : width_(o.width_), height_(o.height_), channels_(o.channels_), stride_(o.stride_) {
    if (o.data_) {
        data_.reset(static_cast<std::uint8_t*>(alignedAlloc(stride_ * static_cast<std::size_t>(height_))));
        std::memcpy(data_.get(), o.data_.get(), stride_ * static_cast<std::size_t>(height_));
    }
}

Image& Image::operator=(const Image& o) {
    if (this != &o) {
        Image tmp(o);
        *this = std::move(tmp);
    }
    return *this;
}

void Image::fillAll(std::uint8_t v) {
    if (data_) std::memset(data_.get(), v, stride_ * static_cast<std::size_t>(height_));
}

void Image::fillPixels(std::uint8_t v) {
    for (int y = 0; y < height_; ++y) std::memset(row(y), v, rowBytes());
}

bool Image::samePixels(const Image& o) const {
    if (width_ != o.width_ || height_ != o.height_ || channels_ != o.channels_) return false;
    for (int y = 0; y < height_; ++y)
        if (std::memcmp(row(y), o.row(y), rowBytes()) != 0) return false;
    return true;
}

// ---------------------------------------------------------------------------- PNM I/O
namespace {
// Reads one header token, skipping whitespace and '#' comments.
std::string nextToken(std::istream& in) {
    std::string tok;
    int c;
    while ((c = in.get()) != EOF) {
        if (c == '#') {
            while ((c = in.get()) != EOF && c != '\n') {}
        } else if (!std::isspace(c)) {
            tok.push_back(static_cast<char>(c));
            break;
        }
    }
    while ((c = in.peek()) != EOF && !std::isspace(c)) tok.push_back(static_cast<char>(in.get()));
    return tok;
}
}  // namespace

Image readPNM(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    const std::string magic = nextToken(in);
    int channels = 0;
    if (magic == "P5") channels = 1;
    else if (magic == "P6") channels = 3;
    else throw std::runtime_error(path + ": not a binary PGM (P5) or PPM (P6) file");
    long w = std::atol(nextToken(in).c_str());
    long h = std::atol(nextToken(in).c_str());
    long maxv = std::atol(nextToken(in).c_str());
    if (w < 1 || h < 1 || w > 65536 || h > 65536) throw std::runtime_error(path + ": bad dimensions");
    if (maxv != 255) throw std::runtime_error(path + ": only maxval 255 is supported");
    in.get();   // the single whitespace after the header
    Image img(static_cast<int>(w), static_cast<int>(h), channels);
    for (int y = 0; y < img.height(); ++y) {
        in.read(reinterpret_cast<char*>(img.row(y)), static_cast<std::streamsize>(img.rowBytes()));
        if (!in) throw std::runtime_error(path + ": truncated pixel data");
    }
    return img;
}

void writePNM(const Image& img, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << (img.channels() == 1 ? "P5" : "P6") << "\n" << img.width() << " " << img.height() << "\n255\n";
    for (int y = 0; y < img.height(); ++y)
        out.write(reinterpret_cast<const char*>(img.row(y)), static_cast<std::streamsize>(img.rowBytes()));
    if (!out) throw std::runtime_error("write failed: " + path);
}

// ---------------------------------------------------------------------------- synthetic images
namespace {
struct XorShift {
    std::uint32_t s;
    explicit XorShift(std::uint32_t seed) : s(seed ? seed : 0x9E3779B9u) {}
    std::uint32_t next() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }
};
}  // namespace

Image makeSyntheticImage(int width, int height, int channels, std::uint32_t seed, int kind) {
    Image img(width, height, channels);
    XorShift rng(seed * 2654435761u + 12345u);
    const int rb = static_cast<int>(img.rowBytes());
    for (int y = 0; y < height; ++y) {
        std::uint8_t* p = img.row(y);
        for (int i = 0; i < rb; ++i) {
            const int x = i / channels, c = i % channels;
            int v = 128;
            switch (kind) {
                case 0: {
                    // smooth gradient + a few stripes + mild noise, different per channel
                    const int grad = (x * 160 / (width > 1 ? width - 1 : 1)) + (y * 60 / (height > 1 ? height - 1 : 1));
                    const int stripe = (((x / 7) + (y / 5)) & 1) ? 20 : -20;
                    const int noise = static_cast<int>(rng.next() % 17) - 8;
                    v = grad + stripe * (c + 1) / 2 + noise + 20 * c;
                    break;
                }
                case 1: v = static_cast<int>(rng.next() & 255); break;
                case 2: v = (rng.next() & 1) ? 255 : 0; break;
                case 3: v = (((x + y) & 1) ? 255 : 0); break;
                default: v = 128; break;
            }
            p[i] = static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    return img;
}

}  // namespace simdip
