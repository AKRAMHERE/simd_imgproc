void rgbToGrayRows(const Image& src, Image& dst, int y0, int y1) {
    const int w = src.width();
    for (int y = y0; y < y1; ++y) {
        const std::uint8_t* __restrict p = src.row(y);
        std::uint8_t* __restrict q = dst.row(y);
        for (int x = 0; x < w; ++x) {
            const unsigned r = p[3 * x], g = p[3 * x + 1], b = p[3 * x + 2];
            q[x] = static_cast<std::uint8_t>((77u * r + 150u * g + 29u * b + 128u) >> 8);
        }
    }
}

void sobelRows(const Image& src, Image& dst, int y0, int y1) {
    const int w = src.width(), h = src.height();
    for (int y = y0; y < y1; ++y) {
        const std::uint8_t* __restrict a = src.row(y > 0 ? y - 1 : 0);
        const std::uint8_t* __restrict b = src.row(y);
        const std::uint8_t* __restrict c = src.row(y < h - 1 ? y + 1 : h - 1);
        std::uint8_t* __restrict q = dst.row(y);
        if (w == 1) {
            q[0] = sobelPixel(a, b, c, 0, 0, 0);
            continue;
        }
        q[0] = sobelPixel(a, b, c, 0, 0, 1);
        for (int x = 1; x < w - 1; ++x) q[x] = sobelPixel(a, b, c, x - 1, x, x + 1);
        q[w - 1] = sobelPixel(a, b, c, w - 2, w - 1, w - 1);
    }
}
