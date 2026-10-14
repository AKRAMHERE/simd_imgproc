void sobelRows(const Image& src, Image& dst, int y0, int y1) {
    const int w = src.width(), h = src.height();
    for (int y = y0; y < y1; ++y) {
        const std::uint8_t* a = src.row(y > 0 ? y - 1 : 0);
        const std::uint8_t* b = src.row(y);
        const std::uint8_t* c = src.row(y < h - 1 ? y + 1 : h - 1);
        std::uint8_t* q = dst.row(y);
        if (w == 1) {
            q[0] = sobelPixel(a, b, c, 0, 0, 0);
            continue;
        }
        q[0] = sobelPixel(a, b, c, 0, 0, 1);
        int x = 1;
        // 16 output pixels per iteration. The loads at x-1, x, x+1 read bytes up to x+16, so we need
        // x + 16 <= w - 1.  Everything is 16-bit: |Gx|,|Gy| <= 1020, |Gx|+|Gy| <= 2040 < 32768.
        for (; x + 16 <= w - 1; x += 16) {
            const __m256i a0 = load16(a + x - 1), a1 = load16(a + x), a2 = load16(a + x + 1);
            const __m256i b0 = load16(b + x - 1), b2 = load16(b + x + 1);
            const __m256i c0 = load16(c + x - 1), c1 = load16(c + x), c2 = load16(c + x + 1);
            const __m256i dxb = _mm256_sub_epi16(b2, b0);
            const __m256i gx = _mm256_add_epi16(_mm256_add_epi16(_mm256_sub_epi16(a2, a0), _mm256_sub_epi16(c2, c0)),
                                                _mm256_add_epi16(dxb, dxb));
            const __m256i top = _mm256_add_epi16(_mm256_add_epi16(a0, a2), _mm256_add_epi16(a1, a1));
            const __m256i bot = _mm256_add_epi16(_mm256_add_epi16(c0, c2), _mm256_add_epi16(c1, c1));
            const __m256i gy = _mm256_sub_epi16(bot, top);
            const __m256i mag = _mm256_add_epi16(_mm256_abs_epi16(gx), _mm256_abs_epi16(gy));
            // packus saturates int16 -> uint8, i.e. min(mag, 255) since mag >= 0.
            // It works inside each 128-bit lane, so the 16 results end up as qwords [0, 0', 1, 1'];
            // permute4x64 with 0x08 moves qwords 0 and 2 to the bottom.
            const __m256i pk = _mm256_permute4x64_epi64(_mm256_packus_epi16(mag, mag), 0x08);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q + x), _mm256_castsi256_si128(pk));
        }
        for (; x < w - 1; ++x) q[x] = sobelPixel(a, b, c, x - 1, x, x + 1);
        q[w - 1] = sobelPixel(a, b, c, w - 2, w - 1, w - 1);
    }
}
