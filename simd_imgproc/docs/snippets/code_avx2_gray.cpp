void rgbToGrayRows(const Image& src, Image& dst, int y0, int y1) {
    const int w = src.width();
    __m128i mask[3][3];
    for (int c = 0; c < 3; ++c)
        for (int s = 0; s < 3; ++s) mask[c][s] = _mm_load_si128(reinterpret_cast<const __m128i*>(kMasks.m[c][s]));
    const __m256i c77 = _mm256_set1_epi16(77), c150 = _mm256_set1_epi16(150), c29 = _mm256_set1_epi16(29),
                  c128 = _mm256_set1_epi16(128);

    for (int y = y0; y < y1; ++y) {
        const std::uint8_t* p = src.row(y);
        std::uint8_t* q = dst.row(y);
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            const std::uint8_t* s = p + 3 * x;      // reads s[0..47], inside the row because x+16 <= w
            const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
            const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 16));
            const __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 32));
            const __m128i r8 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(v0, mask[0][0]), _mm_shuffle_epi8(v1, mask[0][1])),
                                            _mm_shuffle_epi8(v2, mask[0][2]));
            const __m128i g8 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(v0, mask[1][0]), _mm_shuffle_epi8(v1, mask[1][1])),
                                            _mm_shuffle_epi8(v2, mask[1][2]));
            const __m128i b8 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(v0, mask[2][0]), _mm_shuffle_epi8(v1, mask[2][1])),
                                            _mm_shuffle_epi8(v2, mask[2][2]));
            // widen 16 bytes -> 16 x uint16. 77*255 + 150*255 + 29*255 + 128 = 65408 < 65536: no overflow
            // in 16 bits, so the multiply-adds can be done with plain 16-bit instructions.
            __m256i acc = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(r8), c77);
            acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(g8), c150));
            acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(b8), c29));
            acc = _mm256_add_epi16(acc, c128);
            acc = _mm256_srli_epi16(acc, 8);        // logical shift: values are 0..255 now
            const __m128i packed = _mm_packus_epi16(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q + x), packed);
        }
        for (; x < w; ++x) {                         // 0..15 leftover pixels
            const unsigned r = p[3 * x], g = p[3 * x + 1], b = p[3 * x + 2];
            q[x] = static_cast<std::uint8_t>((77u * r + 150u * g + 29u * b + 128u) >> 8);
        }
    }
}
