// Turns 4 vectors of 8 int32 (pixels 0-7, 8-15, 16-23, 24-31) into 32 saturated bytes in pixel order.
inline __m256i packInt32To32Bytes(__m256i c0, __m256i c1, __m256i c2, __m256i c3) {
    const __m256i p01 = _mm256_packs_epi32(c0, c1);    // int32 -> int16 (saturating), per 128-bit lane
    const __m256i p23 = _mm256_packs_epi32(c2, c3);
    const __m256i p = _mm256_packus_epi16(p01, p23);   // int16 -> uint8 (saturating) => clamp to 0..255
    // dwords now are [px0-3, px8-11, px16-19, px24-27 | px4-7, px12-15, px20-23, px28-31]
    return _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}
