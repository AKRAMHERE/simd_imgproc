// Replicated-border float copy of one source row: out[j] = src[clamp(j - r, 0, w-1)], j in [0, len).
void convertRowPadded(const std::uint8_t* s, int w, int r, int len, float* out) {
    int j = 0;
    const float left = static_cast<float>(s[0]);
    for (; j < r && j < len; ++j) out[j] = left;
    // middle part j in [r, r+w): vectorised u8 -> float conversion, 8 pixels at a time
    int x = 0;
    for (; x + 8 <= w && r + x + 8 <= len; x += 8) {
        const __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + x));
        _mm256_storeu_ps(out + r + x, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8)));
    }
    for (j = r + x; j < len; ++j) out[j] = static_cast<float>(s[clampInt(j - r, 0, w - 1)]);
}
