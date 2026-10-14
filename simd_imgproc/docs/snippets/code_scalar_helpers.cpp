inline int clampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline std::uint8_t sobelPixel(const std::uint8_t* a, const std::uint8_t* b, const std::uint8_t* c,
                               int xm, int x, int xp) {
    // a, b, c = rows y-1, y, y+1 ; xm, x, xp = columns x-1, x, x+1 (already clamped by the caller)
    const int gx = (a[xp] - a[xm]) + 2 * (b[xp] - b[xm]) + (c[xp] - c[xm]);
    const int gy = (c[xm] + 2 * c[x] + c[xp]) - (a[xm] + 2 * a[x] + a[xp]);
    const int m = std::abs(gx) + std::abs(gy);
    return static_cast<std::uint8_t>(m > 255 ? 255 : m);
}

inline std::uint8_t roundClampF(float v) {
    const long r = std::lrintf(v);   // round to nearest, ties to even (default rounding mode)
    return static_cast<std::uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
}
