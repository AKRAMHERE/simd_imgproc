void convolveRows(const Image& src, Image& dst, const Kernel& K, int y0, int y1) {
    const int w = src.width(), h = src.height(), k = K.size, r = k / 2;
    const float* __restrict wt = K.w.data();
    const std::uint8_t* rows[kMaxKernelSize];
    for (int y = y0; y < y1; ++y) {
        for (int j = 0; j < k; ++j) rows[j] = src.row(clampInt(y - r + j, 0, h - 1));
        std::uint8_t* __restrict q = dst.row(y);
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            if (x >= r && x < w - r) {                       // interior: no clamping needed
                for (int j = 0; j < k; ++j) {
                    const std::uint8_t* s = rows[j] + (x - r);
                    for (int i = 0; i < k; ++i) acc += wt[j * k + i] * static_cast<float>(s[i]);
                }
            } else {                                         // left/right border: clamp the column
                for (int j = 0; j < k; ++j)
                    for (int i = 0; i < k; ++i)
                        acc += wt[j * k + i] * static_cast<float>(rows[j][clampInt(x - r + i, 0, w - 1)]);
            }
            q[x] = roundClampF(acc);
        }
    }
}
