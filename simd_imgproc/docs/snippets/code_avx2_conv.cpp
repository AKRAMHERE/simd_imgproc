// NV = number of 8-float vectors per block (block = 8*NV output pixels). 8 independent accumulators hide the
// 4-cycle FMA latency on CPUs that can start two FMAs per cycle; 4 is the ablation.
template <int NV>
void convolveRowsImpl(const Image& src, Image& dst, const Kernel& K, int y0, int y1) {
    constexpr int kBlock = 8 * NV;
    const int w = src.width(), h = src.height(), k = K.size, r = k / 2;
    const float* wt = K.w.data();
    const int wRound = (w + kBlock - 1) / kBlock * kBlock;
    const int RL = wRound + 2 * r;               // length of one padded float row

    // k slots of padded float rows; slotRow[s] = which source row currently lives in slot s
    std::unique_ptr<float, FreeDeleter> ring(static_cast<float*>(alignedAlloc(sizeof(float) * static_cast<std::size_t>(RL) * k, 64)));
    int slotRow[kMaxKernelSize];
    for (int s = 0; s < k; ++s) slotRow[s] = -1;

    for (int y = y0; y < y1; ++y) {
        int ids[kMaxKernelSize];
        const float* rowp[kMaxKernelSize];
        for (int j = 0; j < k; ++j) ids[j] = clampInt(y - r + j, 0, h - 1);
        for (int j = 0; j < k; ++j) {
            int slot = -1;
            for (int s = 0; s < k; ++s)
                if (slotRow[s] == ids[j]) { slot = s; break; }
            if (slot < 0) {                       // not cached: reuse a slot that no needed row lives in
                for (int s = 0; s < k && slot < 0; ++s) {
                    bool needed = false;
                    for (int t = 0; t < k; ++t) needed = needed || (slotRow[s] == ids[t]);
                    if (!needed) slot = s;
                }
                convertRowPadded(src.row(ids[j]), w, r, RL, ring.get() + static_cast<std::size_t>(slot) * RL);
                slotRow[slot] = ids[j];
            }
            rowp[j] = ring.get() + static_cast<std::size_t>(slot) * RL;
        }

        std::uint8_t* q = dst.row(y);
        for (int x = 0; x < w; x += kBlock) {
            __m256 acc[NV];
            for (int v = 0; v < NV; ++v) acc[v] = _mm256_setzero_ps();
            for (int j = 0; j < k; ++j) {
                const float* p = rowp[j] + x;
                const float* wk = wt + j * k;
                for (int i = 0; i < k; ++i) {
                    const __m256 wv = _mm256_broadcast_ss(wk + i);
                    for (int v = 0; v < NV; ++v) acc[v] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p + i + 8 * v), acc[v]);
                }
            }
            // round to nearest even (MXCSR default), pack with saturation, store
            __m256i out[NV / 4];
            for (int g = 0; g < NV / 4; ++g)
                out[g] = packInt32To32Bytes(_mm256_cvtps_epi32(acc[4 * g]), _mm256_cvtps_epi32(acc[4 * g + 1]),
                                            _mm256_cvtps_epi32(acc[4 * g + 2]), _mm256_cvtps_epi32(acc[4 * g + 3]));
            if (x + kBlock <= w) {
                for (int g = 0; g < NV / 4; ++g) _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + x + 32 * g), out[g]);
            } else {                              // last, partial block: never write past the row
                alignas(32) std::uint8_t tmp[kBlock];
                for (int g = 0; g < NV / 4; ++g) _mm256_store_si256(reinterpret_cast<__m256i*>(tmp + 32 * g), out[g]);
                std::memcpy(q + x, tmp, static_cast<std::size_t>(w - x));
            }
        }
    }
}
