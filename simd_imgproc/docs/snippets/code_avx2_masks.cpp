struct DeinterleaveMasks {
    // m[channel][source register][destination byte] = index into that source register, or 0x80 = "write 0"
    std::uint8_t m[3][3][16];
};

constexpr DeinterleaveMasks makeMasks() {
    DeinterleaveMasks d{};
    for (int c = 0; c < 3; ++c)
        for (int s = 0; s < 3; ++s)
            for (int i = 0; i < 16; ++i) d.m[c][s][i] = 0x80;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 16; ++i) {
            const int g = 3 * i + c;              // byte position of channel c of pixel i in the 48 bytes
            d.m[c][g / 16][i] = static_cast<std::uint8_t>(g % 16);
        }
    return d;
}
alignas(16) constexpr DeinterleaveMasks kMasks = makeMasks();
