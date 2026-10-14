class Image {
public:
    Image() = default;
    // channels: 1 (gray) or 3 (RGB). stride = 0 means "use defaultStride()".
    Image(int width, int height, int channels = 1, std::size_t stride = 0);
    Image(const Image& other);             // deep copy (keeps the stride of `other`)
    Image& operator=(const Image& other);
    Image(Image&&) noexcept = default;
    Image& operator=(Image&&) noexcept = default;

    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    std::size_t stride() const { return stride_; }
    std::size_t rowBytes() const { return static_cast<std::size_t>(width_) * channels_; }
    bool empty() const { return data_ == nullptr; }

    std::uint8_t* data() { return data_.get(); }
    const std::uint8_t* data() const { return data_.get(); }
    std::uint8_t* row(int y) { return data_.get() + static_cast<std::size_t>(y) * stride_; }
    const std::uint8_t* row(int y) const { return data_.get() + static_cast<std::size_t>(y) * stride_; }

    // Set every byte, including the row padding.
    void fillAll(std::uint8_t v);
    // Set only the pixel bytes of each row (not the padding).
    void fillPixels(std::uint8_t v);
    // True when the pixel bytes are equal (padding is ignored). Dimensions must match.
    bool samePixels(const Image& o) const;

    static std::size_t defaultStride(int width, int channels);

private:
    int width_ = 0, height_ = 0, channels_ = 0;
    std::size_t stride_ = 0;
    std::unique_ptr<std::uint8_t, AlignedFree> data_;
};
