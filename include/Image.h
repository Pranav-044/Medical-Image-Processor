#pragma once

#include <memory>
#include <stdexcept>
#include <algorithm>
#include <string>

/**
 * Image<T> — A templated, cache-friendly image container.
 *
 * Design decisions:
 *  - Flat 1D array (row-major order) via unique_ptr<T[]> — not vector-of-vectors.
 *    This mirrors how real imaging libraries (ITK, OpenCV) store pixels and gives
 *    sequential memory access during convolutions, which is critical for CPU cache
 *    performance on large medical scans (e.g. 4096×4096 CT slices).
 *
 *  - std::unique_ptr for exclusive ownership with zero overhead. No raw new/delete.
 *
 *  - operator()(row, col, ch) for natural pixel access. Bounds-checked in debug builds.
 *
 *  - Rule of Five fully implemented: copy ctor, copy assignment, move ctor, move
 *    assignment, and destructor (handled by unique_ptr).
 *
 * Pixel index formula: data[ch * (height * width) + row * width + col]
 * Channels are stored in planar format (all of channel 0, then all of channel 1, ...).
 */
template <typename T>
class Image {
private:
    int width_;
    int height_;
    int channels_;
    std::unique_ptr<T[]> data_;

    // Returns the flat array index for a given (row, col, channel).
    inline int index(int row, int col, int ch) const {
        return ch * (height_ * width_) + row * width_ + col;
    }

public:
    // ─── Constructors ───────────────────────────────────────────────────────────

    /**
     * Construct a zero-initialised image of size (w × h) with `ch` channels.
     */
    Image(int w, int h, int ch = 1);

    /**
     * Deep copy constructor — allocates new memory and copies all pixel data.
     */
    Image(const Image& other);

    /**
     * Copy assignment — strong exception guarantee via copy-and-swap.
     */
    Image& operator=(Image other) noexcept;

    /**
     * Move constructor — transfers ownership of the pixel buffer in O(1).
     */
    Image(Image&& other) noexcept;

    // Destructor is implicitly correct: unique_ptr handles deallocation.
    ~Image() = default;

    // ─── Pixel Access ────────────────────────────────────────────────────────────

    /**
     * Write access: img(row, col) for grayscale, img(row, col, ch) for colour.
     */
    T& operator()(int row, int col, int ch = 0);

    /**
     * Read-only access.
     */
    const T& operator()(int row, int col, int ch = 0) const;

    // ─── Accessors ───────────────────────────────────────────────────────────────

    int getWidth()    const { return width_;    }
    int getHeight()   const { return height_;   }
    int getChannels() const { return channels_; }
    int totalPixels() const { return width_ * height_; }

    /**
     * Returns a pointer to the raw pixel buffer (for BMP I/O).
     */
    const T* data() const { return data_.get(); }
    T*       data()       { return data_.get(); }

    // ─── Operations ──────────────────────────────────────────────────────────────

    /**
     * Extract a rectangular Region of Interest (ROI).
     * Returns a new Image<T> cropped to [r, r+h) × [c, c+w).
     * Throws std::out_of_range if the ROI exceeds image bounds.
     */
    Image<T> subregion(int r, int c, int h, int w) const;

    /**
     * Fill all pixels with a constant value.
     */
    void fill(T value);

    /**
     * Returns a string representation for debugging: "Image<T>(WxH, C channels)"
     */
    std::string info() const;

    // ─── Friend swap ─────────────────────────────────────────────────────────────
    friend void swap(Image& a, Image& b) noexcept {
        using std::swap;
        swap(a.width_,    b.width_);
        swap(a.height_,   b.height_);
        swap(a.channels_, b.channels_);
        swap(a.data_,     b.data_);
    }
};

// ─── Template Implementation ───────────────────────────────────────────────────
// Kept in the header because templates must be visible at instantiation sites.

template <typename T>
Image<T>::Image(int w, int h, int ch)
    : width_(w), height_(h), channels_(ch)
{
    if (w <= 0 || h <= 0 || ch <= 0)
        throw std::invalid_argument("Image dimensions must be positive.");
    data_ = std::make_unique<T[]>(static_cast<size_t>(w) * h * ch);
    std::fill(data_.get(), data_.get() + w * h * ch, T{});
}

template <typename T>
Image<T>::Image(const Image& other)
    : width_(other.width_), height_(other.height_), channels_(other.channels_)
{
    size_t total = static_cast<size_t>(width_) * height_ * channels_;
    data_ = std::make_unique<T[]>(total);
    std::copy(other.data_.get(), other.data_.get() + total, data_.get());
}

// Copy-and-swap idiom: the parameter is taken by value (invokes copy ctor),
// then swapped with *this. This gives the strong exception guarantee.
template <typename T>
Image<T>& Image<T>::operator=(Image other) noexcept {
    swap(*this, other);
    return *this;
}

template <typename T>
Image<T>::Image(Image&& other) noexcept
    : width_(other.width_), height_(other.height_), channels_(other.channels_),
      data_(std::move(other.data_))
{
    other.width_ = other.height_ = other.channels_ = 0;
}

template <typename T>
T& Image<T>::operator()(int row, int col, int ch) {
#ifdef MEDIMG_DEBUG
    if (row < 0 || row >= height_ || col < 0 || col >= width_ || ch < 0 || ch >= channels_)
        throw std::out_of_range("Image pixel access out of bounds.");
#endif
    return data_[index(row, col, ch)];
}

template <typename T>
const T& Image<T>::operator()(int row, int col, int ch) const {
#ifdef MEDIMG_DEBUG
    if (row < 0 || row >= height_ || col < 0 || col >= width_ || ch < 0 || ch >= channels_)
        throw std::out_of_range("Image pixel access out of bounds.");
#endif
    return data_[index(row, col, ch)];
}

template <typename T>
Image<T> Image<T>::subregion(int r, int c, int h, int w) const {
    if (r < 0 || c < 0 || r + h > height_ || c + w > width_)
        throw std::out_of_range("Subregion exceeds image bounds.");

    Image<T> roi(w, h, channels_);
    for (int ch = 0; ch < channels_; ++ch)
        for (int row = 0; row < h; ++row)
            for (int col = 0; col < w; ++col)
                roi(row, col, ch) = (*this)(r + row, c + col, ch);
    return roi;
}

template <typename T>
void Image<T>::fill(T value) {
    size_t total = static_cast<size_t>(width_) * height_ * channels_;
    std::fill(data_.get(), data_.get() + total, value);
}

template <typename T>
std::string Image<T>::info() const {
    return "Image(" + std::to_string(width_) + "x" + std::to_string(height_)
         + ", " + std::to_string(channels_) + " ch)";
}
