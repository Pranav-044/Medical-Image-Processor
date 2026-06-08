#pragma once

#include <cstdint>
#include <algorithm>
#include <cmath>

/**
 * Utils.h — Low-level helper functions used throughout the pipeline.
 *
 * Kept as free functions in a namespace (not a class) because these are
 * pure utility operations with no state — a class would add no value.
 */
namespace medimg {

// ─── Clamp ────────────────────────────────────────────────────────────────────

/**
 * Clamp a value to [lo, hi]. Used after convolution to keep pixel values
 * within uint8_t range [0, 255] without undefined behaviour from overflow.
 *
 * std::clamp is C++17 — we use it directly here.
 */
template <typename T>
inline T clamp(T value, T lo, T hi) {
    return std::clamp(value, lo, hi);
}

/**
 * Saturate-cast a float to uint8_t. Rounds and clamps in one step.
 * This is the operation applied after every convolution kernel sum.
 *
 * Example: saturate_cast(255.7f) → 255, saturate_cast(-3.2f) → 0
 */
inline uint8_t saturate_cast(float value) {
    int rounded = static_cast<int>(std::round(value));
    return static_cast<uint8_t>(std::clamp(rounded, 0, 255));
}

// ─── Padding ──────────────────────────────────────────────────────────────────

/**
 * Zero-padding boundary check.
 *
 * When applying a convolution kernel near image borders, some kernel positions
 * map to coordinates outside the image. This function returns 0 for such
 * out-of-bounds accesses (zero-padding strategy).
 *
 * Strategy comparison:
 *   - Zero-padding (used here): simple, fast, may darken edges slightly.
 *   - Reflection padding: mirrors border pixels — better for Gaussian blur.
 *   - Clamped/replicate: repeats edge pixels — good for gradients.
 *
 * Zero-padding is standard for medical imaging edge detection (Sobel) because
 * anatomical ROIs are typically centred — border artefacts are acceptable.
 *
 * @param row    Target row index (may be negative or >= height)
 * @param col    Target column index (may be negative or >= width)
 * @param height Image height
 * @param width  Image width
 * @return       true if the coordinate is within bounds
 */
inline bool inBounds(int row, int col, int height, int width) {
    return row >= 0 && row < height && col >= 0 && col < width;
}

// ─── Luminance ────────────────────────────────────────────────────────────────

/**
 * ITU-R BT.601 luminance formula.
 *
 * Converts an RGB triple to a perceptually accurate grayscale intensity.
 * The weights (0.299, 0.587, 0.114) account for the human eye's greater
 * sensitivity to green light. Used by JPEG, MPEG, and broadcast standards.
 *
 * This is NOT a simple average — a naive (R+G+B)/3 would produce
 * visually incorrect results for medical imaging where contrast matters.
 */
inline uint8_t luminance(uint8_t r, uint8_t g, uint8_t b) {
    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    return saturate_cast(y);
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/**
 * Compute gradient magnitude from Gx and Gy components.
 * Used in Sobel edge detection.
 * magnitude = clamp(sqrt(Gx² + Gy²), 0, 255)
 */
inline uint8_t gradientMagnitude(float gx, float gy) {
    float mag = std::sqrt(gx * gx + gy * gy);
    return saturate_cast(mag);
}

/**
 * Gaussian weight for a kernel position (x, y) with standard deviation sigma.
 * G(x,y) = exp(-(x²+y²) / (2*sigma²))
 * The 1/(2*pi*sigma²) normalisation factor is omitted — applied via kernel
 * normalisation in GaussianBlur constructor.
 */
inline float gaussianWeight(int x, int y, float sigma) {
    float sigma2 = sigma * sigma;
    return std::exp(-(x * x + y * y) / (2.0f * sigma2));
}

} // namespace medimg
