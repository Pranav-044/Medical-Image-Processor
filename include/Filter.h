#pragma once

#include "Image.h"
#include <string>
#include <vector>
#include <memory>

/**
 * Filter.h — Abstract filter base class and concrete filter hierarchy.
 *
 * Design Pattern: Template Method / Strategy
 *   - Filter is the abstract interface (Strategy role).
 *   - ConvolutionFilter implements the common kernel-application loop
 *     (Template Method), with subclasses only defining the kernel.
 *   - ProcessingPipeline owns a vector of Filters (Chain of Responsibility).
 *
 * Why abstract base class instead of std::function?
 *   - Filters need state (kernel weights, sigma, kernel size).
 *   - Virtual dispatch allows runtime polymorphism: the pipeline doesn't need
 *     to know which filter it's running — it just calls apply().
 *   - This is how ITK (Insight Toolkit) and VTK structure their filter graphs.
 */

// ─── Abstract Base ─────────────────────────────────────────────────────────────

class Filter {
public:
    virtual ~Filter() = default;

    /**
     * Apply this filter to the input image and return the result.
     * Input is not modified — all filters produce new Image instances.
     * This immutability makes pipelines composable and thread-safe.
     */
    virtual Image<uint8_t> apply(const Image<uint8_t>& input) const = 0;

    /**
     * Human-readable filter name, used by ProcessingPipeline::printStages()
     * and the --benchmark CLI flag.
     */
    virtual std::string name() const = 0;
};

// ─── Convolution Filter ────────────────────────────────────────────────────────

/**
 * ConvolutionFilter — Base class for all kernel-based spatial filters.
 *
 * Implements the shared pixel-neighbourhood loop. Subclasses provide the kernel.
 * Border handling uses zero-padding (see Utils.h for rationale).
 *
 * Complexity: O(W × H × K²) where K is kernel size.
 * For a 3×3 kernel on a 512×512 image: ~2.36M multiply-accumulates.
 */
class ConvolutionFilter : public Filter {
protected:
    std::vector<std::vector<float>> kernel_;
    int kernelSize_;

    /**
     * Apply the stored kernel at pixel (row, col) of the given image channel.
     * Returns the weighted sum as a float (caller handles clamping).
     * Out-of-bounds neighbours are treated as 0 (zero-padding).
     */
    float applyKernelAt(const Image<uint8_t>& img,
                        int row, int col, int ch = 0) const;

public:
    explicit ConvolutionFilter(std::vector<std::vector<float>> kernel);
};

// ─── Gaussian Blur ────────────────────────────────────────────────────────────

/**
 * GaussianBlur — Low-pass filter for noise reduction.
 *
 * Convolves the image with a Gaussian kernel. In medical imaging, this is
 * applied before edge detection to suppress acquisition noise from CT/MRI
 * detectors, preventing false edges in the Sobel output.
 *
 * The kernel is computed from the Gaussian function G(x,y) = exp(-(x²+y²)/2σ²)
 * and normalised so all weights sum to 1 (preserves mean brightness).
 *
 * Typical parameters:
 *   - kernelSize=3, sigma=1.0 — light smoothing
 *   - kernelSize=5, sigma=1.4 — standard medical preprocessing
 */
class GaussianBlur : public ConvolutionFilter {
    float sigma_;
public:
    /**
     * @param kernelSize Must be odd (3, 5, 7...). Larger = stronger blur.
     * @param sigma      Standard deviation of the Gaussian. Larger = smoother.
     */
    explicit GaussianBlur(int kernelSize = 3, float sigma = 1.0f);

    Image<uint8_t> apply(const Image<uint8_t>& input) const override;
    std::string name() const override;

private:
    std::vector<std::vector<float>> buildGaussianKernel(int size, float sigma) const;
};

// ─── Sobel Edge Detection ─────────────────────────────────────────────────────

/**
 * SobelFilter — First-order gradient-based edge detector.
 *
 * Computes the image gradient magnitude using two 3×3 derivative kernels:
 *
 *   Gx (horizontal gradient):     Gy (vertical gradient):
 *   [-1  0 +1]                    [-1 -2 -1]
 *   [-2  0 +2]                    [ 0  0  0]
 *   [-1  0 +1]                    [+1 +2 +1]
 *
 * Magnitude = clamp(sqrt(Gx² + Gy²), 0, 255)
 *
 * Medical relevance: Sobel edge detection is a foundational step in:
 *   - Organ boundary delineation in CT/MRI
 *   - Lesion contour extraction
 *   - Feature extraction before ML segmentation models (U-Net preprocessing)
 *
 * Complexity: O(W × H) — two fixed 3×3 convolutions.
 */
class SobelFilter : public Filter {
public:
    Image<uint8_t> apply(const Image<uint8_t>& input) const override;
    std::string name() const override { return "Sobel Edge Detection"; }

private:
    // The two Sobel kernels are applied separately then combined.
    static const std::vector<std::vector<float>> Gx_;
    static const std::vector<std::vector<float>> Gy_;

    float applyKernel(const Image<uint8_t>& img,
                      const std::vector<std::vector<float>>& kernel,
                      int row, int col) const;
};

// ─── Histogram Equalizer ──────────────────────────────────────────────────────

/**
 * HistogramEqualizer — Contrast enhancement via CDF normalisation.
 *
 * Redistributes pixel intensities to span the full [0, 255] range uniformly.
 * Pipeline: histogram → CDF → LUT (lookup table) → remap each pixel.
 *
 * Medical relevance: Low-contrast CT/MRI scans from older equipment or
 * suboptimal acquisition settings benefit from equalization to make subtle
 * tissue boundaries visible to radiologists.
 *
 * Complexity: O(W×H + 256) — linear in the number of pixels.
 */
class HistogramEqualizer : public Filter {
public:
    Image<uint8_t> apply(const Image<uint8_t>& input) const override;
    std::string name() const override { return "Histogram Equalization"; }

    /**
     * Step 1: Count pixel frequency for each intensity value [0,255].
     * Returns a 256-element array where hist[i] = number of pixels with value i.
     */
    std::array<int, 256> computeHistogram(const Image<uint8_t>& img) const;

    /**
     * Step 2: Build the Cumulative Distribution Function.
     * cdf[i] = sum of hist[0..i]. Represents the total pixels with value ≤ i.
     */
    std::array<int, 256> computeCDF(const std::array<int, 256>& hist) const;

    /**
     * Step 3: Normalise the CDF to [0,255] to build the Look-Up Table.
     * LUT[i] = round((cdf[i] - cdf_min) / (totalPixels - cdf_min) * 255)
     * This is the standard histogram equalisation formula.
     */
    std::array<uint8_t, 256> buildLUT(const std::array<int, 256>& cdf,
                                      int totalPixels) const;
};
