#pragma once

#include "Image.h"
#include <string>
#include <vector>
#include <thread>
#include <future>
#include <memory>
#include <array>

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
 * ConvolutionFilter — Multithreaded base class for all kernel-based spatial filters.
 *
 * Implements the shared pixel-neighbourhood loop. Subclasses provide the kernel.
 * Border handling uses zero-padding (see Utils.h for rationale).
 *
 * Multithreading: apply() partitions the image into horizontal row-bands and
 * dispatches each band to a separate std::async task. Output rows are written
 * to non-overlapping memory regions, so no mutex or synchronisation is needed —
 * a lock-free design. This matches how Siemens Syngo pipelines parallelise
 * slice-level image processing.
 *
 * Complexity: O(W × H × K²) where K is kernel size.
 * For a 3×3 kernel on a 512×512 image: ~2.36M multiply-accumulates per thread.
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

    /**
     * Process a horizontal band of rows [rowStart, rowEnd) into output.
     * Called from worker threads — writes to non-overlapping regions,
     * so this is safe to call concurrently without synchronisation.
     */
    void processRowBand(const Image<uint8_t>& input, Image<uint8_t>& output,
                        int rowStart, int rowEnd) const;

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
 * Uses multithreaded row-band processing inherited from ConvolutionFilter.
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

    std::array<int, 256> computeHistogram(const Image<uint8_t>& img) const;
    std::array<int, 256> computeCDF(const std::array<int, 256>& hist) const;
    std::array<uint8_t, 256> buildLUT(const std::array<int, 256>& cdf,
                                      int totalPixels) const;
};

// ─── Median Filter ────────────────────────────────────────────────────────────

/**
 * MedianFilter — Non-linear noise reduction filter.
 *
 * Replaces each pixel with the median value of its (kernelSize × kernelSize)
 * neighbourhood. Unlike Gaussian blur, median filtering preserves sharp edges
 * while removing noise — a critical property for medical imaging.
 *
 * Medical relevance:
 *   - Removes "salt-and-pepper" noise common in ultrasound acquisitions.
 *   - Preserves vessel boundaries and organ contours in CT/MRI better than
 *     any linear filter. Used in retinal OCT preprocessing pipelines.
 *
 * Algorithm: For each pixel, collect neighbourhood values into a buffer,
 * use std::nth_element (O(N) partial sort) to find the median in-place.
 * Complexity: O(W × H × K²) with a partial sort step per pixel.
 *
 * @param kernelSize  Neighbourhood size (must be odd: 3, 5, 7...).
 */
class MedianFilter : public Filter {
    int kernelSize_;
public:
    explicit MedianFilter(int kernelSize = 3);

    Image<uint8_t> apply(const Image<uint8_t>& input) const override;
    std::string name() const override;
};

// ─── Unsharp Mask ─────────────────────────────────────────────────────────────

/**
 * UnsharpMask — Edge-enhancing sharpening filter.
 *
 * Formula: output = clamp(original + amount * (original - blurred), 0, 255)
 *
 * The "unsharp" name comes from the darkroom technique: a blurred (unsharp)
 * negative is subtracted from the original to produce a high-pass detail
 * signal, which is then added back to amplify fine structures.
 *
 * Medical relevance:
 *   - Sharpens fine structures in digital X-rays (bone trabecular detail).
 *   - Enhances microcalcifications in mammography screening.
 *   - Used in CT window-levelling pipelines for radiologist review.
 *
 * @param amount      Sharpening strength. 1.0 = standard, 2.0+ = aggressive.
 * @param kernelSize  Size of the internal Gaussian blur kernel (must be odd).
 * @param sigma       Sigma of the internal Gaussian blur.
 */
class UnsharpMask : public Filter {
    float amount_;
    int   kernelSize_;
    float sigma_;
public:
    explicit UnsharpMask(float amount = 1.0f, int kernelSize = 3, float sigma = 1.0f);

    Image<uint8_t> apply(const Image<uint8_t>& input) const override;
    std::string name() const override;
};

// ─── Laplacian Filter ─────────────────────────────────────────────────────────

/**
 * LaplacianFilter — Second-order derivative edge detector.
 *
 * Applies the discrete Laplacian kernel to detect regions of rapid intensity
 * change. Unlike Sobel (gradient magnitude), the Laplacian responds to
 * zero-crossings — giving thinner, more precise edge localisation.
 *
 * Kernel (8-connectivity):
 *   [-1, -1, -1]
 *   [-1,  8, -1]
 *   [-1, -1, -1]
 *
 * Medical relevance:
 *   - More sensitive than Sobel to fine detail (hairline fractures in CT,
 *     nerve fibre boundaries in MRI).
 *   - Isotropic response — not biased toward horizontal/vertical edges.
 *   - Apply after Gaussian blur to suppress noise (LoG pipeline).
 *
 * Complexity: O(W × H) — fixed 3×3 kernel.
 */
class LaplacianFilter : public ConvolutionFilter {
public:
    LaplacianFilter();
    Image<uint8_t> apply(const Image<uint8_t>& input) const override;
    std::string name() const override { return "Laplacian Edge Detection"; }
};

// ─── Window / Level ───────────────────────────────────────────────────────────

/**
 * WindowLevel — Radiologist contrast control (CT/MRI window-level adjustment).
 *
 * Maps the sub-range [level - window/2, level + window/2] to the full output
 * range [0, 255]. The identical operation exposed by Siemens Syngo.via W/L sliders.
 *
 * Formula:
 *   lower  = level - window / 2
 *   output = clamp((pixel - lower) / window * 255, 0, 255)
 *
 * @param window  Intensity range to map to [0,255]. Smaller = higher contrast.
 * @param level   Centre of the window (default 128 = midpoint of 8-bit range).
 */
class WindowLevel : public Filter {
    float window_;
    float level_;
public:
    explicit WindowLevel(float window = 200.0f, float level = 128.0f);
    Image<uint8_t> apply(const Image<uint8_t>& input) const override;
    std::string name() const override;
};
