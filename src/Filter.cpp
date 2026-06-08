#include "../include/Filter.h"
#include "../include/Utils.h"
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>

// ─── ConvolutionFilter ─────────────────────────────────────────────────────────

ConvolutionFilter::ConvolutionFilter(std::vector<std::vector<float>> kernel)
    : kernel_(std::move(kernel))
{
    kernelSize_ = static_cast<int>(kernel_.size());
    if (kernelSize_ == 0 || kernel_[0].size() != kernel_.size())
        throw std::invalid_argument("Kernel must be square and non-empty.");
    if (kernelSize_ % 2 == 0)
        throw std::invalid_argument("Kernel size must be odd.");
}

float ConvolutionFilter::applyKernelAt(const Image<uint8_t>& img,
                                        int row, int col, int ch) const
{
    float sum = 0.0f;
    int half = kernelSize_ / 2;

    for (int kr = 0; kr < kernelSize_; ++kr) {
        for (int kc = 0; kc < kernelSize_; ++kc) {
            int ir = row + (kr - half);
            int ic = col + (kc - half);

            // Zero-padding: out-of-bounds pixels contribute 0
            if (medimg::inBounds(ir, ic, img.getHeight(), img.getWidth())) {
                sum += kernel_[kr][kc] * static_cast<float>(img(ir, ic, ch));
            }
        }
    }
    return sum;
}

// ─── GaussianBlur ─────────────────────────────────────────────────────────────

GaussianBlur::GaussianBlur(int kernelSize, float sigma)
    : ConvolutionFilter(buildGaussianKernel(kernelSize, sigma))
    , sigma_(sigma)
{
    if (kernelSize % 2 == 0)
        throw std::invalid_argument("Gaussian kernel size must be odd.");
}

std::vector<std::vector<float>>
GaussianBlur::buildGaussianKernel(int size, float sigma) const
{
    std::vector<std::vector<float>> k(size, std::vector<float>(size, 0.0f));
    int half = size / 2;
    float sum = 0.0f;

    // Compute raw Gaussian weights
    for (int r = -half; r <= half; ++r) {
        for (int c = -half; c <= half; ++c) {
            float w = medimg::gaussianWeight(r, c, sigma);
            k[r + half][c + half] = w;
            sum += w;
        }
    }

    // Normalise so all weights sum to 1 (preserves mean brightness)
    for (auto& row : k)
        for (auto& val : row)
            val /= sum;

    return k;
}

Image<uint8_t> GaussianBlur::apply(const Image<uint8_t>& input) const {
    Image<uint8_t> output(input.getWidth(), input.getHeight(), input.getChannels());

    for (int ch = 0; ch < input.getChannels(); ++ch)
        for (int r = 0; r < input.getHeight(); ++r)
            for (int c = 0; c < input.getWidth(); ++c)
                output(r, c, ch) = medimg::saturate_cast(applyKernelAt(input, r, c, ch));

    return output;
}

std::string GaussianBlur::name() const {
    return "Gaussian Blur (" + std::to_string(kernelSize_) + "x"
         + std::to_string(kernelSize_) + ", sigma=" + std::to_string(sigma_) + ")";
}

// ─── SobelFilter ──────────────────────────────────────────────────────────────

// Horizontal gradient kernel: detects vertical edges
const std::vector<std::vector<float>> SobelFilter::Gx_ = {
    {-1.0f,  0.0f,  1.0f},
    {-2.0f,  0.0f,  2.0f},
    {-1.0f,  0.0f,  1.0f}
};

// Vertical gradient kernel: detects horizontal edges
const std::vector<std::vector<float>> SobelFilter::Gy_ = {
    {-1.0f, -2.0f, -1.0f},
    { 0.0f,  0.0f,  0.0f},
    { 1.0f,  2.0f,  1.0f}
};

float SobelFilter::applyKernel(const Image<uint8_t>& img,
                                const std::vector<std::vector<float>>& kernel,
                                int row, int col) const
{
    float sum = 0.0f;
    int half = 1; // 3x3 kernel

    for (int kr = 0; kr < 3; ++kr) {
        for (int kc = 0; kc < 3; ++kc) {
            int ir = row + (kr - half);
            int ic = col + (kc - half);
            if (medimg::inBounds(ir, ic, img.getHeight(), img.getWidth())) {
                sum += kernel[kr][kc] * static_cast<float>(img(ir, ic, 0));
            }
        }
    }
    return sum;
}

Image<uint8_t> SobelFilter::apply(const Image<uint8_t>& input) const {
    if (input.getChannels() != 1)
        throw std::invalid_argument("Sobel filter requires a grayscale (1-channel) image.");

    Image<uint8_t> output(input.getWidth(), input.getHeight(), 1);

    for (int r = 0; r < input.getHeight(); ++r) {
        for (int c = 0; c < input.getWidth(); ++c) {
            float gx = applyKernel(input, Gx_, r, c);
            float gy = applyKernel(input, Gy_, r, c);
            output(r, c, 0) = medimg::gradientMagnitude(gx, gy);
        }
    }
    return output;
}

// ─── HistogramEqualizer ───────────────────────────────────────────────────────

std::array<int, 256>
HistogramEqualizer::computeHistogram(const Image<uint8_t>& img) const
{
    std::array<int, 256> hist{};
    hist.fill(0);

    for (int r = 0; r < img.getHeight(); ++r)
        for (int c = 0; c < img.getWidth(); ++c)
            ++hist[img(r, c, 0)];

    return hist;
}

std::array<int, 256>
HistogramEqualizer::computeCDF(const std::array<int, 256>& hist) const
{
    std::array<int, 256> cdf{};
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i)
        cdf[i] = cdf[i - 1] + hist[i];
    return cdf;
}

std::array<uint8_t, 256>
HistogramEqualizer::buildLUT(const std::array<int, 256>& cdf, int totalPixels) const
{
    std::array<uint8_t, 256> lut{};

    // Find the minimum non-zero CDF value (cdf_min)
    int cdfMin = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
    }

    int denominator = totalPixels - cdfMin;
    for (int i = 0; i < 256; ++i) {
        if (denominator <= 0) {
            lut[i] = static_cast<uint8_t>(i);
        } else {
            float val = static_cast<float>(cdf[i] - cdfMin) / denominator * 255.0f;
            lut[i] = medimg::saturate_cast(val);
        }
    }
    return lut;
}

Image<uint8_t> HistogramEqualizer::apply(const Image<uint8_t>& input) const {
    if (input.getChannels() != 1)
        throw std::invalid_argument("Histogram equalizer requires a grayscale (1-channel) image.");

    auto hist = computeHistogram(input);
    auto cdf  = computeCDF(hist);
    auto lut  = buildLUT(cdf, input.totalPixels());

    Image<uint8_t> output(input.getWidth(), input.getHeight(), 1);
    for (int r = 0; r < input.getHeight(); ++r)
        for (int c = 0; c < input.getWidth(); ++c)
            output(r, c, 0) = lut[input(r, c, 0)];

    return output;
}
