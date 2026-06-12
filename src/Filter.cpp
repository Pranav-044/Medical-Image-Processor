#include "../include/Filter.h"
#include "../include/Utils.h"
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <thread>
#include <future>
#include <vector>

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
            if (medimg::inBounds(ir, ic, img.getHeight(), img.getWidth())) {
                sum += kernel_[kr][kc] * static_cast<float>(img(ir, ic, ch));
            }
        }
    }
    return sum;
}

void ConvolutionFilter::processRowBand(const Image<uint8_t>& input,
                                       Image<uint8_t>& output,
                                       int rowStart, int rowEnd) const
{
    for (int ch = 0; ch < input.getChannels(); ++ch)
        for (int r = rowStart; r < rowEnd; ++r)
            for (int c = 0; c < input.getWidth(); ++c)
                output(r, c, ch) = medimg::saturate_cast(applyKernelAt(input, r, c, ch));
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

    for (int r = -half; r <= half; ++r) {
        for (int c = -half; c <= half; ++c) {
            float w = medimg::gaussianWeight(r, c, sigma);
            k[r + half][c + half] = w;
            sum += w;
        }
    }

    // Normalise: all weights sum to 1 (preserves mean brightness)
    for (auto& row : k)
        for (auto& val : row)
            val /= sum;

    return k;
}

Image<uint8_t> GaussianBlur::apply(const Image<uint8_t>& input) const {
    Image<uint8_t> output(input.getWidth(), input.getHeight(), input.getChannels());

    // Determine number of hardware threads available
    unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
    int height = input.getHeight();
    int rowsPerThread = std::max(1, height / static_cast<int>(numThreads));

    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);

    int rowStart = 0;
    while (rowStart < height) {
        int rowEnd = std::min(rowStart + rowsPerThread, height);
        // Capture by reference — safe because we wait for all futures before returning
        futures.push_back(std::async(std::launch::async,
            [this, &input, &output, rowStart, rowEnd]() {
                this->processRowBand(input, output, rowStart, rowEnd);
            }
        ));
        rowStart = rowEnd;
    }

    // Wait for all threads to finish
    for (auto& f : futures) f.get();

    return output;
}

std::string GaussianBlur::name() const {
    return "Gaussian Blur (" + std::to_string(kernelSize_) + "x"
         + std::to_string(kernelSize_) + ", sigma=" + std::to_string(sigma_) + ")";
}

// ─── SobelFilter ──────────────────────────────────────────────────────────────

const std::vector<std::vector<float>> SobelFilter::Gx_ = {
    {-1.0f,  0.0f,  1.0f},
    {-2.0f,  0.0f,  2.0f},
    {-1.0f,  0.0f,  1.0f}
};

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
    for (int kr = 0; kr < 3; ++kr) {
        for (int kc = 0; kc < 3; ++kc) {
            int ir = row + (kr - 1);
            int ic = col + (kc - 1);
            if (medimg::inBounds(ir, ic, img.getHeight(), img.getWidth()))
                sum += kernel[kr][kc] * static_cast<float>(img(ir, ic, 0));
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
    int cdfMin = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
    }
    int denominator = totalPixels - cdfMin;
    for (int i = 0; i < 256; ++i) {
        if (denominator <= 0)
            lut[i] = static_cast<uint8_t>(i);
        else {
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

// ─── MedianFilter ─────────────────────────────────────────────────────────────

MedianFilter::MedianFilter(int kernelSize) : kernelSize_(kernelSize) {
    if (kernelSize_ % 2 == 0 || kernelSize_ < 1)
        throw std::invalid_argument("Median filter kernel size must be a positive odd number.");
}

Image<uint8_t> MedianFilter::apply(const Image<uint8_t>& input) const {
    if (input.getChannels() != 1)
        throw std::invalid_argument("MedianFilter requires a grayscale (1-channel) image.");

    int h    = input.getHeight();
    int w    = input.getWidth();
    int half = kernelSize_ / 2;
    int mid  = (kernelSize_ * kernelSize_) / 2;

    Image<uint8_t> output(w, h, 1);
    std::vector<uint8_t> neighbourhood;
    neighbourhood.reserve(kernelSize_ * kernelSize_);

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            neighbourhood.clear();

            // Collect valid neighbourhood pixels (zero-padding: skip OOB)
            for (int kr = -half; kr <= half; ++kr) {
                for (int kc = -half; kc <= half; ++kc) {
                    int nr = r + kr;
                    int nc = c + kc;
                    if (medimg::inBounds(nr, nc, h, w))
                        neighbourhood.push_back(input(nr, nc, 0));
                }
            }

            // std::nth_element is O(N) — more efficient than full sort
            auto midIt = neighbourhood.begin() + neighbourhood.size() / 2;
            std::nth_element(neighbourhood.begin(), midIt, neighbourhood.end());
            output(r, c, 0) = *midIt;
        }
    }
    return output;
}

std::string MedianFilter::name() const {
    return "Median Filter (" + std::to_string(kernelSize_) + "x"
         + std::to_string(kernelSize_) + ")";
}

// ─── UnsharpMask ──────────────────────────────────────────────────────────────

UnsharpMask::UnsharpMask(float amount, int kernelSize, float sigma)
    : amount_(amount), kernelSize_(kernelSize), sigma_(sigma) {}

Image<uint8_t> UnsharpMask::apply(const Image<uint8_t>& input) const {
    if (input.getChannels() != 1)
        throw std::invalid_argument("UnsharpMask requires a grayscale (1-channel) image.");

    // Step 1: Blur the original image
    GaussianBlur blur(kernelSize_, sigma_);
    Image<uint8_t> blurred = blur.apply(input);

    // Step 2: output = clamp(original + amount * (original - blurred), 0, 255)
    int h = input.getHeight();
    int w = input.getWidth();
    Image<uint8_t> output(w, h, 1);

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            float orig  = static_cast<float>(input(r, c, 0));
            float blur_ = static_cast<float>(blurred(r, c, 0));
            float detail = orig - blur_;
            float sharpened = orig + amount_ * detail;
            output(r, c, 0) = medimg::saturate_cast(sharpened);
        }
    }
    return output;
}

std::string UnsharpMask::name() const {
    return "Unsharp Mask (amount=" + std::to_string(amount_) + ", sigma="
         + std::to_string(sigma_) + ")";
}

// ─── LaplacianFilter ──────────────────────────────────────────────────────────

// 8-connectivity Laplacian kernel — responds equally to edges in all directions.
// Sum of all weights = 0 (pure high-pass: DC component is eliminated).
LaplacianFilter::LaplacianFilter()
    : ConvolutionFilter({{
        {-1.0f, -1.0f, -1.0f},
        {-1.0f,  8.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f}
    }})
{}

Image<uint8_t> LaplacianFilter::apply(const Image<uint8_t>& input) const {
    if (input.getChannels() != 1)
        throw std::invalid_argument("LaplacianFilter requires a grayscale (1-channel) image.");

    Image<uint8_t> output(input.getWidth(), input.getHeight(), 1);

    for (int r = 0; r < input.getHeight(); ++r) {
        for (int c = 0; c < input.getWidth(); ++c) {
            float val = applyKernelAt(input, r, c, 0);
            // Take absolute value — edges produce both positive and negative responses
            output(r, c, 0) = medimg::saturate_cast(std::abs(val));
        }
    }
    return output;
}

// ─── WindowLevel ──────────────────────────────────────────────────────────────

WindowLevel::WindowLevel(float window, float level)
    : window_(window), level_(level)
{
    if (window_ <= 0.0f)
        throw std::invalid_argument("WindowLevel: window must be positive.");
}

Image<uint8_t> WindowLevel::apply(const Image<uint8_t>& input) const {
    float lower = level_ - window_ / 2.0f;
    Image<uint8_t> output(input.getWidth(), input.getHeight(), input.getChannels());

    for (int ch = 0; ch < input.getChannels(); ++ch) {
        for (int r = 0; r < input.getHeight(); ++r) {
            for (int c = 0; c < input.getWidth(); ++c) {
                float pixel = static_cast<float>(input(r, c, ch));
                float mapped = (pixel - lower) / window_ * 255.0f;
                output(r, c, ch) = medimg::saturate_cast(mapped);
            }
        }
    }
    return output;
}

std::string WindowLevel::name() const {
    return "Window/Level (W=" + std::to_string(static_cast<int>(window_))
         + ", L=" + std::to_string(static_cast<int>(level_)) + ")";
}
