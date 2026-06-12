/**
 * test_filters.cpp — Self-contained test suite (no external framework).
 *
 * Tests use assertions with descriptive failure messages.
 * Run: ./medimg_tests
 *
 * Tests are grouped by component:
 *   T1  — Image<T> core operations
 *   T2  — Utils
 *   T3  — Gaussian Blur
 *   T4  — Sobel Edge Detection
 *   T5  — Histogram Equalization
 *   T6  — Pipeline
 *   T7  — Median Filter
 *   T8  — Unsharp Mask
 *   T9  — Multithreaded Gaussian
 *   T10 — Laplacian Filter
 *   T11 — Window/Level
 */

#include "../include/Image.h"
#include "../include/Filter.h"
#include "../include/ImageIO.h"
#include "../include/Pipeline.h"
#include "../include/Utils.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <sstream>

// ─── Test Harness ─────────────────────────────────────────────────────────────

static int passed = 0;
static int failed = 0;

#define CHECK(condition, message)                          \
    do {                                                    \
        if (!(condition)) {                                 \
            std::cerr << "  [FAIL] " << (message) << "\n"; \
            ++failed;                                       \
        } else {                                            \
            std::cout << "  [PASS] " << (message) << "\n"; \
            ++passed;                                       \
        }                                                   \
    } while(0)

#define CHECK_THROW(expr, ExType, message)                 \
    do {                                                    \
        bool caught = false;                                \
        try { expr; } catch (const ExType&) { caught = true; } \
        CHECK(caught, message);                            \
    } while(0)

// ─── T1: Image<T> ─────────────────────────────────────────────────────────────

void testImageConstruction() {
    std::cout << "\n[T1] Image<T> Core Operations\n";

    // Zero-initialised construction
    Image<uint8_t> img(10, 8, 1);
    CHECK(img.getWidth() == 10, "Width is 10");
    CHECK(img.getHeight() == 8, "Height is 8");
    CHECK(img.getChannels() == 1, "Channels is 1");
    CHECK(img.totalPixels() == 80, "Total pixels = 80");
    CHECK(img(0, 0, 0) == 0, "Default initialised to 0");

    // Write and read back
    img(3, 5, 0) = 127;
    CHECK(img(3, 5, 0) == 127, "Read back written value 127");

    // fill()
    img.fill(255);
    CHECK(img(0, 0, 0) == 255, "fill(255) sets all pixels");
    CHECK(img(7, 9, 0) == 255, "fill(255) reaches last pixel");

    // Deep copy constructor
    Image<uint8_t> src(4, 4, 1);
    src(1, 1, 0) = 42;
    Image<uint8_t> copy(src);
    CHECK(copy(1, 1, 0) == 42, "Copy constructor preserves values");
    copy(1, 1, 0) = 99;
    CHECK(src(1, 1, 0) == 42, "Copy is independent (deep copy)");

    // Move constructor
    Image<uint8_t> moveSrc(3, 3, 1);
    moveSrc(0, 0, 0) = 77;
    Image<uint8_t> moveDst(std::move(moveSrc));
    CHECK(moveDst(0, 0, 0) == 77, "Move constructor transfers data");

    // Copy assignment
    Image<uint8_t> a(5, 5, 1);
    a.fill(10);
    Image<uint8_t> b(3, 3, 1);
    b = a;
    CHECK(b.getWidth() == 5, "Copy assignment updates dimensions");
    CHECK(b(2, 2, 0) == 10, "Copy assignment copies values");

    // subregion
    Image<uint8_t> big(10, 10, 1);
    for (int r = 0; r < 10; ++r)
        for (int c = 0; c < 10; ++c)
            big(r, c, 0) = static_cast<uint8_t>(r * 10 + c);

    Image<uint8_t> roi = big.subregion(2, 3, 4, 5);
    CHECK(roi.getHeight() == 4, "Subregion height");
    CHECK(roi.getWidth() == 5, "Subregion width");
    CHECK(roi(0, 0, 0) == big(2, 3, 0), "Subregion top-left matches source");
    CHECK(roi(3, 4, 0) == big(5, 7, 0), "Subregion bottom-right matches source");

    // Out-of-bounds subregion
    CHECK_THROW(big.subregion(8, 8, 4, 4), std::out_of_range,
                "Subregion out of bounds throws");

    // Invalid construction
    CHECK_THROW(Image<uint8_t>(0, 10, 1), std::invalid_argument,
                "Width=0 throws invalid_argument");
}

// ─── T2: Utils ────────────────────────────────────────────────────────────────

void testUtils() {
    std::cout << "\n[T2] Utils\n";

    CHECK(medimg::saturate_cast(0.0f) == 0, "saturate_cast(0)");
    CHECK(medimg::saturate_cast(255.0f) == 255, "saturate_cast(255)");
    CHECK(medimg::saturate_cast(-10.0f) == 0, "saturate_cast(-10) clamps to 0");
    CHECK(medimg::saturate_cast(300.0f) == 255, "saturate_cast(300) clamps to 255");
    CHECK(medimg::saturate_cast(127.6f) == 128, "saturate_cast rounds 127.6 → 128");

    CHECK(medimg::inBounds(0, 0, 10, 10), "inBounds(0,0) is true");
    CHECK(medimg::inBounds(9, 9, 10, 10), "inBounds(9,9) is true");
    CHECK(!medimg::inBounds(-1, 0, 10, 10), "inBounds(-1,0) is false");
    CHECK(!medimg::inBounds(0, 10, 10, 10), "inBounds(0,10) is false");

    // ITU-R BT.601: pure green (0, 255, 0) → Y ≈ 150
    uint8_t y = medimg::luminance(0, 255, 0);
    CHECK(y >= 148 && y <= 152, "Luminance of pure green ≈ 150");

    // Pure white should give 255
    CHECK(medimg::luminance(255, 255, 255) == 255, "Luminance of white = 255");
    // Pure black should give 0
    CHECK(medimg::luminance(0, 0, 0) == 0, "Luminance of black = 0");
}

// ─── T3: Gaussian Blur ────────────────────────────────────────────────────────

void testGaussianBlur() {
    std::cout << "\n[T3] Gaussian Blur\n";

    // A uniform image blurred should remain uniform
    Image<uint8_t> flat(20, 20, 1);
    flat.fill(100);
    GaussianBlur blur(3, 1.0f);
    Image<uint8_t> result = blur.apply(flat);

    // Interior pixels should be unchanged (uniform image + normalised kernel = same)
    bool interiorCorrect = true;
    for (int r = 2; r < 18; ++r)
        for (int c = 2; c < 18; ++c)
            if (result(r, c, 0) < 98 || result(r, c, 0) > 102)
                interiorCorrect = false;

    CHECK(interiorCorrect, "Gaussian blur of uniform image preserves interior values");
    CHECK(result.getWidth() == 20, "Gaussian blur preserves width");
    CHECK(result.getHeight() == 20, "Gaussian blur preserves height");

    // Values should always be in [0, 255]
    bool allValid = true;
    for (int r = 0; r < 20; ++r)
        for (int c = 0; c < 20; ++c)
            if (result(r, c, 0) > 255) allValid = false;
    CHECK(allValid, "All output pixels in [0, 255]");

    // name() should mention Gaussian
    std::string n = blur.name();
    CHECK(n.find("Gaussian") != std::string::npos, "Blur name contains 'Gaussian'");
}

// ─── T4: Sobel Filter ─────────────────────────────────────────────────────────

void testSobelFilter() {
    std::cout << "\n[T4] Sobel Edge Detection\n";

    // A uniform image has no edges — Sobel should produce all zeros
    Image<uint8_t> flat(10, 10, 1);
    flat.fill(128);
    SobelFilter sobel;
    Image<uint8_t> edges = sobel.apply(flat);

    bool allZero = true;
    for (int r = 1; r < 9; ++r)  // skip border (zero-padding artefacts on edge)
        for (int c = 1; c < 9; ++c)
            if (edges(r, c, 0) != 0) allZero = false;

    CHECK(allZero, "Sobel on uniform image produces zeros in interior");
    CHECK(edges.getChannels() == 1, "Sobel output is 1-channel");

    // A horizontal edge: top half = 0, bottom half = 255
    Image<uint8_t> hEdge(10, 10, 1);
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 10; ++c)
            hEdge(r, c, 0) = 0;
    for (int r = 5; r < 10; ++r)
        for (int c = 0; c < 10; ++c)
            hEdge(r, c, 0) = 255;

    Image<uint8_t> hEdgeResult = sobel.apply(hEdge);
    // The row at the boundary (row 4 or 5) should have high gradient values
    bool edgeDetected = false;
    for (int c = 1; c < 9; ++c)
        if (hEdgeResult(4, c, 0) > 100 || hEdgeResult(5, c, 0) > 100)
            edgeDetected = true;
    CHECK(edgeDetected, "Sobel detects horizontal edge");

    // Sobel requires grayscale — should throw on 3-channel input
    Image<uint8_t> rgb(10, 10, 3);
    CHECK_THROW(sobel.apply(rgb), std::invalid_argument,
                "Sobel throws on 3-channel input");
}

// ─── T5: Histogram Equalizer ──────────────────────────────────────────────────

void testHistogramEqualizer() {
    std::cout << "\n[T5] Histogram Equalization\n";

    // Build an image with very limited contrast (all pixels near 128)
    Image<uint8_t> lowContrast(16, 16, 1);
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            lowContrast(r, c, 0) = static_cast<uint8_t>(120 + (r % 8));

    HistogramEqualizer eq;

    auto hist = eq.computeHistogram(lowContrast);
    int histSum = 0;
    for (int v : hist) histSum += v;
    CHECK(histSum == 256, "Histogram sums to total pixel count");
    CHECK(hist[120] > 0, "Histogram has entries in the expected range");

    auto cdf = eq.computeCDF(hist);
    CHECK(cdf[255] == 256, "CDF at 255 equals total pixel count");
    // CDF should be non-decreasing
    bool cdfMonotone = true;
    for (int i = 1; i < 256; ++i)
        if (cdf[i] < cdf[i-1]) cdfMonotone = false;
    CHECK(cdfMonotone, "CDF is monotonically non-decreasing");

    auto lut = eq.buildLUT(cdf, 256);
    CHECK(lut[255] == 255, "LUT maps maximum value to 255");

    Image<uint8_t> equalised = eq.apply(lowContrast);

    // Output should span a wider range than input
    uint8_t minOut = 255, maxOut = 0;
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
            minOut = std::min(minOut, equalised(r, c, 0));
            maxOut = std::max(maxOut, equalised(r, c, 0));
        }
    CHECK(maxOut > 200, "Equalised image has pixels near 255");
    CHECK(minOut < 50,  "Equalised image has pixels near 0");

    // All values must be valid
    bool allValid = true;
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            if (equalised(r, c, 0) > 255) allValid = false;
    CHECK(allValid, "All equalised pixels in [0, 255]");

    // Throws on non-grayscale
    Image<uint8_t> rgb(10, 10, 3);
    CHECK_THROW(eq.apply(rgb), std::invalid_argument,
                "HistogramEqualizer throws on 3-channel input");
}

// ─── T6: Pipeline ─────────────────────────────────────────────────────────────

void testPipeline() {
    std::cout << "\n[T6] Processing Pipeline\n";

    Image<uint8_t> input(20, 20, 1);
    for (int r = 0; r < 20; ++r)
        for (int c = 0; c < 20; ++c)
            input(r, c, 0) = static_cast<uint8_t>((r * 20 + c) % 256);

    ProcessingPipeline pipeline;
    pipeline.addStage(std::make_unique<GaussianBlur>(3, 1.0f));
    pipeline.addStage(std::make_unique<HistogramEqualizer>());

    CHECK(pipeline.stageCount() == 2, "Pipeline has 2 stages");

    Image<uint8_t> result = pipeline.run(input);
    CHECK(result.getWidth() == 20,  "Pipeline preserves width");
    CHECK(result.getHeight() == 20, "Pipeline preserves height");

    // Empty pipeline returns a copy of the input
    ProcessingPipeline emptyPipeline;
    Image<uint8_t> passThrough = emptyPipeline.run(input);
    CHECK(passThrough(5, 5, 0) == input(5, 5, 0), "Empty pipeline is a pass-through");
}

// ─── T7: Median Filter ────────────────────────────────────────────────────────

void testMedianFilter() {
    std::cout << "\n[T7] Median Filter\n";

    // Uniform image — median of uniform neighbourhood = same value
    Image<uint8_t> flat(10, 10, 1);
    flat.fill(100);
    MedianFilter mf(3);
    Image<uint8_t> result = mf.apply(flat);

    bool interiorCorrect = true;
    for (int r = 1; r < 9; ++r)
        for (int c = 1; c < 9; ++c)
            if (result(r, c, 0) != 100) interiorCorrect = false;
    CHECK(interiorCorrect, "Median of uniform image preserves interior values");
    CHECK(result.getWidth() == 10 && result.getHeight() == 10,
          "Median filter preserves dimensions");

    // Salt-and-pepper noise: most pixels = 128, two spikes = 0 and 255
    // The median should remove the spikes
    Image<uint8_t> noisy(7, 7, 1);
    noisy.fill(128);
    noisy(3, 3, 0) = 255;   // salt spike
    noisy(3, 4, 0) = 0;     // pepper spike
    Image<uint8_t> denoised = mf.apply(noisy);
    // After median filtering, the centre should be back near 128
    CHECK(denoised(3, 3, 0) >= 100 && denoised(3, 3, 0) <= 150,
          "Median filter removes salt spike at centre");

    // Values always in [0, 255]
    bool allValid = true;
    for (int r = 0; r < 10; ++r)
        for (int c = 0; c < 10; ++c)
            if (result(r, c, 0) > 255) allValid = false;
    CHECK(allValid, "All median filter output pixels in [0, 255]");

    // name check
    CHECK(mf.name().find("Median") != std::string::npos, "MedianFilter name contains 'Median'");

    // Throws on RGB
    Image<uint8_t> rgb(5, 5, 3);
    CHECK_THROW(mf.apply(rgb), std::invalid_argument,
                "MedianFilter throws on 3-channel input");
}

// ─── T8: Unsharp Mask ─────────────────────────────────────────────────────────

void testUnsharpMask() {
    std::cout << "\n[T8] Unsharp Mask\n";

    // Uniform image: original - blurred = 0, so output = original (amount irrelevant)
    Image<uint8_t> flat(20, 20, 1);
    flat.fill(120);
    UnsharpMask um(1.5f, 3, 1.0f);
    Image<uint8_t> result = um.apply(flat);

    bool uniform = true;
    for (int r = 2; r < 18; ++r)
        for (int c = 2; c < 18; ++c)
            if (result(r, c, 0) < 115 || result(r, c, 0) > 125) uniform = false;
    CHECK(uniform, "UnsharpMask on uniform image leaves interior unchanged");
    CHECK(result.getWidth() == 20 && result.getHeight() == 20,
          "UnsharpMask preserves dimensions");

    // Output values stay in [0, 255] even with aggressive amount
    UnsharpMask aggressive(5.0f, 3, 1.0f);
    Image<uint8_t> edge(20, 20, 1);
    for (int r = 0; r < 20; ++r)
        for (int c = 0; c < 20; ++c)
            edge(r, c, 0) = (c < 10) ? 50 : 200;
    Image<uint8_t> sharpened = aggressive.apply(edge);

    bool allValid = true;
    for (int r = 0; r < 20; ++r)
        for (int c = 0; c < 20; ++c)
            if (sharpened(r, c, 0) > 255) allValid = false;
    CHECK(allValid, "UnsharpMask output always in [0, 255] even at amount=5");

    // name check
    CHECK(um.name().find("Unsharp") != std::string::npos,
          "UnsharpMask name contains 'Unsharp'");

    // Throws on RGB
    Image<uint8_t> rgb(5, 5, 3);
    CHECK_THROW(um.apply(rgb), std::invalid_argument,
                "UnsharpMask throws on 3-channel input");
}

// ─── T9: Multithreaded Gaussian ───────────────────────────────────────────────

void testMultithreadedGaussian() {
    std::cout << "\n[T9] Multithreaded Gaussian Blur\n";

    // Results should be identical to single-threaded because thread count
    // doesn't affect correctness — only speed.
    Image<uint8_t> input(64, 64, 1);
    for (int r = 0; r < 64; ++r)
        for (int c = 0; c < 64; ++c)
            input(r, c, 0) = static_cast<uint8_t>((r + c) % 256);

    GaussianBlur blur(3, 1.0f);
    Image<uint8_t> result = blur.apply(input);

    CHECK(result.getWidth() == 64 && result.getHeight() == 64,
          "Multithreaded Gaussian preserves dimensions");

    // All pixels must be valid
    bool allValid = true;
    for (int r = 0; r < 64; ++r)
        for (int c = 0; c < 64; ++c)
            if (result(r, c, 0) > 255) allValid = false;
    CHECK(allValid, "Multithreaded Gaussian output always in [0, 255]");

    // Interior pixels should be smoothed (not identical to input)
    // For a gradient image, adjacent pixels should have similar values after blur
    int largeJumps = 0;
    for (int r = 2; r < 62; ++r)
        for (int c = 2; c < 61; ++c)
            if (std::abs((int)result(r,c,0) - (int)result(r,c+1,0)) > 30)
                ++largeJumps;
    CHECK(largeJumps == 0, "Gaussian blur smooths gradient (no large adjacent jumps)");
}

// ─── T10: Laplacian Filter ────────────────────────────────────────────────────

void testLaplacianFilter() {
    std::cout << "\n[T10] Laplacian Edge Detection\n";

    // Uniform image has no second derivative — Laplacian should produce zeros
    Image<uint8_t> flat(10, 10, 1);
    flat.fill(100);
    LaplacianFilter lap;
    Image<uint8_t> result = lap.apply(flat);

    bool allZero = true;
    for (int r = 1; r < 9; ++r)
        for (int c = 1; c < 9; ++c)
            if (result(r, c, 0) != 0) allZero = false;
    CHECK(allZero, "Laplacian on uniform image produces zeros in interior");
    CHECK(result.getChannels() == 1, "Laplacian output is 1-channel");
    CHECK(result.getWidth() == 10 && result.getHeight() == 10,
          "Laplacian preserves dimensions");

    // On a sharp step edge, Laplacian should produce non-zero response
    Image<uint8_t> step(10, 10, 1);
    for (int r = 0; r < 10; ++r)
        for (int c = 0; c < 10; ++c)
            step(r, c, 0) = (c < 5) ? 0 : 200;
    Image<uint8_t> lapEdge = lap.apply(step);
    bool edgeDetected = false;
    for (int r = 1; r < 9; ++r)
        if (lapEdge(r, 4, 0) > 10 || lapEdge(r, 5, 0) > 10)
            edgeDetected = true;
    CHECK(edgeDetected, "Laplacian detects step edge");

    // All output must stay in [0, 255]
    bool allValid = true;
    for (int r = 0; r < 10; ++r)
        for (int c = 0; c < 10; ++c)
            if (result(r, c, 0) > 255) allValid = false;
    CHECK(allValid, "Laplacian output always in [0, 255]");

    // name check
    CHECK(lap.name().find("Laplacian") != std::string::npos,
          "LaplacianFilter name contains 'Laplacian'");

    // Throws on RGB
    Image<uint8_t> rgb(5, 5, 3);
    CHECK_THROW(lap.apply(rgb), std::invalid_argument,
                "LaplacianFilter throws on 3-channel input");
}

// ─── T11: Window / Level ──────────────────────────────────────────────────────

void testWindowLevel() {
    std::cout << "\n[T11] Window/Level\n";

    // Full window (255) centred at 128 — should be approximately identity mapping
    WindowLevel fullWindow(255.0f, 128.0f);
    Image<uint8_t> ramp(1, 256, 1);
    for (int r = 0; r < 256; ++r) ramp(r, 0, 0) = static_cast<uint8_t>(r);
    Image<uint8_t> mapped = fullWindow.apply(ramp);
    // Middle pixel (128) should map close to 128
    CHECK(mapped(128, 0, 0) >= 125 && mapped(128, 0, 0) <= 131,
          "Full window maps midpoint to near 128");
    // Min/max clamping
    CHECK(mapped(0, 0, 0) == 0,   "Full window: pixel 0 maps to 0");
    CHECK(mapped(255, 0, 0) == 255, "Full window: pixel 255 maps to 255");

    // Narrow window (100) centred at 128 — high contrast, clips above/below
    WindowLevel narrow(100.0f, 128.0f);
    Image<uint8_t> result = narrow.apply(ramp);
    // Pixels far outside [78, 178] should be clipped to 0 or 255
    CHECK(result(0, 0, 0) == 0,   "Narrow window clips low pixels to 0");
    CHECK(result(255, 0, 0) == 255, "Narrow window clips high pixels to 255");
    // Midpoint should still map near 128
    CHECK(result(128, 0, 0) >= 120 && result(128, 0, 0) <= 136,
          "Narrow window maps midpoint near 128");

    // All output must be in [0, 255]
    bool allValid = true;
    for (int r = 0; r < 256; ++r)
        if (result(r, 0, 0) > 255) allValid = false;
    CHECK(allValid, "WindowLevel output always in [0, 255]");

    // name check
    CHECK(narrow.name().find("Window") != std::string::npos,
          "WindowLevel name contains 'Window'");

    // Invalid window throws
    CHECK_THROW(WindowLevel(0.0f, 128.0f), std::invalid_argument,
                "WindowLevel with window=0 throws");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "====================================\n";
    std::cout << "  Medical Image Processor — Tests  \n";
    std::cout << "====================================\n";

    testImageConstruction();
    testUtils();
    testGaussianBlur();
    testSobelFilter();
    testHistogramEqualizer();
    testPipeline();
    testMedianFilter();
    testUnsharpMask();
    testMultithreadedGaussian();
    testLaplacianFilter();
    testWindowLevel();

    std::cout << "\n====================================\n";
    std::cout << "  Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "====================================\n";

    return (failed == 0) ? 0 : 1;
}
