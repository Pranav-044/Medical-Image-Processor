/**
 * test_filters.cpp — Self-contained test suite (no external framework).
 *
 * Tests use assertions with descriptive failure messages.
 * Run: ./medimg_tests
 *
 * Tests are grouped by component:
 *   T1 — Image<T> core operations
 *   T2 — BMP I/O
 *   T3 — Gaussian Blur
 *   T4 — Sobel Edge Detection
 *   T5 — Histogram Equalization
 *   T6 — Pipeline
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
    // Fill with gradient 0-255
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

    std::cout << "\n====================================\n";
    std::cout << "  Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "====================================\n";

    return (failed == 0) ? 0 : 1;
}
