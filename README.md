# Medical Image Processor

> Diagnostic imaging pipelines preprocess raw scans before radiologist review. This tool replicates two foundational steps — contrast enhancement (histogram equalization) and feature extraction (Sobel edge detection) — used in modalities like CT and MRI.

Built from scratch in **C++17** with no external image libraries. Every algorithm is implemented at the mathematical level — the kind of transparent, auditable code required in regulated medical device software (IEC 62304).

---

## Architecture

```
medical-image-processor/
├── include/
│   ├── Image.h       # Templated Image<T> — core data structure
│   ├── Filter.h      # Abstract Filter + ConvolutionFilter + Sobel + Gaussian + Histogram
│   ├── ImageIO.h     # Manual BMP file parsing (24-bit, BGR, bottom-up)
│   ├── Pipeline.h    # Composable filter chain with benchmarking
│   └── Utils.h       # saturate_cast, luminance, gradient magnitude, padding
├── src/
│   ├── Filter.cpp
│   ├── ImageIO.cpp
│   ├── Pipeline.cpp
│   └── main.cpp      # CLI with --filter, --pipeline, --benchmark flags
├── tests/
│   └── test_filters.cpp   # Self-contained test suite (no framework)
├── samples/               # Test BMP images
└── CMakeLists.txt
```

---

## Build

**Requirements:** CMake 3.16+, GCC 10+ or Clang 12+ (C++17 required)

```bash
git clone https://github.com/wrewre/Medical-Image-Processor.git
cd Medical-Image-Processor

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces two binaries in `build/`:
- `medimg` — the main CLI tool
- `medimg_tests` — the test runner

---

## Usage

```bash
# Single filter
./build/medimg --input samples/scan.bmp --output out/edges.bmp --filter sobel

# Multiple filters applied in sequence
./build/medimg --input samples/scan.bmp --output out/result.bmp --filter gaussian --filter equalize

# Full diagnostic preprocessing pipeline (Gaussian → Equalize → Sobel)
./build/medimg --input samples/scan.bmp --output out/result.bmp --pipeline full

# With per-stage timing
./build/medimg --input samples/scan.bmp --output out/result.bmp --pipeline full --benchmark
```

Available filters: `sobel`, `gaussian`, `equalize`

---

## Run Tests

```bash
./build/medimg_tests
```

---

## Design Decisions

### `Image<T>` — Flat 1D array, not `vector<vector<T>>`

Pixel data is stored as a single `unique_ptr<T[]>` in row-major order:

```
index = channel * (height * width) + row * width + col
```

**Why:** A 2D vector allocates each row separately — that's N heap allocations with no memory locality. A flat array stores the entire image contiguously, which is critical for cache performance when iterating over millions of pixels during convolution. This is how ITK, OpenCV, and the DICOM PixelData attribute store images internally.

### `unique_ptr` for ownership

No raw `new`/`delete`. `unique_ptr<T[]>` enforces single ownership and automatic cleanup, preventing the double-free and memory leak bugs that cause safety-critical failures in medical software.

### Abstract `Filter` base class with pure virtual `apply()`

All filters implement a single interface: take an image, return a new image. The pipeline doesn't know which filter it's running — it just calls `apply()`. This is the **Strategy pattern**, which allows adding new filters (median, Laplacian, bilateral) without modifying any existing code.

### Zero-padding for border handling

When a convolution kernel extends outside the image boundary, out-of-bounds pixels are treated as 0. Alternatives:

| Strategy | Description | Best for |
|---|---|---|
| **Zero-padding** (used here) | Border pixels treated as 0 | Edge detection (Sobel) |
| Reflection | Border mirrored across edge | Gaussian blur |
| Replicate | Edge pixel value repeated | Gradient estimation |

Zero-padding is standard for Sobel in medical imaging because anatomical ROIs are centred — minor border darkening is acceptable.

---

## Algorithm Details

### Gaussian Blur — O(W × H × K²)

Convolves with a kernel computed from `G(x,y) = exp(-(x²+y²) / 2σ²)`, normalised to sum to 1. Applied before Sobel to suppress CT/MRI acquisition noise.

### Sobel Edge Detection — O(W × H)

Two fixed 3×3 kernels applied separately:

```
Gx = [[-1,  0, +1],    Gy = [[-1, -2, -1],
      [-2,  0, +2],          [ 0,  0,  0],
      [-1,  0, +1]]          [+1, +2, +1]]

magnitude = clamp(sqrt(Gx² + Gy²), 0, 255)
```

### Histogram Equalization — O(W × H + 256)

1. Compute frequency histogram (256 bins)
2. Build CDF: `cdf[i] = Σ hist[0..i]`
3. Normalise to LUT: `lut[i] = (cdf[i] - cdf_min) / (N - cdf_min) × 255`
4. Remap each pixel through the LUT

Medical relevance: used to improve contrast in low-exposure CT scans and older MRI equipment before radiologist review.

---

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Core Data Structure | ✅ | `Image<T>`, `Utils.h` |
| 2 — BMP I/O | ✅ | Manual 24-bit BMP read/write |
| 3 — Filter Hierarchy | ✅ | Gaussian, Sobel |
| 4 — Histogram Equalisation | ✅ | CDF-based LUT |
| 5 — Pipeline + CLI | ✅ | Composable pipeline, `--benchmark` |
| 6 — Tests | ✅ | Self-contained test suite |

---

## License

MIT
