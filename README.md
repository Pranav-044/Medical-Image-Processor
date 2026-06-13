# Medical Image Processor

> Diagnostic imaging pipelines preprocess raw scans before radiologist review. This tool replicates foundational steps — contrast enhancement (histogram equalization), noise reduction (Gaussian and Median filtering), and feature extraction (Sobel / Laplacian edge detection) — used in modalities like CT and MRI.

Built from scratch in **C++17** with no external image libraries. Every algorithm is implemented at the mathematical level — the kind of transparent, auditable code required in regulated medical device software (IEC 62304).

---

## Architecture

```
medical-image-processor/
├── include/
│   ├── Image.h       # Templated Image<T> — core data structure
│   ├── Filter.h      # Abstract Filter + full filter hierarchy (7 filters)
│   ├── ImageIO.h     # Manual BMP file parsing (24-bit, BGR, bottom-up)
│   ├── Pipeline.h    # Composable filter chain with benchmarking
│   └── Utils.h       # saturate_cast, luminance, gradient magnitude, padding
├── src/
│   ├── Filter.cpp    # All filter implementations (multithreaded Gaussian)
│   ├── ImageIO.cpp
│   ├── Pipeline.cpp
│   └── main.cpp      # CLI: --filter, --pipeline full/denoise/log, --benchmark, --stats
├── tests/
│   └── test_filters.cpp   # Self-contained test suite (11 groups, 60+ assertions)
├── samples/               # Test BMP images
└── CMakeLists.txt
```

---

## Build

**Requirements:** CMake 3.16+, GCC 10+ or Clang 12+ or MSVC 19.14+ (C++17 required)

```bash
git clone https://github.com/Pranav-044/Medical-Image-Processor.git
cd Medical-Image-Processor

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Produces two binaries:
- `medimg` — the CLI tool
- `medimg_tests` — the test runner

### Optional: Install

```bash
cmake --install build --prefix /usr/local
```

---

## Usage

```bash
# Single filter
./build/medimg --input samples/scan.bmp --output out/edges.bmp --filter sobel

# Chain filters in sequence
./build/medimg --input samples/scan.bmp --output out/result.bmp --filter median --filter unsharp

# Full diagnostic preprocessing pipeline (Gaussian → Equalize → Sobel)
./build/medimg --input samples/scan.bmp --output out/result.bmp --pipeline full

# Denoise + detail enhancement pipeline (Median → Unsharp Mask)
./build/medimg --input samples/scan.bmp --output out/result.bmp --pipeline denoise

# Laplacian of Gaussian (LoG) pipeline (Gaussian → Laplacian)
./build/medimg --input samples/scan.bmp --output out/result.bmp --pipeline log

# With per-stage timing (microsecond precision)
./build/medimg --input samples/scan.bmp --output out/result.bmp --pipeline full --benchmark

# Print pixel statistics (min / max / mean / std dev)
./build/medimg --input samples/scan.bmp --output out/result.bmp --filter sobel --stats

# Print version
./build/medimg --version
```

**Available filters:** `sobel`, `gaussian`, `equalize`, `median`, `unsharp`, `laplacian`, `windowlevel`

**Available pipelines:** `full`, `denoise`, `log`

---

## Run Tests

```bash
./build/medimg_tests
```

---

## Filters

| Filter | Type | Complexity | Medical Use |
|---|---|---|---|
| **Gaussian Blur** | Linear convolution | O(W×H×K²) | Noise suppression before edge detection |
| **Sobel Edge Detection** | Gradient magnitude | O(W×H) | Organ boundary delineation in CT/MRI |
| **Histogram Equalization** | CDF normalisation | O(W×H + 256) | Contrast enhancement on low-exposure scans |
| **Median Filter** | Non-linear, nth_element | O(W×H×K²) | Salt-and-pepper noise removal in ultrasound |
| **Unsharp Mask** | High-pass amplification | O(W×H×K²) | Sharpening microcalcifications in mammography |
| **Laplacian Filter** | Second-order derivative | O(W×H) | Fine detail / hairline fracture detection in CT |
| **Window / Level** | Linear remap + clamp | O(W×H) | Radiologist contrast control (CT/MRI W/L sliders) |

---

## Design Decisions

### `Image<T>` — Flat 1D array, not `vector<vector<T>>`

Pixel data is stored as a single `unique_ptr<T[]>` in planar row-major order:

```
index = channel * (height * width) + row * width + col
```

**Why:** Contiguous memory gives sequential access during convolutions — critical for CPU cache performance on large medical scans (4096×4096 CT slices). This matches how ITK, OpenCV, and DICOM PixelData store images internally.

### Multithreaded Convolution (lock-free)

`GaussianBlur::apply()` uses `std::async` to partition the image into horizontal row-bands, dispatching each band to a separate OS thread. Because output rows are non-overlapping, no mutex or synchronisation is needed — a completely lock-free design:

```
Thread 0: rows [0,   H/4)
Thread 1: rows [H/4, H/2)
Thread 2: rows [H/2, 3H/4)
Thread 3: rows [3H/4, H)   ← std::thread::hardware_concurrency() threads
```

This is how Siemens Syngo.via parallelises slice-level image processing.

### `unique_ptr` for ownership

No raw `new`/`delete`. `unique_ptr<T[]>` enforces single ownership and automatic cleanup, preventing double-free and memory leak bugs that cause safety-critical failures in medical software.

### Abstract `Filter` base class

All filters implement a single interface: take an image, return a new image. The pipeline doesn't know which filter it's running — it just calls `apply()`. Adding a new filter (e.g., bilateral, anisotropic diffusion) requires zero changes to existing code — Open/Closed Principle.

The `[[nodiscard]]` attribute on `apply()` and `name()` ensures callers cannot accidentally discard the returned image — a compile-time safety net.

### Zero-padding for border handling

Out-of-bounds pixels during convolution are treated as 0. Standard for Sobel in medical imaging where anatomical ROIs are centred — border artefacts are acceptable.

---

## Algorithm Details

### Gaussian Blur — O(W × H × K²)

Convolves with a kernel computed from `G(x,y) = exp(-(x²+y²) / 2σ²)`, normalised to sum to 1. Parallelised across CPU cores.

### Sobel Edge Detection — O(W × H)

```
Gx = [[-1,  0, +1],    Gy = [[-1, -2, -1],
      [-2,  0, +2],          [ 0,  0,  0],
      [-1,  0, +1]]          [+1, +2, +1]]

magnitude = clamp(sqrt(Gx² + Gy²), 0, 255)
```

### Histogram Equalization — O(W × H + 256)

1. Compute frequency histogram (256 bins)
2. Build CDF: `cdf[i] = Σ hist[0..i]`
3. Build LUT: `lut[i] = (cdf[i] - cdf_min) / (N - cdf_min) × 255`
4. Remap pixels via LUT

### Median Filter — O(W × H × K²)

Collects neighbourhood pixels into a buffer, uses `std::nth_element` (O(N) partial sort) to find the median in-place without a full sort. Preserves edges — a property no linear filter can offer.

### Unsharp Mask — O(W × H × K²)

```
detail   = original - gaussian_blur(original)
output   = clamp(original + amount × detail, 0, 255)
```

### Laplacian Filter — O(W × H)

Applies the 8-connectivity discrete Laplacian kernel. Sum of all weights = 0 (pure high-pass). Absolute value taken so both positive and negative edge responses are preserved:

```
kernel = [[-1, -1, -1],
          [-1,  8, -1],
          [-1, -1, -1]]

output = clamp(|Laplacian(pixel)|, 0, 255)
```

Combine with Gaussian blur first (LoG pipeline) for noise-robust fine-edge detection.

### Window / Level — O(W × H)

```
lower  = level - window / 2
output = clamp((pixel - lower) / window × 255, 0, 255)
```

Identical to the W/L sliders in Siemens Syngo.via and OsiriX. Narrow window → high contrast; wide window → full dynamic range.

---

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Core Data Structure | ✅ | `Image<T>`, `Utils.h` |
| 2 — BMP I/O | ✅ | Manual 24-bit BMP read/write |
| 3 — Filter Hierarchy | ✅ | Gaussian (multithreaded), Sobel |
| 4 — Histogram Equalisation | ✅ | CDF-based LUT |
| 5 — Pipeline + CLI | ✅ | 3 predefined pipelines, `--benchmark`, `--stats`, `--version` |
| 6 — Tests | ✅ | 11 test groups, 60+ assertions |
| 7 — Advanced Filters | ✅ | Median Filter, Unsharp Mask |
| 8 — Multithreading | ✅ | Lock-free row-band parallelism |
| 9 — Extended Filters | ✅ | Laplacian (LoG pipeline), Window/Level |

---

## License

MIT
