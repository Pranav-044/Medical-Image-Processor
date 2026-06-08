#pragma once

#include "Image.h"
#include <string>
#include <stdexcept>

/**
 * ImageIO.h — BMP file I/O and colour space conversion.
 *
 * Implements manual BMP parsing without external libraries.
 * Understanding the BMP binary format is a practical low-level C++ exercise
 * directly applicable to working with proprietary medical imaging formats (DICOM).
 *
 * BMP format quirks handled here:
 *   1. Pixels stored BOTTOM-UP (row 0 in file = last row visually)
 *   2. BGR channel order (not RGB)
 *   3. Row stride padded to 4-byte boundaries
 *   4. 54-byte header (14-byte file header + 40-byte DIB header)
 *
 * Only uncompressed 24-bit BMP (BI_RGB) is supported — the format used by
 * most medical image converters when exporting from DICOM for preview.
 */

// ─── BMP Binary Structures ────────────────────────────────────────────────────
// These map directly onto the BMP file bytes. packed attribute prevents
// compiler from inserting padding between fields.

#pragma pack(push, 1)

struct BMPFileHeader {
    uint16_t signature;    // 0x4D42 = "BM" in little-endian
    uint32_t fileSize;     // Total file size in bytes
    uint16_t reserved1;    // Unused (0)
    uint16_t reserved2;    // Unused (0)
    uint32_t dataOffset;   // Byte offset to pixel data from file start
};

struct BMPInfoHeader {
    uint32_t headerSize;   // Size of this header (40 bytes for BITMAPINFOHEADER)
    int32_t  width;        // Image width in pixels
    int32_t  height;       // Image height (negative = top-down, positive = bottom-up)
    uint16_t colorPlanes;  // Must be 1
    uint16_t bitsPerPixel; // 24 for this implementation
    uint32_t compression;  // 0 = BI_RGB (no compression)
    uint32_t imageSize;    // Raw pixel data size (can be 0 for BI_RGB)
    int32_t  xPixelsPerMeter;
    int32_t  yPixelsPerMeter;
    uint32_t colorsInTable;
    uint32_t importantColors;
};

#pragma pack(pop)

// ─── ImageIO class ────────────────────────────────────────────────────────────

class ImageIO {
public:
    /**
     * Load a 24-bit uncompressed BMP file.
     * Returns a 3-channel Image<uint8_t> with channels in RGB order.
     *
     * Throws std::runtime_error if:
     *   - File cannot be opened
     *   - BMP signature is invalid
     *   - Bit depth is not 24
     *   - Compression is not BI_RGB
     */
    static Image<uint8_t> loadBMP(const std::string& path);

    /**
     * Save a grayscale or RGB Image<uint8_t> as a 24-bit BMP.
     * For grayscale input (1 channel), each pixel is written as R=G=B.
     *
     * Throws std::runtime_error if the file cannot be written.
     */
    static void saveBMP(const Image<uint8_t>& img, const std::string& path);

    /**
     * Convert a 3-channel RGB image to 1-channel grayscale.
     * Uses the ITU-R BT.601 luminance formula:
     *   Y = 0.299*R + 0.587*G + 0.114*B
     *
     * This is the standard used in JPEG encoding and broadcast television.
     * It accounts for the human eye's higher sensitivity to green wavelengths.
     *
     * Throws std::invalid_argument if input does not have exactly 3 channels.
     */
    static Image<uint8_t> toGrayscale(const Image<uint8_t>& rgb);

    /**
     * Returns the row stride (bytes per row) for a BMP with the given width.
     * BMP rows must be padded to 4-byte boundaries.
     * stride = (width * 3 + 3) & ~3
     */
    static int rowStride(int width);
};
