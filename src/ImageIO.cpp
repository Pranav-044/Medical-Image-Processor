#include "ImageIO.h"
#include "Utils.h"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <vector>

// ─── Row stride ───────────────────────────────────────────────────────────────

int ImageIO::rowStride(int width) {
    // BMP rows must be padded to 4-byte (32-bit) boundaries.
    // Formula: stride = (width * bytesPerPixel + 3) & ~3
    return (width * 3 + 3) & ~3;
}

// ─── loadBMP ──────────────────────────────────────────────────────────────────

Image<uint8_t> ImageIO::loadBMP(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open BMP file: " + path);

    // Read file header (14 bytes)
    BMPFileHeader fileHeader;
    file.read(reinterpret_cast<char*>(&fileHeader), sizeof(BMPFileHeader));

    if (fileHeader.signature != 0x4D42)  // "BM" in little-endian
        throw std::runtime_error("Not a valid BMP file (bad signature): " + path);

    // Read DIB / info header (40 bytes for BITMAPINFOHEADER)
    BMPInfoHeader infoHeader;
    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(BMPInfoHeader));

    if (infoHeader.bitsPerPixel != 24)
        throw std::runtime_error("Only 24-bit BMP is supported. Got: "
            + std::to_string(infoHeader.bitsPerPixel) + "-bit");

    if (infoHeader.compression != 0)
        throw std::runtime_error("Only uncompressed BMP (BI_RGB) is supported.");

    int width  = infoHeader.width;
    // Negative height means top-down storage — we normalise to positive
    int height = std::abs(infoHeader.height);
    bool topDown = (infoHeader.height < 0);

    Image<uint8_t> img(width, height, 3);  // 3 channels: R, G, B

    int stride = rowStride(width);
    std::vector<uint8_t> rowBuf(stride);

    // Seek to pixel data
    file.seekg(fileHeader.dataOffset, std::ios::beg);

    for (int row = 0; row < height; ++row) {
        // BMP stores rows bottom-up by default unless topDown flag is set
        int targetRow = topDown ? row : (height - 1 - row);

        file.read(reinterpret_cast<char*>(rowBuf.data()), stride);
        if (!file)
            throw std::runtime_error("Unexpected end of BMP file while reading row "
                + std::to_string(row));

        for (int col = 0; col < width; ++col) {
            // BMP pixels are stored in BGR order (not RGB)
            uint8_t b = rowBuf[col * 3 + 0];
            uint8_t g = rowBuf[col * 3 + 1];
            uint8_t r = rowBuf[col * 3 + 2];

            img(targetRow, col, 0) = r;
            img(targetRow, col, 1) = g;
            img(targetRow, col, 2) = b;
        }
    }

    return img;
}

// ─── saveBMP ──────────────────────────────────────────────────────────────────

void ImageIO::saveBMP(const Image<uint8_t>& img, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot write BMP file: " + path);

    int width    = img.getWidth();
    int height   = img.getHeight();
    int channels = img.getChannels();
    int stride   = rowStride(width);

    uint32_t pixelDataSize = static_cast<uint32_t>(stride) * height;
    uint32_t fileSize = 54 + pixelDataSize;  // 14 (file hdr) + 40 (info hdr) + pixels

    // Write file header
    BMPFileHeader fileHeader{};
    fileHeader.signature  = 0x4D42;
    fileHeader.fileSize   = fileSize;
    fileHeader.reserved1  = 0;
    fileHeader.reserved2  = 0;
    fileHeader.dataOffset = 54;
    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(BMPFileHeader));

    // Write info header
    BMPInfoHeader infoHeader{};
    infoHeader.headerSize      = 40;
    infoHeader.width           = width;
    infoHeader.height          = height;  // positive = bottom-up storage
    infoHeader.colorPlanes     = 1;
    infoHeader.bitsPerPixel    = 24;
    infoHeader.compression     = 0;
    infoHeader.imageSize       = pixelDataSize;
    infoHeader.xPixelsPerMeter = 2835;  // 72 DPI
    infoHeader.yPixelsPerMeter = 2835;
    infoHeader.colorsInTable   = 0;
    infoHeader.importantColors = 0;
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(BMPInfoHeader));

    // Write pixel data bottom-up, BGR order, with row padding
    std::vector<uint8_t> rowBuf(stride, 0);
    for (int row = height - 1; row >= 0; --row) {
        for (int col = 0; col < width; ++col) {
            if (channels == 1) {
                // Grayscale: write same value for R, G, B
                uint8_t val = img(row, col, 0);
                rowBuf[col * 3 + 0] = val;  // B
                rowBuf[col * 3 + 1] = val;  // G
                rowBuf[col * 3 + 2] = val;  // R
            } else {
                // RGB → BGR
                rowBuf[col * 3 + 0] = img(row, col, 2);  // B
                rowBuf[col * 3 + 1] = img(row, col, 1);  // G
                rowBuf[col * 3 + 2] = img(row, col, 0);  // R
            }
        }
        file.write(reinterpret_cast<const char*>(rowBuf.data()), stride);
    }
}

// ─── toGrayscale ──────────────────────────────────────────────────────────────

Image<uint8_t> ImageIO::toGrayscale(const Image<uint8_t>& rgb) {
    if (rgb.getChannels() != 3)
        throw std::invalid_argument("toGrayscale requires a 3-channel RGB image.");

    Image<uint8_t> gray(rgb.getWidth(), rgb.getHeight(), 1);

    for (int r = 0; r < rgb.getHeight(); ++r) {
        for (int c = 0; c < rgb.getWidth(); ++c) {
            uint8_t R = rgb(r, c, 0);
            uint8_t G = rgb(r, c, 1);
            uint8_t B = rgb(r, c, 2);
            gray(r, c, 0) = medimg::luminance(R, G, B);
        }
    }
    return gray;
}
