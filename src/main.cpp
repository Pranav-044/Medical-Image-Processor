/**
 * main.cpp — Medical Image Processor CLI v2.2
 *
 * Available filters: sobel, gaussian, equalize, median, unsharp, laplacian, windowlevel
 * Available pipelines: full, denoise, log (Laplacian of Gaussian)
 * Extra flags: --stats (print image statistics), --benchmark, --version
 */

#include "Image.h"
#include "Filter.h"
#include "ImageIO.h"
#include "Pipeline.h"
#include "Utils.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

static constexpr const char* VERSION = "2.2.0";

// ─── Argument Parsing ─────────────────────────────────────────────────────────

struct Args {
    std::string inputPath;
    std::string outputPath;
    std::vector<std::string> filters;
    std::string pipeline;
    bool benchmark = false;
    bool stats     = false;
    bool help      = false;
    bool version   = false;
};

void printUsage(const std::string& prog) {
    std::cout << "\nMedical Image Processor v" << VERSION << "\n"
              << "================================\n\n"
              << "Usage:\n"
              << "  " << prog << " --input <file.bmp> --output <file.bmp> [options]\n\n"
              << "Options:\n"
              << "  --filter <name>       Apply a filter. Can be repeated.\n"
              << "                        sobel, gaussian, equalize, median,\n"
              << "                        unsharp, laplacian, windowlevel\n"
              << "  --pipeline full       Gaussian → Equalize → Sobel\n"
              << "  --pipeline denoise    Median → Unsharp Mask\n"
              << "  --pipeline log        Gaussian → Laplacian (Laplacian of Gaussian)\n"
              << "  --stats               Print pixel statistics (min/max/mean/stddev)\n"
              << "  --benchmark           Print per-stage processing time\n"
              << "  --version             Print version and exit\n"
              << "  --help                Show this message\n\n"
              << "Examples:\n"
              << "  " << prog << " --input scan.bmp --output edges.bmp --filter laplacian --stats\n"
              << "  " << prog << " --input scan.bmp --output out.bmp   --filter windowlevel --filter sobel\n"
              << "  " << prog << " --input scan.bmp --output out.bmp   --pipeline log --benchmark\n\n";
}

Args parseArgs(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--version" || arg == "-v") {
            args.version = true;
        } else if (arg == "--input" && i + 1 < argc) {
            args.inputPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.outputPath = argv[++i];
        } else if (arg == "--filter" && i + 1 < argc) {
            args.filters.push_back(argv[++i]);
        } else if (arg == "--pipeline" && i + 1 < argc) {
            args.pipeline = argv[++i];
        } else if (arg == "--benchmark") {
            args.benchmark = true;
        } else if (arg == "--stats") {
            args.stats = true;
        } else {
            std::cerr << "Warning: unknown argument '" << arg << "'\n";
        }
    }
    return args;
}

// ─── Image Statistics ─────────────────────────────────────────────────────────

/**
 * Compute and print pixel statistics for a grayscale image.
 * Outputs: min, max, mean, standard deviation — the same metrics shown
 * in DICOM viewer histogram panels.
 */
void printStats(const Image<uint8_t>& img, const std::string& label) {
    int total = img.totalPixels();
    if (total == 0) return;

    long long sum = 0;
    uint8_t minVal = 255, maxVal = 0;

    for (int r = 0; r < img.getHeight(); ++r) {
        for (int c = 0; c < img.getWidth(); ++c) {
            uint8_t v = img(r, c, 0);
            sum += v;
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
    }

    double mean = static_cast<double>(sum) / total;

    double variance = 0.0;
    for (int r = 0; r < img.getHeight(); ++r)
        for (int c = 0; c < img.getWidth(); ++c) {
            double diff = static_cast<double>(img(r, c, 0)) - mean;
            variance += diff * diff;
        }
    double stddev = std::sqrt(variance / total);

    std::cout << "\n  [stats: " << label << "]\n"
              << "    Min:    " << static_cast<int>(minVal) << "\n"
              << "    Max:    " << static_cast<int>(maxVal) << "\n"
              << "    Mean:   " << std::fixed << std::setprecision(2) << mean << "\n"
              << "    StdDev: " << std::fixed << std::setprecision(2) << stddev << "\n";
}

// ─── Filter Factory ───────────────────────────────────────────────────────────

std::unique_ptr<Filter> makeFilter(const std::string& name) {
    if (name == "sobel")       return std::make_unique<SobelFilter>();
    if (name == "gaussian")    return std::make_unique<GaussianBlur>(3, 1.0f);
    if (name == "equalize")    return std::make_unique<HistogramEqualizer>();
    if (name == "median")      return std::make_unique<MedianFilter>(3);
    if (name == "unsharp")     return std::make_unique<UnsharpMask>(1.0f, 3, 1.0f);
    if (name == "laplacian")   return std::make_unique<LaplacianFilter>();
    if (name == "windowlevel") return std::make_unique<WindowLevel>(200.0f, 128.0f);
    throw std::invalid_argument("Unknown filter: '" + name
        + "'. Valid: sobel, gaussian, equalize, median, unsharp, laplacian, windowlevel");
}

// ─── Entry Point ──────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    try {
        Args args = parseArgs(argc, argv);

        if (args.version) {
            std::cout << "Medical Image Processor v" << VERSION << "\n";
            return 0;
        }

        if (args.help || argc == 1) {
            printUsage(argv[0]);
            return 0;
        }

        if (args.inputPath.empty() || args.outputPath.empty()) {
            std::cerr << "Error: --input and --output are required.\n";
            printUsage(argv[0]);
            return 1;
        }

        if (args.filters.empty() && args.pipeline.empty()) {
            std::cerr << "Error: specify at least one --filter or --pipeline.\n";
            printUsage(argv[0]);
            return 1;
        }

        // ── Load Input ────────────────────────────────────────────────────────
        std::cout << "Loading: " << args.inputPath << "\n";
        if (!fs::exists(args.inputPath))
            throw std::runtime_error("Input file not found: " + args.inputPath);

        Image<uint8_t> rgb = ImageIO::loadBMP(args.inputPath);
        std::cout << "  " << rgb.info() << "\n";

        Image<uint8_t> gray = (rgb.getChannels() == 3)
            ? ImageIO::toGrayscale(rgb)
            : rgb;
        std::cout << "  Converted to grayscale: " << gray.info() << "\n";

        // Print input statistics if requested
        if (args.stats) printStats(gray, "input");

        // ── Build Pipeline ────────────────────────────────────────────────────
        ProcessingPipeline pipeline;
        pipeline.withBenchmark(args.benchmark);

        if (!args.pipeline.empty()) {
            if (args.pipeline == "full") {
                std::cout << "\nRunning full diagnostic preprocessing pipeline:\n";
                pipeline.addStage(std::make_unique<GaussianBlur>(3, 1.0f));
                pipeline.addStage(std::make_unique<HistogramEqualizer>());
                pipeline.addStage(std::make_unique<SobelFilter>());
            } else if (args.pipeline == "denoise") {
                std::cout << "\nRunning denoise + detail enhancement pipeline:\n";
                pipeline.addStage(std::make_unique<MedianFilter>(3));
                pipeline.addStage(std::make_unique<UnsharpMask>(1.0f, 3, 1.0f));
            } else if (args.pipeline == "log") {
                std::cout << "\nRunning Laplacian of Gaussian (LoG) pipeline:\n";
                pipeline.addStage(std::make_unique<GaussianBlur>(3, 1.0f));
                pipeline.addStage(std::make_unique<LaplacianFilter>());
            } else {
                throw std::invalid_argument("Unknown pipeline: '" + args.pipeline
                    + "'. Valid: full, denoise, log");
            }
        } else {
            for (const auto& f : args.filters)
                pipeline.addStage(makeFilter(f));
        }

        pipeline.printStages();
        std::cout << "\nProcessing...\n";

        // ── Execute ───────────────────────────────────────────────────────────
        Image<uint8_t> result = pipeline.run(gray);

        // Print output statistics if requested
        if (args.stats) printStats(result, "output");

        // ── Save Output ───────────────────────────────────────────────────────
        std::cout << "Saving: " << args.outputPath << "\n";
        ImageIO::saveBMP(result, args.outputPath);
        std::cout << "Done. Output: " << fs::absolute(args.outputPath).string() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
