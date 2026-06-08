/**
 * main.cpp — Medical Image Processor CLI
 *
 * Usage:
 *   ./medimg --input scan.bmp --output result.bmp --filter sobel
 *   ./medimg --input scan.bmp --output result.bmp --filter gaussian
 *   ./medimg --input scan.bmp --output result.bmp --filter equalize
 *   ./medimg --input scan.bmp --output result.bmp --pipeline full
 *   ./medimg --input scan.bmp --output result.bmp --filter sobel --benchmark
 *
 * --pipeline full runs: Gaussian Blur → Histogram Equalization → Sobel
 * This is a realistic preprocessing sequence for CT/MRI edge-enhanced imaging.
 */

#include "../include/Image.h"
#include "../include/Filter.h"
#include "../include/ImageIO.h"
#include "../include/Pipeline.h"
#include "../include/Utils.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

// ─── Argument Parsing ─────────────────────────────────────────────────────────

struct Args {
    std::string inputPath;
    std::string outputPath;
    std::vector<std::string> filters;  // may be multiple (applied in order)
    std::string pipeline;              // "full" or empty
    bool benchmark = false;
    bool help = false;
};

void printUsage(const std::string& prog) {
    std::cout << "\nMedical Image Processor v1.0\n"
              << "============================\n\n"
              << "Usage:\n"
              << "  " << prog << " --input <file.bmp> --output <file.bmp> [options]\n\n"
              << "Options:\n"
              << "  --filter <name>      Apply a filter. Can be repeated.\n"
              << "                       Values: sobel, gaussian, equalize\n"
              << "  --pipeline full      Run: Gaussian → Equalize → Sobel\n"
              << "  --benchmark          Print per-stage processing time in ms\n"
              << "  --help               Show this message\n\n"
              << "Examples:\n"
              << "  " << prog << " --input chest.bmp --output edges.bmp --filter sobel\n"
              << "  " << prog << " --input scan.bmp  --output out.bmp   --pipeline full --benchmark\n\n";
}

Args parseArgs(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            args.help = true;
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
        } else {
            std::cerr << "Warning: unknown argument '" << arg << "'\n";
        }
    }
    return args;
}

// ─── Filter Factory ───────────────────────────────────────────────────────────

std::unique_ptr<Filter> makeFilter(const std::string& name) {
    if (name == "sobel")    return std::make_unique<SobelFilter>();
    if (name == "gaussian") return std::make_unique<GaussianBlur>(3, 1.0f);
    if (name == "equalize") return std::make_unique<HistogramEqualizer>();
    throw std::invalid_argument("Unknown filter: '" + name
        + "'. Valid options: sobel, gaussian, equalize");
}

// ─── Entry Point ──────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    try {
        Args args = parseArgs(argc, argv);

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

        // Convert to grayscale (all filters operate on single-channel images)
        Image<uint8_t> gray = (rgb.getChannels() == 3)
            ? ImageIO::toGrayscale(rgb)
            : rgb;
        std::cout << "  Converted to grayscale: " << gray.info() << "\n";

        // ── Build Pipeline ────────────────────────────────────────────────────
        ProcessingPipeline pipeline;
        pipeline.withBenchmark(args.benchmark);

        if (!args.pipeline.empty()) {
            // Predefined pipeline
            if (args.pipeline == "full") {
                std::cout << "\nRunning full diagnostic preprocessing pipeline:\n";
                pipeline.addStage(std::make_unique<GaussianBlur>(3, 1.0f));
                pipeline.addStage(std::make_unique<HistogramEqualizer>());
                pipeline.addStage(std::make_unique<SobelFilter>());
            } else {
                throw std::invalid_argument("Unknown pipeline: '" + args.pipeline
                    + "'. Valid: full");
            }
        } else {
            // User-specified filter sequence
            for (const auto& f : args.filters)
                pipeline.addStage(makeFilter(f));
        }

        pipeline.printStages();
        std::cout << "\nProcessing...\n";

        // ── Execute ───────────────────────────────────────────────────────────
        Image<uint8_t> result = pipeline.run(gray);

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
