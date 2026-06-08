#pragma once

#include "Filter.h"
#include "Image.h"
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <iostream>

/**
 * Pipeline.h — Composable filter chain with optional benchmarking.
 *
 * Design Pattern: Chain of Responsibility
 *   Each stage transforms the image and passes the result to the next.
 *   Stages are owned via unique_ptr — the pipeline owns all filters.
 *
 * The ProcessingPipeline::run() method executes stages sequentially.
 * With benchmarking enabled, it reports per-stage latency in milliseconds —
 * the same metric used in Siemens Syngo.via performance profiling.
 *
 * Predefined pipelines (--pipeline full):
 *   Gaussian Blur → Histogram Equalization → Sobel Edge Detection
 *   This sequence mirrors real CT preprocessing:
 *     1. Noise reduction (Gaussian)
 *     2. Contrast normalisation (histogram equalisation)
 *     3. Feature extraction for segmentation (Sobel)
 */
class ProcessingPipeline {
    std::vector<std::unique_ptr<Filter>> stages_;
    bool benchmark_ = false;

public:
    /**
     * Add a filter stage. Fluent interface allows chaining:
     *   pipeline.addStage(...).addStage(...).addStage(...)
     */
    ProcessingPipeline& addStage(std::unique_ptr<Filter> f);

    /**
     * Enable per-stage timing output to stdout.
     */
    ProcessingPipeline& withBenchmark(bool enable = true);

    /**
     * Execute all stages in sequence.
     * Each stage receives the output of the previous stage as its input.
     */
    Image<uint8_t> run(const Image<uint8_t>& input) const;

    /**
     * Print the pipeline configuration (stage names in order).
     */
    void printStages() const;

    /**
     * Returns the number of stages in this pipeline.
     */
    size_t stageCount() const { return stages_.size(); }
};
