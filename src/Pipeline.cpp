#include "Pipeline.h"
#include <numeric>

ProcessingPipeline& ProcessingPipeline::addStage(std::unique_ptr<Filter> f) {
    stages_.push_back(std::move(f));
    return *this;
}

ProcessingPipeline& ProcessingPipeline::withBenchmark(bool enable) {
    benchmark_ = enable;
    return *this;
}

Image<uint8_t> ProcessingPipeline::run(const Image<uint8_t>& input) const {
    if (stages_.empty())
        return input;

    Image<uint8_t> current = input;
    long long totalUs = 0;

    for (const auto& stage : stages_) {
        if (benchmark_) {
            auto t0 = std::chrono::high_resolution_clock::now();
            current = stage->apply(current);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            totalUs += us;

            // Format as ms with 3 decimal places for readability
            double ms = static_cast<double>(us) / 1000.0;
            std::cout << "  [bench] " << stage->name()
                      << " — " << std::fixed << std::setprecision(3) << ms << " ms\n";
        } else {
            current = stage->apply(current);
        }
    }

    if (benchmark_ && stages_.size() > 1) {
        double totalMs = static_cast<double>(totalUs) / 1000.0;
        std::cout << "  [bench] Total — "
                  << std::fixed << std::setprecision(3) << totalMs << " ms\n";
    }

    return current;
}

void ProcessingPipeline::printStages() const {
    std::cout << "Pipeline (" << stages_.size() << " stage"
              << (stages_.size() != 1 ? "s" : "") << "):\n";
    for (size_t i = 0; i < stages_.size(); ++i)
        std::cout << "  [" << (i + 1) << "] " << stages_[i]->name() << "\n";
}
