#include "../include/Pipeline.h"

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

    for (const auto& stage : stages_) {
        if (benchmark_) {
            auto t0 = std::chrono::high_resolution_clock::now();
            current = stage->apply(current);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::cout << "[benchmark] " << stage->name()
                      << " — " << ms << " ms\n";
        } else {
            current = stage->apply(current);
        }
    }
    return current;
}

void ProcessingPipeline::printStages() const {
    std::cout << "Pipeline (" << stages_.size() << " stage"
              << (stages_.size() != 1 ? "s" : "") << "):\n";
    for (size_t i = 0; i < stages_.size(); ++i)
        std::cout << "  [" << (i + 1) << "] " << stages_[i]->name() << "\n";
}
