#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include "pipeline.h"
#include "postprocess.h"
#include "stb_image.h"  
#include "mem_info.h"

static void report(const char* name, std::vector<double> v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    std::cout << name << " | mean " << mean << " | p50 " << v[v.size() / 2]
              << " | p95 " << v[static_cast<size_t>(v.size() * 0.95)] << " ms\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: app <model.pte> <image> [threads=4] "
                     "[threshold=9.4901] [iterations=1] [heatmap.png]\n";
        return 1;
    }
    PipelineSettings settings;
    settings.model_path = argv[1];
    const char* image_path = argv[2];
    if (argc > 3) settings.thread_count = std::atoi(argv[3]);
    if (argc > 4) settings.threshold = std::strtof(argv[4], nullptr);
    int iterations = argc > 5 ? std::max(1, std::atoi(argv[5])) : 1;
    const char* heatmap_path = argc > 6 ? argv[6] : nullptr;

    int w = 0, h = 0, c = 0;
    std::unique_ptr<uint8_t, void (*)(void*)> img(
        stbi_load(image_path, &w, &h, &c, 4), stbi_image_free);
    if (!img) {
        std::cerr << "could not load image: " << image_path << "\n";
        return 1;
    }

    PatchCorePipeline pipeline(settings);
    if (!pipeline.init()) {
        std::cerr << "pipeline init failed\n";
        return 1;
    }

    const bool want_heatmap = (heatmap_path != nullptr);
    PipelineResult first;
    std::vector<double> t_pre, t_inf, t_post, t_total;

    for (int i = 0; i < iterations; ++i) {
        PipelineResult r = pipeline.inspect(img.get(), w, h, w * 4, want_heatmap);
        if (!r.ok) {
            std::cerr << "inspect failed on iteration " << i << "\n";
            return 1;
        }
        t_pre.push_back(r.time_preprocess_ms);
        t_inf.push_back(r.time_inference_ms);
        t_post.push_back(r.time_postprocess_ms);
        t_total.push_back(r.time_total_ms);
        if (i == 0) first = std::move(r);
    }

    std::cout << "score " << first.score << " | threshold " << settings.threshold
              << " | margin " << first.margin << " | "
              << (first.is_defect ? "DEFECT" : "OK") << "\n";

    if (want_heatmap) {
        if (first.heatmap_rgb.empty() ||
            !save_heatmap(heatmap_path, first.heatmap_rgb,
                          first.heatmap_width, first.heatmap_height))
            std::cerr << "heatmap save failed\n";
    }

    report("preprocess ", t_pre);
    report("inference  ", t_inf);
    report("postprocess", t_post);
    report("total      ", t_total);
    double mean_total = std::accumulate(t_total.begin(), t_total.end(), 0.0) / t_total.size();
    std::cout << "pipeline FPS: " << 1000.0 / mean_total << "\n";
    print_mem("Main pipeline execution");
    return 0;
}