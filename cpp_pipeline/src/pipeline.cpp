#include "pipeline.h"
#include "postprocess.h"
#include <chrono>
#include <iostream>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
}

PatchCorePipeline::PatchCorePipeline(const PipelineSettings& settings)
    : m_settings(settings),
      m_model(std::make_unique<Model>(settings.thread_count,
                                      settings.model_path.c_str())),
      m_pre(settings.input_size, settings.input_size) {}

bool PatchCorePipeline::init() {
    if (!m_settings.valid()) {
        std::cerr << "[Pipeline] Invalid settings.\n";
        return false;
    }
    m_initialized = m_model->init();
    return m_initialized;
}

PipelineResult PatchCorePipeline::inspect(const uint8_t* pixels, int width,
                                          int height, int stride,
                                          bool want_heatmap) {
    PipelineResult res;
    auto t_total = Clock::now();

    if (!m_initialized) {
        std::cerr << "[Pipeline] Refused: not initialized.\n";
        return res;
    }
    if (!pixels || width <= 0 || height <= 0 || stride < width * 4) {
        std::cerr << "[Pipeline] Refused: bad pixel buffer or dimensions.\n";
        return res;
    }

    // 1. preprocess
    auto t0 = Clock::now();
    executorch::extension::TensorPtr input =
        m_pre.img_pixel_to_tensor(pixels, width, height, stride);
    res.time_preprocess_ms = elapsed_ms(t0, Clock::now());
    if (!input) {
        std::cerr << "[Pipeline] Preprocess failed.\n";
        return res;
    }

    // 2. inference
    t0 = Clock::now();
    PatchCoreOutput out = m_model->run(input);
    res.time_inference_ms = elapsed_ms(t0, Clock::now());
    if (!out.ok) {
        std::cerr << "[Pipeline] Inference failed.\n";
        return res;
    }

    // 3. decision + optional heatmap
    t0 = Clock::now();
    res.score     = out.pred_score;
    res.margin    = out.pred_score - m_settings.threshold;
    res.is_defect = !is_good(out.pred_score, m_settings.threshold);

    if (want_heatmap) {
        res.heatmap_rgb = heatmap(out.anomaly_map, out.width, out.height,
                                  m_settings.threshold);
        if (res.heatmap_rgb.empty()) {
            std::cerr << "[Pipeline] Warning: heatmap failed.\n";
        } else {
            res.heatmap_width  = out.width;
            res.heatmap_height = out.height;
        }
    }
    res.time_postprocess_ms = elapsed_ms(t0, Clock::now());

    res.time_total_ms = elapsed_ms(t_total, Clock::now());
    res.ok = true;
    return res;
}