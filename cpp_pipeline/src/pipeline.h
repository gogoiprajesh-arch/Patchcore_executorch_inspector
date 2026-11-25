#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "image_process.h"
#include "model.h"

struct PipelineSettings {
    std::string model_path;
    int   thread_count = 4;
    float threshold    = 9.4901f;   // provisional, comes from make_reference.py
    int   input_size   = 256;

    bool valid() const {
        return !model_path.empty() && threshold > 0.0f &&
               thread_count > 0 && input_size > 0;
    }
};

struct PipelineResult {
    bool   ok        = false;  
    bool   is_defect = true;   
    float  score     = 0.0f;
    float  margin    = 0.0f;   // score - threshold
    double time_preprocess_ms  = 0.0;
    double time_inference_ms   = 0.0;
    double time_postprocess_ms = 0.0;
    double time_total_ms       = 0.0;
    std::vector<uint8_t> heatmap_rgb;  
    int heatmap_width  = 0;
    int heatmap_height = 0;
};

class PatchCorePipeline {
public:
    explicit PatchCorePipeline(const PipelineSettings& settings);
    bool init();
    PipelineResult inspect(const uint8_t* pixels, int width, int height,
                           int stride, bool want_heatmap);

private:
    PipelineSettings       m_settings;    
    std::unique_ptr<Model> m_model;         
    PreProcessor           m_pre;
    bool                   m_initialized = false;
};