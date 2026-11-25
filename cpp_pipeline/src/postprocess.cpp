#include <iostream>
#include "postprocess.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



RGB apply_colormap(float v) {
    if (v <= 0.0f) return {0, 0, 255};       // Blue
    if (v >= 1.0f) return {255, 0, 0};       // Red
    
    if (v < 0.333f) {
        // Blue to Green
        float t = v / 0.333f;
        return {0, static_cast<uint8_t>(255 * t), static_cast<uint8_t>(255 * (1.0f - t))};
    } else if (v < 0.666f) {
        // Green to Yellow
        float t = (v - 0.333f) / 0.333f;
        return {static_cast<uint8_t>(255 * t), 255, 0};
    } else {
        // Yellow to Red
        float t = (v - 0.666f) / 0.334f;
        return {255, static_cast<uint8_t>(255 * (1.0f - t)), 0};
    }
}

std::vector<float> normalize_map(const std::vector<float>& anomaly_map, float threshold) {
    float max_range = 2.0f * threshold;
    std::vector<float> norm_map;
    norm_map.reserve(anomaly_map.size());

    for (float val : anomaly_map) {
        val = val / max_range;
        val = std::clamp(val, 0.0f, 1.0f); // Clamp to 0..1
        norm_map.push_back(val);
    }
    return norm_map;
}

bool is_good(float pred_score, float threshold){
    return (pred_score < threshold); //Return true for good images
}

std::vector<uint8_t> heatmap(const std::vector<float>& anomaly_map, int width, int height, float threshold){
    if (width <= 0 || height <= 0 || threshold <= 0.0f ||
        anomaly_map.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
        std::cerr << "ERROR : invalid heatmap input\n";
        return {};
    }
    std::vector<float> norm_map = normalize_map(anomaly_map, threshold);
    
    
    std::vector<uint8_t> rgb_buffer(width * height * 3);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            RGB color = apply_colormap(norm_map[idx]);
            
            rgb_buffer[idx * 3 + 0] = color.r;
            rgb_buffer[idx * 3 + 1] = color.g;
            rgb_buffer[idx * 3 + 2] = color.b;
        }
    }

    return rgb_buffer;
}

bool save_heatmap(const char* output_path, std::vector<uint8_t>& rgb_buffer, int width, int height){
    int stride_in_bytes = width * 3;
    bool write = stbi_write_png(output_path, width, height, 3, rgb_buffer.data(), stride_in_bytes);  //returns 0 when failed
    return write;
}



