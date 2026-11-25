#pragma once
#include <vector>
#include <algorithm>
#include <cstdint>

struct RGB { uint8_t r, g, b; };

RGB apply_colormap(float v);

std::vector<float> normalize_map(const std::vector<float>& anomaly_map, float threshold);

bool is_good(float pred_score, float threshold);

std::vector<uint8_t> heatmap(const std::vector<float>& anomaly_map, int width, int height, float threshold);

bool save_heatmap(const char* output_path, std::vector<uint8_t>& rgb_buffer, int width, int height);
