#pragma once
#include <vector>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/module/module.h>


using namespace executorch::extension;

struct PatchCoreOutput{
    bool ok = false;
    float pred_score = 0.0f;
    int height = 0;
    int width = 0;
    std::vector<float> anomaly_map;
};

class Model{
private:
    int num_threads;
    Module module;

public:
    Model(int threads, const char* model_file);
    bool init();
    PatchCoreOutput run(const TensorPtr& input);
};