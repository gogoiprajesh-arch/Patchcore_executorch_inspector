#include <iostream>
#include <string>
#include <chrono>    
#include <cmath> 
#include <algorithm>  
#include <thread>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include "model.h"
#include <executorch/extension/threadpool/threadpool.h>



using executorch::aten::ScalarType;
using executorch::runtime::Error;
using namespace executorch::extension;

Model::Model(int threads, const char* model_file):
        num_threads(threads),
        module(model_file, Module::LoadMode::Mmap) {};

bool Model::init(){
    auto* pool = ::executorch::extension::threadpool::get_threadpool();
    pool->_unsafe_reset_threadpool(num_threads);
    std::cout << "threads: " << pool->get_thread_count() << "\n";
    
    auto err = module.load_method("forward");
    if (err != Error::Ok) {
        std::cerr << "load_method failed, code " << (int)err << "\n";
        return false;
    }

    std::vector<float> dummy_data(1 * 3 * 256 * 256, 0.5f);
    auto warmup_input = from_blob(dummy_data.data(), {1, 3, 256, 256});

    for(int i = 0 ; i<10; i++){
        auto res = module.forward(warmup_input);
        if(!res.ok()){
            std::cerr<<"model failed initiaslisation"<<(int)res.error()<<"\n";
            return false;
        }
    }
    return true;
};

PatchCoreOutput Model::run(const TensorPtr& input){
    PatchCoreOutput out;
    auto res = module.forward(input);
    if(!res.ok()){
        std::cerr << "model.forward failed, code " << (int)res.error() << "\n";
        return out;
    }
    if (res->size() < 2) {
        std::cerr << "check failed: expected at least 2 outputs, got " << res->size() << "\n";
        return out;
    }

    if (!res->at(0).isTensor() || !res->at(1).isTensor()) {
        std::cerr << "check failed: output entries are not both tensors\n";
        return out;
    }

    const auto& score_tensor = res->at(0).toTensor();
    const auto& map_tensor = res->at(1).toTensor();

    if (score_tensor.scalar_type() != ScalarType::Float || map_tensor.scalar_type() != ScalarType::Float || map_tensor.size(0) != 1 || map_tensor.size(1) != 1) {
        std::cerr << "check failed: output tensors are not float32\n";
        return out;
    }

    if (score_tensor.numel() != 1) {
        std::cerr << "check failed: score_tensor numel is " << score_tensor.numel() << ", expected 1\n";
        return out;
    }

    if (map_tensor.dim() != 4 || map_tensor.size(2) != 256 || map_tensor.size(3) != 256) {
        std::cerr << "check failed: map_tensor shape is not 4D with 256x256 spatial dims\n";
        return out;
    }

    out.height = static_cast<int>(map_tensor.size(2));
    out.width  = static_cast<int>(map_tensor.size(3));


    out.pred_score = score_tensor.const_data_ptr<float>()[0];

    if (!std::isfinite(out.pred_score)) {
        std::cerr << "check failed: score is NaN or inf\n";
        return PatchCoreOutput{};
    }

    
    const float* map_ptr = map_tensor.const_data_ptr<float>();
    size_t num_elements = map_tensor.numel();
    out.anomaly_map.assign(map_ptr, map_ptr + num_elements);

    out.ok = true;
    return out;
};
