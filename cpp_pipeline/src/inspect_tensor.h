#pragma once
#include <iostream>
#include <algorithm>
#include <numeric>
#include <executorch/extension/tensor/tensor.h>

using executorch::aten::ScalarType;
using executorch::extension::TensorPtr;

inline bool inspect_tensor(const TensorPtr& t, const char* name) {
    if (!t) {
        std::cerr << "[" << name << "] error: TensorPtr is null\n";
        return false;
    }

    if (t->scalar_type() != ScalarType::Float) {
        std::cerr << "[" << name << "] error: dtype is not float32\n";
        return false;
    }
    std::cout << "[" << name << "] dims: " << t->dim() << " | shape: [";
    for (int i = 0; i < t->dim(); ++i) {
        std::cout << t->size(i) << (i + 1 < t->dim() ? ", " : "");
    }
    std::cout << "]\n";

    const float* ptr = t->const_data_ptr<float>();
    size_t count = t->numel();
    if (count == 0 || !ptr) {
        std::cerr << "[" << name << "] error: empty tensor or null data ptr\n";
        return false;
    }

    auto [min_it, max_it] = std::minmax_element(ptr, ptr + count);

    double sum = std::accumulate(ptr, ptr + count, 0.0);
    double mean = sum / static_cast<double>(count);

    std::cout << "[" << name << "] numel: " << count
              << " | min: " << *min_it
              << " | max: " << *max_it
              << " | mean: " << mean << "\n";

    return true;
}