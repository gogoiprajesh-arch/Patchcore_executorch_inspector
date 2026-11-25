#pragma once
#include <iostream>
#include <cstdint>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/image/image_processor.h>



class PreProcessor {
private:
    executorch::extension::image::ImageProcessor processor;

public:
    PreProcessor(int target_width, int target_height);
    
    executorch::extension::TensorPtr img_pixel_to_tensor(const uint8_t* pixel_buffer,int width, int height, int stride_bytes);
    executorch::extension::TensorPtr img_to_tensor(const char* file_path);
    
};