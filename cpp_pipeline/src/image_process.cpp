
#include "image_process.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace executorch::extension;
using namespace ::executorch::extension::image;

PreProcessor::PreProcessor(int target_width, int target_height) 
    : processor({.target_width = target_width, .target_height = target_height}) {
}

TensorPtr PreProcessor::img_pixel_to_tensor(const uint8_t* pixel_buffer,int width, int height, int stride_bytes) {
    if (pixel_buffer == nullptr || width <= 0 || height <= 0 ||
        stride_bytes < width * 4) {
        std::cerr << "ERROR: invalid pixel buffer or dimensions\n";
        return nullptr;
    }

    auto result = processor.process(
        pixel_buffer, 
        width, 
        height, 
        stride_bytes, 
        ColorFormat::RGBA
    );

    if (!result.ok()) {
        std::cerr << "ERROR: image processor failed\n";
        return nullptr;
    }
    return result.get();
}

TensorPtr PreProcessor::img_to_tensor(const char* file_path){

    int original_height, original_width, original_channels;
    
    uint8_t* raw_rgba_buffer = stbi_load(
        file_path,
        &original_width,
        &original_height,
        &original_channels,
        4
    );
    if(raw_rgba_buffer == nullptr){
        std::cerr<<"raw rgba buffer loading failed"<<"\n";
        return nullptr;
    }

    auto result = img_pixel_to_tensor(raw_rgba_buffer, original_width, original_height, original_width * 4);
    stbi_image_free(raw_rgba_buffer);

    return result;
}


