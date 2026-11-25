cmake_minimum_required(VERSION 3.24)
project(patchcore_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ET_SRC_DIR "/home/prajesh_gogoi/executorch")
set(ET_BUILD_DIR "/home/prajesh_gogoi/executorch/cmake-out-android")

# 1. Header include paths
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${ET_SRC_DIR}/.."
    "${ET_SRC_DIR}/runtime/core/portable_type/c10"
    "${ET_BUILD_DIR}/runtime/core/portable_type/c10"
    "${ET_BUILD_DIR}/include"
    "${ET_BUILD_DIR}/include/executorch/runtime/core/portable_type/c10"
    "${ET_BUILD_DIR}/schema/include"
    "${ET_SRC_DIR}/third-party/flatbuffers/include"
)


# 2. Library search paths 
link_directories(
    "${ET_BUILD_DIR}/lib"
    "${ET_BUILD_DIR}/backends/xnnpack/third-party/cpuinfo"
    "${ET_BUILD_DIR}/third-party/gflags"
)

add_executable(app 
    src/main.cpp 
    src/model.cpp 
    src/image_process.cpp
    src/postprocess.cpp
    src/pipeline.cpp
    src/mem_info.cpp
    "${ET_SRC_DIR}/extension/image/image_processor.cpp"
    "${ET_SRC_DIR}/extension/image/image_processor_common.cpp"
    "${ET_SRC_DIR}/extension/image/image_processor_simd.cpp"
)

                    
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)



# 3. Link required archives
target_link_libraries(app PRIVATE
    -Wl,--whole-archive
    xnnpack_backend
    optimized_native_cpu_ops_lib
    -Wl,--no-whole-archive
    -Wl,--start-group
    extension_module_static
    extension_runner_util
    extension_evalue_util
    extension_tensor
    extension_data_loader
    extension_flat_tensor
    extension_named_data_map
    extension_threadpool
    executorch
    executorch_core
    optimized_kernels
    portable_kernels
    optimized_portable_kernels
    kernels_util_all_deps
    cpublas
    eigen_blas
    XNNPACK
    xnnpack-microkernels-prod
    kleidiai
    pthreadpool
    cpuinfo
    cpuinfo_internals
    flatccrt
    gflags_nothreads
    -Wl,--end-group

    # Android system logging library
    log
)