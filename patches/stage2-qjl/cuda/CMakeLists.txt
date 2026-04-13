cmake_minimum_required(VERSION 3.18)
project(tqp_cuda CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89;120")
endif()

add_library(ggml_tq_paper_cuda SHARED
    tqp-quantize.cu
    tqp-prepare-query.cu
    tqp-vec-dot.cu
    tqp-dequantize.cu
)

target_include_directories(ggml_tq_paper_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../c
)

set_target_properties(ggml_tq_paper_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME ggml_tq_paper_cuda
)
