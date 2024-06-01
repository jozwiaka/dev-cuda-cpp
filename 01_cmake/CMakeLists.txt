cmake_minimum_required(VERSION 3.20)
project(MyCudaProject CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(exe main.cu)
target_link_libraries(exe ${CUDA_LIBRARIES})
