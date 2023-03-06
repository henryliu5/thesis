cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Fast_Inference)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
pybind11_add_module(fast_cpp src/pybind_test.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(fast_cpp PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET fast_cpp PROPERTY CXX_STANDARD 17)