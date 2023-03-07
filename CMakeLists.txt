cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fast_inference)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package (Threads)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_definitions(-g -pthread)

pybind11_add_module(fast_inference_cpp src/cache_manager.cpp)
target_link_libraries(fast_inference_cpp PRIVATE "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET fast_inference_cpp PROPERTY CXX_STANDARD 17)

pybind11_add_module(fast_cpp_test src/pybind_test.cpp)
target_link_libraries(fast_cpp_test PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET fast_cpp_test PROPERTY CXX_STANDARD 17)

add_executable(cpp_test src/cpp_test.cpp)