cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fast_inference)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost COMPONENTS thread system REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

include_directories(include ${Boost_INCLUDE_DIR})

pybind11_add_module(fast_inference_cpp src/fast_inference.cpp src/cache_manager.cpp)
target_link_libraries(fast_inference_cpp PRIVATE "${TORCH_LIBRARIES}" ${Boost_LIBRARIES})
set_property(TARGET fast_inference_cpp PROPERTY CXX_STANDARD 17)

pybind11_add_module(pybind_example src/pybind_test.cpp)
target_link_libraries(pybind_example PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET pybind_example PROPERTY CXX_STANDARD 17)

# add_executable(cpp_test src/cpp_test.cpp)
# target_link_libraries(cpp_test PRIVATE "${TORCH_LIBRARIES}" ${Boost_LIBRARIES})
# set_property(TARGET cpp_test PROPERTY CXX_STANDARD 17)

add_executable(ipc_test src/ipc_test.cpp)
target_link_libraries(ipc_test PRIVATE "${TORCH_LIBRARIES}" ${Boost_LIBRARIES})
set_property(TARGET ipc_test PROPERTY CXX_STANDARD 17)