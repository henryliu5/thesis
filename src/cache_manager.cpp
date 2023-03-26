#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "cache_manager.hpp"

// This mechanism exists since I only want to include pybind11 in this file due to
// how pybind is added to CMake targets (pybind_add_module), which is not ideal
// for the cpp_test.cpp
void CacheManager::gilRelease(std::function<void()> f) {
    // std::cout << "releasing gil" << std::endl;
    py::gil_scoped_release release;
    f();
}

PYBIND11_MODULE(fast_inference_cpp, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<CacheManager>(m, "CacheManager")
        .def(py::init<const int, const int, const int, const int, bool>())
        .def("set_cache", &CacheManager::setCache)
        .def("wait_for_queue", &CacheManager::waitForQueue)
        .def("get_counts", &CacheManager::getCounts)
        .def("thread_enter", &CacheManager::threadEnter)
        .def("thread_exit", &CacheManager::threadExit)
        // .def("receive_new_features", &CacheManager::receiveNewFeatures)
        .def("set_cache_candidates", &CacheManager::setCacheCandidates)
        .def("place_feats_in_queue", &CacheManager::placeFeatsInQueue)
        .def("lock", &CacheManager::lock)
        .def("unlock", &CacheManager::unlock)
        .def("read_lock", &CacheManager::readLock)
        .def("read_unlock", &CacheManager::readUnlock);
}
