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

void init_manager(py::module &m){
    py::class_<CacheManager>(m, "CacheManager")
        .def(py::init<const int, const int, const int, const int, bool, bool, const int, const int, const int>())
        .def("set_cache", &CacheManager::setCache)
        .def("wait_for_queue", &CacheManager::waitForQueue)
        .def("thread_enter", &CacheManager::threadEnter)
        .def("thread_exit", &CacheManager::threadExit)
        // .def("receive_new_features", &CacheManager::receiveNewFeatures)
        .def("set_cache_candidates", &CacheManager::setCacheCandidates)
        .def("place_feats_in_queue", &CacheManager::placeFeatsInQueue)
        .def("read_lock", &CacheManager::readLock)
        .def("read_unlock", &CacheManager::readUnlock)
        .def("write_lock", &CacheManager::writeLock)
        .def("write_unlock", &CacheManager::writeUnlock);
}
