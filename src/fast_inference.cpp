#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "shm.hpp"

void init_manager(py::module &m);

PYBIND11_MODULE(fast_inference_cpp, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("shm_setup", &shmSetup, "Created shared memory resources");

    py::class_<SHMLocks>(m, "SHMLocks")
        .def(py::init<>())
        .def("read_lock", &SHMLocks::readLock)
        .def("read_unlock", &SHMLocks::readUnlock)
        .def("write_lock", &SHMLocks::writeLock)
        .def("write_unlock", &SHMLocks::writeUnlock);

    init_manager(m);
}
