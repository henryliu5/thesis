#include <torch/extension.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <iostream>

int add(int i, int j) {
    return i + j;
}

torch::Tensor rand_tensor(){
    torch::Tensor tensor = torch::rand({2, 3});
    return tensor;
}

torch::Tensor incr(torch::Tensor t){
    t += 10;
    return t;
}

void print_neg(torch::Tensor t){
    std::cout << (-1 * t) << std::endl;
}

PYBIND11_MODULE(fast_cpp, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
    m.def("rand_tensor", &rand_tensor, "A function that adds two numbers");
    m.def("incr", &incr, "Increment a tensor");
    m.def("print_neg", &print_neg, "Increment a tensor");
}