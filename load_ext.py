import torch
import pybind_example

print(pybind_example.rand_tensor())
print(torch.cuda.get_device_properties(torch.device('cuda')))

t = torch.tensor([1, 2, 3, 4])
print(t)
t = pybind_example.incr(t)
print(t)
pybind_example.print_neg(t)