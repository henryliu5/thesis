import torch
import fast_cpp

print(fast_cpp.rand_tensor())
print(torch.cuda.get_device_properties(torch.device('cuda')))

t = torch.tensor([1, 2, 3, 4])
print(t)
t = fast_cpp.incr(t)
print(t)
fast_cpp.print_neg(t)