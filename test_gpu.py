import torch
print(torch.cuda.is_available())
tensor = torch.tensor((), dtype=torch.int32)
tensor = tensor.new_ones((2, 3))
tensor = tensor.to('cuda')