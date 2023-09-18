from torch import nn
import torch

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(f"==>> input: {input}")
target = torch.empty(3, dtype=torch.long).random_(5)
print(f"==>> target: {target}")
output = loss(input, target)
output.backward()
