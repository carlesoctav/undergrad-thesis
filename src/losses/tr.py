import torch.nn as nn
import torch
logits = torch.randn(2, 3)  # Example logits with 2 samples and 3 classes
labels = torch.tensor([1, 2])  # Example labels for the 2 samples

criterion = nn.CrossEntropyLoss()  # Initialize cross-entropy loss module
loss = criterion(logits, labels)  # Compute cross-entropy loss

print(loss)  # Print scalar value of loss