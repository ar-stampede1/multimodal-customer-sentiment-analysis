import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

result = torch.matmul(x, y)

print("Matrix multiplication successful!")
print("Result shape:", result.shape)
print("GPU:", torch.cuda.get_device_name(0))

print("GPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**2, 2), "MB")