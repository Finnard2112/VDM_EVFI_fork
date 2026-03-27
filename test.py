import torch

# Check if a GPU is available
print(f"Is CUDA available? {torch.cuda.is_available()}")

# Check the number of GPUs
print(f"Number of GPUs: {torch.cuda.device_count()}")
