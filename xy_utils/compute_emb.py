import torch
import random
import threading

import time
import random

# Function to continuously occupy memory and compute on a specific GPU
def occupy_gpu(gpu_index):
    # Set the current device to the specific GPU
    torch.cuda.set_device(gpu_index)
    
    # Generate a random size multiplier between 10000 and 20000
    size = random.randint(40000, 50000)
    
    # Create a large tensor of random floats, utilizing significant memory
    large_tensor = torch.randn(size, size, device='cuda')
    
    # Run an infinite loop to keep the GPU busy
    while True:
        # Perform a simple computation repeatedly
        time.sleep(random.uniform(0, 1))
        large_tensor = large_tensor * torch.tensor(1.1, device='cuda')  # Slightly alter the tensor to ensure active computation

# List of GPU indices for your four RTX A5000 GPUs
gpu_indices = [1,2,4,7]  # Adjust based on your actual GPU indices from the screenshot

# Loop through the GPUs and occupy each
for index in gpu_indices:
    # Start the function in a separate thread to ensure all GPUs are used concurrently
    thread = threading.Thread(target=occupy_gpu, args=(index,))
    thread.start()

print("Started continuous computation on GPUs:", gpu_indices)
