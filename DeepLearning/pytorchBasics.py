import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device.type , device.index)

if torch.cuda.is_available():
    # Get the name of your card (e.g., "NVIDIA GeForce RTX 4070")
    gpu_name = torch.cuda.get_device_name(0)
    
    # Get total memory properties
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1e9  # Convert bytes to GB
    
    # Check current memory usage
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    
    print(f"GPU Name: {gpu_name}")
    print(f"Total VRAM: {total_memory:.2f} GB")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
else:
    print("CUDA is not available.")


