import torch.distributed as dist

def is_main_process():
    """
    Returns True if the current process is the main process (rank 0), 
    and False otherwise.
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True  # If not using distributed, assume it's the main process