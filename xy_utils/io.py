

def resolution_minmax(current_resolution, max_resolution):
    current_width, current_height = current_resolution
    max_width, max_height = max_resolution
    
    # Calculate aspect ratio of the current resolution
    aspect_ratio = current_width / current_height
    
    # Calculate potential new dimensions
    new_width = min(current_width, max_width)
    new_height = min(current_height, max_height)
    
    # Adjust dimensions to maintain aspect ratio
    if new_width / new_height > aspect_ratio:
        new_width = int(new_height * aspect_ratio)
    else:
        new_height = int(new_width / aspect_ratio)
    
    return (new_width, new_height)

def dict_emb_for_save(weight_dict):
    """
    Convert a dictionary of embeddings to a dictionary of lists for saving.
    Excludes items where the value is None.
    """
    return {key: value.detach().cpu() for key, value in weight_dict.items() if value is not None}    