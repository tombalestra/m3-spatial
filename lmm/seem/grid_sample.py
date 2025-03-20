import torch
import math
import cv2
import numpy as np
import os

def create_circular_grid_masks(height, width, dot_spacing=50, dot_radius=10):
    """
    Create a grid mask tensor with individual circular dots at regular intervals,
    spanning the full width of the image including near the margins.
    
    Args:
    height (int): Height of the image.
    width (int): Width of the image.
    dot_spacing (int): Spacing between dot centers.
    dot_radius (int): Radius of each dot.
    
    Returns:
    torch.Tensor: A boolean tensor of shape [n, h, w] where n is the number of dots.
    """
    # Calculate the number of dots in each dimension
    n_rows = math.ceil(height / dot_spacing) + 1
    n_cols = math.ceil(width / dot_spacing) + 1

    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

    # Initialize the output tensor
    masks = torch.zeros((n_rows * n_cols, height, width), dtype=torch.bool)

    # Generate individual masks for each dot
    for i in range(n_rows):
        for j in range(n_cols):
            dot_index = i * n_cols + j
            center_y = i * dot_spacing
            center_x = j * dot_spacing
            
            # Calculate distances from this dot center
            distances = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
            
            # Create the mask for this dot: True where distance <= radius
            dot_mask = distances <= dot_radius
            masks[dot_index] = dot_mask

    return masks

def save_mask_images(masks, output_dir='output_masks'):
    """
    Save individual dot masks and combined mask as images using OpenCV.
    
    Args:
    masks (torch.Tensor): The 3D tensor of masks.
    output_dir (str): Directory to save the output images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual dot masks
    for i in range(min(4, masks.shape[0])):  # Save first 4 dot masks
        mask_np = masks[i].numpy().astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, f'dot_mask_{i+1}.png'), mask_np)
    
    # Save combined mask
    combined_mask = masks.any(dim=0).numpy().astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, 'combined_mask.png'), combined_mask)

# Example usage
# h, w = 300, 400  # Adjust these values to match your desired dimensions
# dot_spacing = 50  # Distance between dot centers
# dot_radius = 10   # Radius of each dot

# grid_masks = create_circular_grid_masks(h, w, dot_spacing, dot_radius)

# print(f"Mask shape: {grid_masks.shape}")
# print(f"Number of dots: {grid_masks.shape[0]}")

# # Save mask images
# save_mask_images(grid_masks)

# print("Mask images have been saved in the 'output_masks' directory.")