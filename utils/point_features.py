# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.nn import functional as F

"""
Shape shorthand in this module:

    N: minibatch dimension size, i.e. the number of RoIs for instance segmenation or the
        number of images for semantic segmenation.
    R: number of ROIs, combined over all images, in the minibatch
    P: number of points
"""

def coords_sample(input, num_points, **kwargs):
    h, w, c = input.shape
    device = input.device
    dtype = input.dtype
    
    # Generate random indices on the same device as input
    indices = torch.randint(0, h * w, (num_points,), device=device)
    
    # Convert to 2D coordinates
    y = indices // w
    x = indices % w
    
    # Normalize coordinates to [0, 1] using the input's dtype
    y_norm = y.to(dtype) / (h - 1)
    x_norm = x.to(dtype) / (w - 1)
    
    # Combine normalized coordinates and channel values
    result = torch.cat([y_norm.unsqueeze(1), x_norm.unsqueeze(1)], dim=1)
    return result

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output