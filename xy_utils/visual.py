import torch
import torch.nn.functional as F
import numpy as np
import cv2

from lmm.dataset.utils import add_image_marker
from .memory import index_to_raw

def compute_global_pca(embedding, pre_normalized=False):
    """
    Compute a global PCA transformation from an embedding tensor.
    
    Args:
        embedding (Tensor): A tensor of shape (C, H, W).
        pre_normalized (bool): If True, assume the embedding is already normalized
            (i.e. each spatial vector has unit norm). Otherwise, normalize over the
            channel dimension (to avoid division by zero issues).
    
    Returns:
        pca_components (Tensor): PCA components of shape (C, 3).
        pca_min (Tensor): Minimum values along each PCA dimension (shape [3]).
        pca_max (Tensor): Maximum values along each PCA dimension (shape [3]).
    """
    if not pre_normalized:
        embedding = embedding / embedding.norm(dim=0, keepdim=True)
    C, H, W = embedding.shape
    reshaped = embedding.view(C, -1).t()  # shape: (H*W, C)
    
    # Compute PCA to reduce the channel dimension to 3 components.
    # torch.pca_lowrank returns U, S, V such that the projection is given by V.
    U, S, V = torch.pca_lowrank(reshaped, q=3)
    pca_components = V[:, :3]  # shape: (C, 3)
    
    # Project data using the PCA components
    pca_features = torch.matmul(reshaped, pca_components)  # shape: (H*W, 3)
    
    # Compute global min and max for normalization
    pca_min = pca_features.min(dim=0)[0]
    pca_max = pca_features.max(dim=0)[0]

    return {'pca_components': pca_components.cpu(), 'pca_min': pca_min.cpu(), 'pca_max': pca_max.cpu()}    

def pca_to_rgb(feature_tensor, pca_components=None, pca_min=None, pca_max=None, pre_normalized=True):
    """
    Convert an embedding tensor to an RGB image using global PCA parameters.
    If PCA parameters are not provided, they are computed from the feature_tensor.
    
    Args:
        feature_tensor (Tensor): Embedding tensor of shape (C, H, W).
        pca_components (Tensor, optional): Pre-computed PCA components, shape (C, 3).
        pca_min (Tensor, optional): Global minimum values from the PCA projection (shape [3]).
        pca_max (Tensor, optional): Global maximum values from the PCA projection (shape [3]).
        pre_normalized (bool): Indicates if feature_tensor is already normalized (each vector has
            unit norm). Set this to True if the embedding has been normalized before invoking
            this function (as in your pipeline).
    
    Returns:
        np.ndarray: An RGB image with values in 0-255 (dtype uint8).
    """
    # If PCA parameters are not provided, compute them.
    if pca_components is None or pca_min is None or pca_max is None:
        pca_params = compute_global_pca(feature_tensor, pre_normalized=pre_normalized)
        pca_components = pca_params['pca_components']
        pca_min = pca_params['pca_min']
        pca_max = pca_params['pca_max']

    C, H, W = feature_tensor.shape
    reshaped = feature_tensor.view(C, -1).t()  # shape: (H*W, C)
    pca_features = torch.matmul(reshaped, pca_components)  # shape: (H*W, 3)
    
    # Normalize the PCA projections using the pre-computed global min and max
    range_vals = pca_max - pca_min
    range_vals[range_vals == 0] = 1.0  # Avoid division by zero
    normalized = (pca_features - pca_min) / range_vals
    # Clamp to [0,1] to be safe before scaling
    normalized = torch.clamp(normalized, 0, 1)

    # Reshape back to image dimensions and scale to 0-255.
    pca_image = normalized.view(H, W, 3)
    pca_image_np = (pca_image.detach().cpu().numpy() * 255).astype(np.uint8)
    
    return pca_image_np

def visualize_combined(args, gt_image, image, embedding, emb_proj, emb_mem, output_filename="combined_visualization.png"):
    # Ensure all inputs are numpy arrays
    gt_image_np = gt_image.cpu().numpy() if isinstance(gt_image, torch.Tensor) else gt_image
    image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
    
    # post processing embedding
    c,h,w = embedding.shape

    # Use interpolate to downsample with bilinear mode
    ratio = 4
    embedding = F.interpolate(embedding[None,], size=(h//ratio, w//ratio), mode='bilinear', align_corners=False)[0]
    embedding = index_to_raw(embedding, emb_proj, emb_mem, _temp=args.softmax_temp).float()
    embedding = F.interpolate(embedding.permute(2,0,1)[None,], size=(h, w), mode='bilinear', align_corners=False)[0]    
    embedding = embedding / embedding.norm(dim=0, keepdim=True)

    # Convert embedding to RGB using PCA
    try:
        embedding_rgb = pca_to_rgb(embedding.cpu())
    except:
        embedding_rgb = np.zeros((h, w, 3), dtype=np.uint8) + 255
    
    # Ensure all images are in the range [0, 255] and correct data type
    gt_image_np = (gt_image_np * 255).astype(np.uint8)
    image_np = (image_np * 255).astype(np.uint8)
    
    # Ensure all images have shape (H, W, C)
    if gt_image_np.shape[0] == 3:
        gt_image_np = np.transpose(gt_image_np, (1, 2, 0))
    if image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Create a blank canvas to hold all images side by side
    h, w = gt_image_np.shape[:2]
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # Place images side by side
    combined[:, :w] = gt_image_np
    combined[:, w:2*w] = image_np
    combined[:, 2*w:] = embedding_rgb
    
    # Save the combined image
    cv2.imwrite(output_filename, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

def visualize_combined_all(args, gt_image, image, embeddings, emb_projs, emb_mems, output_filename="combined_visualization.png"):
    # Ensure all inputs are numpy arrays
    gt_image_np = gt_image.cpu().numpy() if isinstance(gt_image, torch.Tensor) else gt_image
    image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
    down_ratio = 4
    
    # post processing embedding
    c,h,w = embeddings.shape
    gt_image_np = (gt_image_np * 255).astype(np.uint8)
    image_np = (image_np * 255).astype(np.uint8)

    # Ensure all images have shape (H, W, C)
    if gt_image_np.shape[0] == 3:
        gt_image_np = np.transpose(gt_image_np, (1, 2, 0))
    if image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))

    gt_image_np = cv2.resize(gt_image_np, (w//down_ratio, h//down_ratio), interpolation=cv2.INTER_LINEAR)
    image_np = cv2.resize(image_np, (w//down_ratio, h//down_ratio), interpolation=cv2.INTER_LINEAR)
    _h, _w = gt_image_np.shape[:2]

    models = {
        'clip': (args.use_clip, args.clip_bit),
        'llama3': (args.use_llama3, args.llama3_bit),
        'siglip': (args.use_siglip, args.siglip_bit),
        'dinov2': (args.use_dinov2, args.dinov2_bit),
        'seem': (args.use_seem, args.seem_bit),
        'llamav': (args.use_llamav, args.llamav_bit),
    }

    emb_rgbs = []
    emb_names = []
    for model, (use_model, bit_range) in models.items():
        if use_model:
            # Extract embedding for the current model
            embedding = embeddings[bit_range[0]:bit_range[1], :, :]
            
            # Downsample the embedding
            embedding = F.interpolate(embedding[None,], size=(h//down_ratio, w//down_ratio), 
                                      mode='bilinear', align_corners=False)[0]
            
            # Convert indices to raw embeddings
            embedding = index_to_raw(embedding, emb_projs[model], emb_mems[model], _temp=args.softmax_temp).float()
            embedding = embedding.permute(2,0,1)
            
            # Normalize the embedding
            embedding = embedding / embedding.norm(dim=0, keepdim=True)
            
            # Convert embedding to RGB using PCA
            try:
                embedding_rgb = pca_to_rgb(embedding.cpu())
            except:
                embedding_rgb = np.zeros((_h, _w, 3), dtype=np.uint8) + 255
            
            emb_rgbs.append(embedding_rgb)
            emb_names.append(model)

    # Create a blank canvas to hold all images side by side
    combined = np.zeros((_h, _w * (2 + len(emb_names)), 3), dtype=np.uint8)
    
    # Place images side by side
    combined[:, :_w] = gt_image_np
    combined[:, _w:2*_w] = image_np
    
    for idx, (emb_rgb, emb_name) in enumerate(zip(emb_rgbs, emb_names)):
        combined[:, 2*_w + idx*_w: 2*_w + (idx+1)*_w] = add_image_marker(emb_rgb, emb_name, padding=5)
    
    # Save the combined image
    cv2.imwrite(output_filename, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

def vpca_embeddings(embedding, pca_components=None, pca_min=None, pca_max=None, pre_normalized=False):
    """
    Convert an embedding to an RGB image using provided global PCA parameters.
    If the PCA parameters are not provided, they are computed from the input embedding.
    
    Args:
        embedding (Tensor): An embedding tensor of shape (C, H, W).
        pca_components (Tensor, optional): PCA components (C, 3) from a global PCA.
        pca_min (Tensor, optional): Global minimum values along each PCA dimension.
        pca_max (Tensor, optional): Global maximum values along each PCA dimension.
        pre_normalized (bool): Indicates whether the input embedding is already normalized.
            If False, the embedding will be normalized before processing.
    
    Returns:
        np.ndarray: An RGB image (H, W, 3) in uint8.
    """
    c, h, w = embedding.shape

    if not pre_normalized:
        embedding = embedding / embedding.norm(dim=0, keepdim=True)

    try:
        embedding_rgb = pca_to_rgb(
            embedding.float().cpu(),
            pca_components,
            pca_min,
            pca_max,
            pre_normalized=pre_normalized,
        )
    except Exception as e:
        print(f"Error in global PCA conversion: {e}")
        embedding_rgb = np.full((h, w, 3), 255, dtype=np.uint8)

    return embedding_rgb

# Example usage:
# Assuming gt_image, image, and embedding are your input tensors
# visualize_combined(gt_image, image, embedding)