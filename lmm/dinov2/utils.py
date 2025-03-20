import torch
import matplotlib.pyplot as plt


def visualize_feature(feature_tensor, size=(16, 16), filename="mask.png"):
    # Ensure the feature tensor is 2D
    if feature_tensor.dim() > 2:
        feature_tensor = feature_tensor.view(-1, feature_tensor.size(-1))
    
    # Perform PCA to reduce to 3 dimensions
    U, S, V = torch.pca_lowrank(feature_tensor, q=3)
    pca_features = torch.matmul(feature_tensor, V[:, :3])
    
    # Normalize to [0, 1] range
    pca_min, pca_max = pca_features.min(dim=0)[0], pca_features.max(dim=0)[0]
    pca_norm = (pca_features - pca_min) / (pca_max - pca_min)
    
    # Reshape to image dimensions
    pca_image = pca_norm.view(size[0], size[1], 3)
    
    # Convert to numpy for matplotlib
    pca_image_np = pca_image.numpy()
    
    # Create a new figure
    plt.figure(figsize=(5, 5))
    plt.imshow(pca_image_np)
    plt.axis('off')
    plt.title("Feature Visualization")
    
    # Save the figure
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure to free up memory
    
    print(f"Visualization saved as {filename}")

def visualize_feature_l2(feature_tensor, size=(16, 16), filename="mask_l2.png", cmap='viridis'):
    if feature_tensor.dim() > 2:
        feature_tensor = feature_tensor.view(size[0], size[1], -1)
    
    # Compute L2 norm for each feature vector
    l2_norm = torch.norm(feature_tensor, p=2, dim=2)
    
    # Normalize to [0, 1] range
    l2_min, l2_max = l2_norm.min(), l2_norm.max()
    l2_norm_normalized = (l2_norm - l2_min) / (l2_max - l2_min)
    
    # Convert to numpy for matplotlib
    l2_image_np = l2_norm_normalized.numpy()
    
    plt.figure(figsize=(5, 5))
    im = plt.imshow(l2_image_np, cmap=cmap)
    plt.colorbar(im, label="Normalized L2 Norm")
    plt.axis('off')
    plt.title("Feature Visualization (L2 Norm)")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"L2 norm visualization saved as {filename}")