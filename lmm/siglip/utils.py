import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from PIL import Image
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

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
    
    pca_image_np = pca_image_np * 255
    cv2.imwrite(filename, pca_image_np)
    print(f"Visualization saved as {filename}") 
    
def visualize_feature_kmeans(features, image, size=(27, 27), filename="mask_kmeans.png", n_clusters=3, alpha=0.7):
    # Ensure features is a numpy array
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    # Reshape the features
    feature_map = features.reshape(-1, features.shape[-1])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_map)
    
    # Calculate distances to cluster centers
    distances = kmeans.transform(feature_map)
    
    # Normalize distances
    normalized_distances = 1 - (distances / distances.max(axis=0))
    
    # Create a colormap for each cluster
    colors = cv2.applyColorMap(np.arange(0, 255, 255/n_clusters, dtype=np.uint8), cv2.COLORMAP_RAINBOW)
    colors = colors.squeeze()
    
    # Create the visualization
    vis = np.zeros((*size, 3), dtype=np.float32)  # Change to float32 for smoother interpolation
    for i in range(n_clusters):
        mask = cluster_labels == i
        vis[mask.reshape(size)] = colors[i] * normalized_distances[mask, i][:, np.newaxis]
    
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Resize the mask to match the original image size with smoother interpolation
    vis_resized = cv2.resize(vis, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    vis_resized = np.clip(vis_resized, 0, 255).astype(np.uint8)  # Clip values and convert back to uint8
    
    # Ensure the image is in RGB format (PIL uses RGB by default)
    if len(image_np.shape) == 2:  # If grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # If RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Blend the original image with the visualization
    blended = cv2.addWeighted(image_np, 1 - alpha, vis_resized, alpha, 0)
    
    # Convert back to PIL Image
    result_image = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    
    # Save the blended image
    result_image.save(filename)
    
    print(f"Visualization overlaid on image and saved as {filename}")
    return result_image

def visualize_feature_gmm(features, image, size=(27, 27), filename="mask_gmm.png", n_components=10, alpha=0.7):
    # Ensure features is a numpy array
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    # Reshape the features
    feature_map = features.reshape(-1, features.shape[-1])
    
    # Perform Gaussian Mixture Model clustering
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(feature_map)
    
    # Get cluster assignments and probabilities
    cluster_labels = gmm.predict(feature_map)
    probabilities = gmm.predict_proba(feature_map)
    
    # Create a colormap for each cluster
    colors = cv2.applyColorMap(np.arange(0, 255, 255/n_components, dtype=np.uint8), cv2.COLORMAP_RAINBOW)
    colors = colors.squeeze()
    
    # Create the visualization
    vis = np.zeros((*size, 3), dtype=np.float32)
    for i in range(n_components):
        mask = cluster_labels == i
        vis[mask.reshape(size)] = colors[i] * probabilities[mask, i][:, np.newaxis]
    
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Resize the mask to match the original image size with smoother interpolation
    vis_resized = cv2.resize(vis, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    vis_resized = np.clip(vis_resized, 0, 255).astype(np.uint8)
    
    # Ensure the image is in RGB format (PIL uses RGB by default)
    if len(image_np.shape) == 2:  # If grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # If RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Blend the original image with the visualization
    blended = cv2.addWeighted(image_np, 1 - alpha, vis_resized, alpha, 0)
    
    # Convert back to PIL Image
    result_image = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    
    # Save the blended image
    result_image.save(filename)
    
    print(f"Visualization overlaid on image and saved as {filename}")
    
    return result_image


def visualize_mask_logits(mask_logits, image, size=(27, 27), filename="siglip_logits.png", alpha=0.6):
    # Ensure mask_logits is a numpy array
    if isinstance(mask_logits, torch.Tensor):
        mask_logits = mask_logits.cpu().numpy()
    
    # Reshape the mask_logits if necessary
    if len(mask_logits.shape) > 2:
        mask_logits = mask_logits.reshape(size[0], size[1])
    
    # Normalize the logits to 0-1 range
    mask_norm = (mask_logits - mask_logits.min()) / (mask_logits.max() - mask_logits.min() + 1e-6)
    
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Resize the mask_norm to match the original image size
    mask_resized = cv2.resize(mask_norm, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Ensure the image is in RGB format (PIL uses RGB by default)
    if len(image_np.shape) == 2:  # If grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # If RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Create a colormap
    colormap = plt.get_cmap('jet')
    heatmap = (colormap(mask_resized)[:, :, :3] * 255).astype(np.uint8)
    
    # Create a mask for non-zero areas in the heatmap
    mask = mask_resized > 0
    
    # Blend the heatmap with the original image only where the mask is True
    blended = image_np.copy()
    blended[mask] = cv2.addWeighted(image_np[mask], 1-alpha, heatmap[mask], alpha, 0)
    
    # Convert back to PIL Image
    result_image = Image.fromarray(blended)
    
    # Save the result image
    result_image.save(filename)
    
    # print(f"Visualization overlaid on image and saved as {filename}")
    return result_image