from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import matplotlib.pyplot as plt
import torch
import numpy as np

from .modeling_siglip import SiglipModel

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

def visualize_feature_l2(feature_tensor, size=(16, 16), filename="mask_l2_robust.png", cmap='viridis', lower_percentile=1, upper_percentile=99):
    if feature_tensor.dim() > 2:
        feature_tensor = feature_tensor.view(size[0], size[1], -1)
    
    # Compute L2 norm for each feature vector
    l2_norm = torch.norm(feature_tensor, p=2, dim=2)
    
    # Convert to numpy for percentile calculation
    l2_norm_np = l2_norm.numpy()
    
    # Calculate percentile-based min and max
    l2_min = np.percentile(l2_norm_np, lower_percentile)
    l2_max = np.percentile(l2_norm_np, upper_percentile)
    
    # Clip and normalize to [0, 1] range
    l2_norm_normalized = np.clip(l2_norm_np, l2_min, l2_max)
    l2_norm_normalized = (l2_norm_normalized - l2_min) / (l2_max - l2_min)
    
    plt.figure(figsize=(5, 5))
    im = plt.imshow(l2_norm_normalized, cmap=cmap)
    plt.colorbar(im, label=f"Normalized L2 Norm ({lower_percentile}th to {upper_percentile}th percentile)")
    plt.axis('off')
    plt.title("Feature Visualization (L2 Norm)")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Robust L2 norm visualization saved as {filename}")

model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a photo of 2 cats", "a photo of 2 dogs"]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.vision_model_output.last_hidden_state
last_hidden_state = last_hidden_state.reshape(1, 27, 27, -1)
visualize_feature(last_hidden_state, size=(27, 27))