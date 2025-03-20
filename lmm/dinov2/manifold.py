import os
import json
import umap
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler


def load_dinov2_embeddings(data_root):
    # Load the JSON file containing image information
    json_path = os.path.join(data_root, "dinov2_info.json")
    with open(json_path, "r") as f:
        info = json.load(f)
    
    # Iterate through each image in the dataset
    all_pixel_embeddings = []
    all_image_embeddings = []
    for image_info in info["images"]:
        # Load the embeddings
        emb_path = image_info["emb_pth"]
        embeddings = torch.load(emb_path)
        
        # Extract pixel embeddings and image embeddings
        pixel_embeds = embeddings["pixel_embeds"].view(-1, embeddings["pixel_embeds"].shape[-1])
        image_embeds = embeddings["image_embeds"].view(-1, embeddings["image_embeds"].shape[-1])
        
        all_pixel_embeddings.append(pixel_embeds)
        all_image_embeddings.append(image_embeds)
    
    all_pixel_embeddings = torch.cat(all_pixel_embeddings, dim=0)
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    return all_pixel_embeddings, all_image_embeddings

def create_continuous_colormap():
    # Creating a colormap that transitions smoothly between multiple colors
    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
    return LinearSegmentedColormap.from_list('custom_continuous', colors, N=256)

def visualize_umap(embeddings, output_file, title):
    embeddings_np = embeddings.numpy()
    
    print(f"Applying UMAP to {title}...")
    umap_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(embeddings_np)
    
    print(f"Creating visualization for {title}...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a continuous colormap
    cmap = create_continuous_colormap()
    
    # Normalize the UMAP coordinates to [0, 1] range for coloring
    scaler = MinMaxScaler()
    umap_normalized = scaler.fit_transform(umap_embeddings)
    
    # Calculate color based on normalized UMAP coordinates
    colors = np.sum(umap_normalized, axis=1)
    
    scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                         c=colors, cmap=cmap, s=0.1, alpha=0.7)
    
    plt.setp(ax, xticks=[], yticks=[])
    plt.title(f"UMAP Visualization of DINOv2 {title} (Continuous Colormap)")
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Position in UMAP space')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    data_root = "/data/xueyanz/data/tandt/train"  # Update this path as needed
    
    print("Loading DINOv2 embeddings...")
    pixel_embeddings, image_embeddings = load_dinov2_embeddings(data_root)
    
    print(f"Loaded pixel embeddings shape: {pixel_embeddings.shape}")
    print(f"Loaded image embeddings shape: {image_embeddings.shape}")
    
    # Visualize pixel embeddings
    pixel_output_file = "dinov2_umap_pixel_embeddings.png"
    print("Applying UMAP and visualizing pixel embeddings...")
    visualize_umap(pixel_embeddings, pixel_output_file, "Pixel Embeddings")
    print(f"Pixel embeddings visualization saved to {pixel_output_file}")
    
    # Visualize image embeddings
    image_output_file = "dinov2_umap_image_embeddings.png"
    print("Applying UMAP and visualizing image embeddings...")
    visualize_umap(image_embeddings, image_output_file, "Image Embeddings")
    print(f"Image embeddings visualization saved to {image_output_file}")