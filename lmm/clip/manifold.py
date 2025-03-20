import umap
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler

def load_clip_embeddings(data_root):
    json_path = os.path.join(data_root, "clip_info.json")
    with open(json_path, "r") as f:
        info = json.load(f)

    all_embeddings = []
    for image_info in info["images"]:
        emb_path = image_info["emb_pth"]
        embeddings = torch.load(emb_path)
        pixel_embeds = embeddings["pixel_embeds"]
        pixel_embeds = pixel_embeds.view(-1, pixel_embeds.shape[-1])
        all_embeddings.append(pixel_embeds)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

def create_continuous_colormap():
    # Creating a colormap that transitions smoothly between multiple colors
    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
    return LinearSegmentedColormap.from_list('custom_continuous', colors, N=256)

def visualize_umap(embeddings, output_file):
    embeddings_np = embeddings.numpy()

    print("Applying UMAP...")
    umap_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(embeddings_np)

    print("Creating visualization...")
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
    plt.title("UMAP Visualization of CLIP Embeddings (Continuous Colormap)")
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Position in UMAP space')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    data_root = "/data/xueyanz/data/tandt/train"  # Update this path as needed
    output_file = "clip_umap_embeddings_continuous.png"

    print("Loading CLIP embeddings...")
    all_embeddings = load_clip_embeddings(data_root)
    
    print(f"Loaded embeddings shape: {all_embeddings.shape}")
    
    print("Applying UMAP and visualizing...")
    visualize_umap(all_embeddings, output_file)
    
    print(f"Visualization saved to {output_file}")