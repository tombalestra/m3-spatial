import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import umap
from sklearn.preprocessing import MinMaxScaler

def load_clip_embeddings(data_root, device):
    json_path = os.path.join(data_root, "clip_info.json")
    with open(json_path, "r") as f:
        info = json.load(f)
    
    all_embeddings = []
    for image_info in info["images"]:
        emb_path = os.path.join(data_root, "/".join(image_info["emb_pth"].split('/')[-3:]))
        embeddings = torch.load(emb_path, map_location=device)
        pixel_embeds = embeddings["pixel_embeds"]
        pixel_embeds = pixel_embeds.view(-1, pixel_embeds.shape[-1])
        all_embeddings.append(pixel_embeds)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

def filter_embeddings_efficient(embeddings, threshold=0.9, chunk_size=1000):
    num_embeddings = embeddings.shape[0]
    device = embeddings.device
    
    normalized_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    filtered_indices = []
    used_mask = torch.zeros(num_embeddings, dtype=torch.bool, device=device)

    for i in tqdm(range(0, num_embeddings, chunk_size), desc="Filtering embeddings"):
        chunk = normalized_embeddings[i:i+chunk_size]
        similarity_chunk = torch.mm(chunk, normalized_embeddings.t())
        
        for j in range(chunk.shape[0]):
            if used_mask[i+j]:
                continue
            
            similar_indices = torch.where(similarity_chunk[j] >= threshold)[0]
            # If all, it means that we model all aspects of the feature.
            if not used_mask[similar_indices].all():
                filtered_indices.append(i+j)
                used_mask[similar_indices] = True

    filtered_embeddings = embeddings[filtered_indices]
    return filtered_embeddings

def visualize_umap(original_embeddings, memory_embeddings, output_file):
    combined_embeddings = np.vstack((original_embeddings.cpu().numpy(), memory_embeddings.cpu().numpy()))
    
    print("Applying UMAP...")
    umap_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(combined_embeddings)
    
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot original embeddings in blue
    ax.scatter(umap_embeddings[:len(original_embeddings), 0], 
               umap_embeddings[:len(original_embeddings), 1], 
               c='blue', s=0.1, alpha=0.5, label='Original')
    
    # Plot memory embeddings in red
    ax.scatter(umap_embeddings[len(original_embeddings):, 0], 
               umap_embeddings[len(original_embeddings):, 1], 
               c='red', s=0.1, alpha=0.7, label='Memory')
    
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("UMAP Visualization of Original and Memory Embeddings")
    plt.legend()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    data_root = "/data/xueyanz/data/3dgs/train"  # Update this path
    output_file = "clip_umap_original_and_memory.png"
    
    # Set up CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading CLIP embeddings...")
    all_embeddings = load_clip_embeddings(data_root, device)
    print(f"Loaded embeddings shape: {all_embeddings.shape}")
    
    print("Extracting memory features...")
    threshold = 0.65
    memory_embeddings = filter_embeddings_efficient(all_embeddings, threshold=threshold, chunk_size=5000)
    print(f"Extracted memory embeddings shape: {memory_embeddings.shape}")
    
    print("Applying UMAP and visualizing...")
    visualize_umap(all_embeddings, memory_embeddings, output_file)
    
    print(f"Visualization saved to {output_file}")