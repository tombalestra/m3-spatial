import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import umap
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import os
os.environ['HOST'] = 'localhost'  # or '127.0.0.1'


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

# def distance_to_origin(points):
#     return np.sqrt(np.sum(points**2, axis=1))

# def visualize_umap_3d(original_embeddings, memory_embeddings, output_file):
#     combined_embeddings = np.vstack((original_embeddings.cpu().numpy(), memory_embeddings.cpu().numpy()))
    
#     print("Applying 3D UMAP...")
#     umap_embeddings = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(combined_embeddings)
    
#     print("Creating 3D visualization...")
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     original_umap = umap_embeddings[:len(original_embeddings)]
#     memory_umap = umap_embeddings[len(original_embeddings):]
    
#     # Calculate distances to origin
#     original_distances = distance_to_origin(original_umap)
#     memory_distances = distance_to_origin(memory_umap)
    
#     # Normalize distances to [0, 1] range
#     max_distance = max(original_distances.max(), memory_distances.max())
#     original_colors = 1 - original_distances / max_distance
#     memory_colors = 1 - memory_distances / max_distance
    
#     # Create custom colormaps
#     blue_cmap = LinearSegmentedColormap.from_list("", ["#FFFFFF", "#0000FF"])
#     red_cmap = LinearSegmentedColormap.from_list("", ["#FFFFFF", "#FF0000"])
    
#     # Plot original embeddings in blue (plotted first, so they'll be on the bottom)
#     scatter_original = ax.scatter(original_umap[:, 0], original_umap[:, 1], original_umap[:, 2], 
#                                   c=original_colors, cmap=blue_cmap, s=1, alpha=0.5, label='Original', zorder=10)
    
#     # Plot memory embeddings in red (plotted second, so they'll overlay the blue points)
#     scatter_memory = ax.scatter(memory_umap[:, 0], memory_umap[:, 1], memory_umap[:, 2], 
#                                 c=memory_colors, cmap=red_cmap, s=1, alpha=0.7, label='Memory', zorder=20)
    
#     # Set labels and title
#     ax.set_xlabel('UMAP 1')
#     ax.set_ylabel('UMAP 2')
#     ax.set_zlabel('UMAP 3')
#     ax.set_title("3D UMAP of Original and Memory Embeddings")
    
#     # Add legend
#     ax.legend()
    
#     # Adjust axis limits to fit the data more compactly
#     ax.set_xlim(umap_embeddings[:, 0].min(), umap_embeddings[:, 0].max())
#     ax.set_ylim(umap_embeddings[:, 1].min(), umap_embeddings[:, 1].max())
#     ax.set_zlim(umap_embeddings[:, 2].min(), umap_embeddings[:, 2].max())
    
#     # Adjust the view angle for better visibility
#     ax.view_init(elev=20, azim=45)
    
#     # Add colorbars
#     plt.colorbar(scatter_original, ax=ax, label='Distance to Origin (Original)', pad=0.1)
#     plt.colorbar(scatter_memory, ax=ax, label='Distance to Origin (Memory)', pad=0.1)
    
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     plt.close()


def visualize_umap_3d(original_embeddings, memory_embeddings, port=6007):
    combined_embeddings = np.vstack((original_embeddings.cpu().numpy(), memory_embeddings.cpu().numpy()))
    
    print("Applying 3D UMAP...")
    umap_embeddings = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(combined_embeddings)
    
    original_umap = umap_embeddings[:len(original_embeddings)]
    memory_umap = umap_embeddings[len(original_embeddings):]
    
    # Calculate distances to origin
    def distance_to_origin(points):
        return np.sqrt(np.sum(points**2, axis=1))
    
    original_distances = distance_to_origin(original_umap)
    memory_distances = distance_to_origin(memory_umap)
    
    # Normalize distances to [0, 1] range
    max_distance = max(original_distances.max(), memory_distances.max())
    original_colors = 1 - original_distances / max_distance
    memory_colors = 1 - memory_distances / max_distance
    
    # Create Plotly figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    # Add original embeddings trace
    fig.add_trace(
        go.Scatter3d(
            x=original_umap[:, 0],
            y=original_umap[:, 1],
            z=original_umap[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=original_colors,
                colorscale='Blues',
                opacity=0.5
            ),
            name='Original'
        )
    )
    
    # Add memory embeddings trace
    fig.add_trace(
        go.Scatter3d(
            x=memory_umap[:, 0],
            y=memory_umap[:, 1],
            z=memory_umap[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=memory_colors,
                colorscale='Reds',
                opacity=0.7
            ),
            name='Memory'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="3D UMAP of Original and Memory Embeddings",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
        legend_title="Embedding Type",
        margin=dict(r=0, b=0, l=0, t=40)
    )
    
    # Create Dash app
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        dcc.Graph(figure=fig, style={'height': '100vh'})
    ])
    
    # Run the app
    host = "127.0.0.1"
    print(f"Starting Dash server on http://{host}:{port}")
    app.run_server(debug=True, host=host, port=port)

if __name__ == "__main__":
    data_root = "/data/xueyanz/data/3dgs/train"  # Update this path
    output_file = "clip_umap_3d_original_and_memory.png"
    
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
    
    print("Applying 3D UMAP and visualizing...")
    # visualize_umap_3d(all_embeddings, memory_embeddings, output_file)
    visualize_umap_3d(all_embeddings, memory_embeddings, port=6047)
    print(f"Visualization saved to {output_file}")