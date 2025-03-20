import os
import json
import torch
import torch.nn.functional as F
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


def load_siglip_embeddings(data_root, device):
    json_path = os.path.join(data_root, "siglip_info.json")
    with open(json_path, "r") as f:
        info = json.load(f)
    
    all_embeddings = []
    for image_info in tqdm(info["images"], desc="Loading SIGLIP embeddings"):
        emb_path = os.path.join(data_root, "/".join(image_info["emb_pth"].split('/')[-3:]))
        embeddings = torch.load(emb_path, map_location=device)
        pixel_embeds = embeddings["pixel_embeds"]
        pixel_embeds = pixel_embeds.view(-1, pixel_embeds.shape[-1])
        all_embeddings.append(pixel_embeds)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

def visualize_umap_3d(original_embeddings, memory_embeddings, output_file="", embed_file=""):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    from matplotlib.colors import LinearSegmentedColormap
    import os
    import umap

    # Try to load existing embeddings first
    try:
        print(f"Attempting to load existing embeddings from {embed_file}...")
        umap_embeddings = np.load(embed_file)
        original_umap = umap_embeddings[:len(original_embeddings)]
        memory_umap = umap_embeddings[len(original_embeddings):]
        print("Successfully loaded existing embeddings.")
    except (FileNotFoundError, IOError):
        print(f"No existing embeddings found at {embed_file}")
        print("Computing new UMAP embeddings...")
        # Compute new embeddings
        combined_embeddings = torch.cat((original_embeddings, memory_embeddings), dim=0).cpu().numpy()
        umap_embeddings = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(combined_embeddings)
        
        original_umap = umap_embeddings[:len(original_embeddings)]
        memory_umap = umap_embeddings[len(original_embeddings):]
        
        # Save the computed embeddings
        print(f"Saving computed embeddings to {embed_file}")
        np.save(embed_file, umap_embeddings)
        print("Embeddings saved successfully.")
    
    def distance_to_origin(points):
        return np.sqrt(np.sum(points**2, axis=1))

    def normalize_distances(distances):
        return (distances - distances.min()) / (distances.max() - distances.min())

    def create_color_maps():
        blue_cmap = LinearSegmentedColormap.from_list("light_to_dark_blue", ["lightblue", "darkblue"])
        red_cmap = LinearSegmentedColormap.from_list("light_to_dark_red", ["#ffcdd2", "#b71c1c"])
        return blue_cmap, red_cmap

    def create_3d_plot(original_umap, memory_umap, original_colors, memory_colors, elevation, azimuth, zoom_factor):
        fig = plt.figure(figsize=(20, 16), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # Make the background transparent
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_facecolor('none')

        blue_cmap, red_cmap = create_color_maps()

        # Plot original points with blue colormap
        scatter_original = ax.scatter(original_umap[:, 0], original_umap[:, 1], original_umap[:, 2],
                   c=original_colors, cmap=blue_cmap, s=1, alpha=0.2, label='Original')
        
        # Plot memory points with red colormap
        scatter_memory = ax.scatter(memory_umap[:, 0], memory_umap[:, 1], memory_umap[:, 2],
                   c=memory_colors, cmap=red_cmap, s=2, alpha=1.0, label='Memory')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.view_init(elev=elevation, azim=azimuth)

        xlim = np.array([-3, 3]) * zoom_factor
        ylim = np.array([-3, 3]) * zoom_factor
        zlim = np.array([-3, 3]) * zoom_factor

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        ax.set_axis_off()

        plt.tight_layout(pad=0)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return fig, ax

    def calculate_initial_view(original_umap, memory_umap):
        all_points = np.vstack((original_umap, memory_umap))
        max_distance = np.max(np.linalg.norm(all_points, axis=1))
        zoom_adjustment_factor = 2.8
        initial_zoom = zoom_adjustment_factor / max_distance
        center_of_mass = np.mean(all_points, axis=0)
        return initial_zoom, center_of_mass

    def make_frame(t, original_umap, memory_umap, original_colors, memory_colors, initial_zoom, center_of_mass, frames_dir):
        initial_elevation = 30
        initial_azimuth = 45
        
        elevation = initial_elevation + 40 * np.sin(t * 0.5)
        azimuth = initial_azimuth + t * 36
        zoom_factor = initial_zoom + t * (0.1/20)
        
        fig, ax = create_3d_plot(original_umap, memory_umap, original_colors, memory_colors, elevation, azimuth, zoom_factor)
        
        limit = 1 / zoom_factor
        ax.set_xlim(-limit + center_of_mass[0], limit + center_of_mass[0])
        ax.set_ylim(-limit + center_of_mass[1], limit + center_of_mass[1])
        ax.set_zlim(-limit + center_of_mass[2], limit + center_of_mass[2])

        if frames_dir:
            fig.savefig(f"{frames_dir}/frame_{int(t*100):05d}.png", dpi=300, bbox_inches='tight')
        
        frame = mplfig_to_npimage(fig)
        plt.close(fig)
        return frame

    # Calculate distances and colors
    print("Calculating distances and colors...")
    original_distances = distance_to_origin(original_umap)
    memory_distances = distance_to_origin(memory_umap)
    original_colors = normalize_distances(original_distances)
    memory_colors = normalize_distances(memory_distances)

    # Calculate initial view parameters
    initial_zoom, center_of_mass = calculate_initial_view(original_umap, memory_umap)

    # Setup video parameters
    duration = 20  # seconds
    fps = 1
    
    # Create frames directory if needed
    frames_dir = os.path.splitext(output_file)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # Create and save video
    print(f"Generating video animation... Duration: {duration}s, FPS: {fps}")
    video = VideoClip(
        lambda t: make_frame(t, original_umap, memory_umap, original_colors, memory_colors, 
                           initial_zoom, center_of_mass, frames_dir), 
        duration=duration
    )

    video.write_videofile(output_file, fps=fps)
    print(f"Video saved as {output_file}")
    print(f"Individual frames saved in {frames_dir}")

if __name__ == "__main__":
    model_type = "siglip"
    data_name = "train"
    data_root = "/data/xueyanz/data/3dgs/{}".format(data_name)  # Update this path

    mem_root = os.path.join(data_root, "{}/mem85.emb".format(model_type))
    output_file = "/data/xueyanz/output/mmm/visual/{}_{}.mp4".format(model_type, data_name)
    embed_file = "/data/xueyanz/output/mmm/visual/{}_{}_emb.npy".format(model_type, data_name)
    
    # Set up CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading {} embeddings...".format(model_type))
    all_embeddings = globals()["load_{}_embeddings".format(model_type)](data_root, device)
    print(f"Loaded embeddings shape: {all_embeddings.shape}")
    
    print("Extracting memory features...")
    memory_embeddings = torch.load(mem_root, map_location=device)

    print("Applying 3D UMAP and visualizing...")
    visualize_umap_3d(all_embeddings, memory_embeddings, output_file=output_file, embed_file=embed_file)
    print('Done.')
