import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from matplotlib.colors import LinearSegmentedColormap

def load_embeddings():
    embeddings = np.load("3d_emb.npy")
    original_umap = embeddings[:310632]
    memory_umap = embeddings[310632:]
    return original_umap, memory_umap

def distance_to_origin(points):
    return np.sqrt(np.sum(points**2, axis=1))

def normalize_distances(distances):
    return (distances - distances.min()) / (distances.max() - distances.min())

def create_color_maps():
    blue_cmap = LinearSegmentedColormap.from_list("light_to_dark_blue", ["lightblue", "darkblue"])
    red_cmap = LinearSegmentedColormap.from_list("light_to_dark_red", ["lightcoral", "darkred"])
    return blue_cmap, red_cmap

def create_3d_plot(original_umap, memory_umap, original_colors, memory_colors, elevation, azimuth, zoom_factor):
    # Increase the figure size and DPI for higher resolution
    fig = plt.figure(figsize=(20, 16), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Make the background transparent
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_facecolor('none')

    blue_cmap, red_cmap = create_color_maps()

    # Plot original points with blue colormap
    scatter_original = ax.scatter(original_umap[:, 0], original_umap[:, 1], original_umap[:, 2],
               c=original_colors, cmap=blue_cmap, s=1, alpha=0.1, label='Original')
    
    # Plot memory points with red colormap
    scatter_memory = ax.scatter(memory_umap[:, 0], memory_umap[:, 1], memory_umap[:, 2],
               c=memory_colors, cmap=red_cmap, s=7, alpha=1.0, label='Memory')

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
    
    # Adjust this factor to bring the view closer (higher value = closer view)
    zoom_adjustment_factor = 2.8
    
    initial_zoom = zoom_adjustment_factor / max_distance
    
    # Calculate the center of mass
    center_of_mass = np.mean(all_points, axis=0)
    
    return initial_zoom, center_of_mass

def make_frame(t, original_umap, memory_umap, original_colors, memory_colors, initial_zoom, center_of_mass, output_folder):
    initial_elevation = 30
    initial_azimuth = 45
    
    elevation = initial_elevation + 40 * np.sin(t * 0.5)
    azimuth = initial_azimuth + t * 36
    zoom_factor = initial_zoom + t * (0.1/20)
    
    fig, ax = create_3d_plot(original_umap, memory_umap, original_colors, memory_colors, elevation, azimuth, zoom_factor)
    
    # Adjust the plot limits to center on the origin
    limit = 1 / zoom_factor
    ax.set_xlim(-limit + center_of_mass[0], limit + center_of_mass[0])
    ax.set_ylim(-limit + center_of_mass[1], limit + center_of_mass[1])
    ax.set_zlim(-limit + center_of_mass[2], limit + center_of_mass[2])

    fig.savefig(f"{output_folder}/frame_{int(t*100):05d}.png", dpi=300, bbox_inches='tight')
    frame = mplfig_to_npimage(fig)
    plt.close(fig)
    return frame

def main():
    original_umap, memory_umap = load_embeddings()

    original_distances = distance_to_origin(original_umap)
    memory_distances = distance_to_origin(memory_umap)

    original_colors = normalize_distances(original_distances)
    memory_colors = normalize_distances(memory_distances)

    initial_zoom, center_of_mass = calculate_initial_view(original_umap, memory_umap)

    duration = 20  # seconds (10 for rotation, 5 for zoom, 5 for final rotation)
    fps = 30
    output_folder = "/data/xueyanz/data/3dgs/train/visual_manifold/clip"
    os.makedirs(output_folder, exist_ok=True)
    video = VideoClip(lambda t: make_frame(t, original_umap, memory_umap, original_colors, memory_colors, initial_zoom, center_of_mass, output_folder), duration=duration)

    output_file = "3d_visualization_improved.mp4"
    video.write_videofile(output_file, fps=fps)
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    main()