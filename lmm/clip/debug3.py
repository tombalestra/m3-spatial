import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

def generate_3d_data(num_points=1000):
    x = np.random.randn(num_points)
    y = np.random.randn(num_points)
    z = np.random.randn(num_points)
    return x, y, z

def create_3d_plot(x, y, z, elevation, azimuth, zoom_factor):
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=15)

    # Disable ticks and grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Set view angles
    ax.view_init(elev=elevation, azim=azimuth)

    # Compute the limits dynamically based on the zoom factor
    xlim = np.array([-3, 3]) * zoom_factor
    ylim = np.array([-3, 3]) * zoom_factor
    zlim = np.array([-3, 3]) * zoom_factor

    # Apply axis limits to simulate zoom
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_axis_off()

    plt.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return fig, ax

def make_frame(t):
    rotation_duration = 10
    zoom_duration = 5
    total_duration = 20
    
    if t < rotation_duration:  # First 10 seconds: rotation
        elevation = 20 + 70 * np.sin(t * 0.5)  # Elevation angle
        azimuth = t * 36  # Rotation angle (full 360 degrees in 10 seconds)
        zoom_factor = 1  # No zoom during rotation
    elif t < rotation_duration + zoom_duration:  # Next 5 seconds: zoom in
        # Keep the final elevation and azimuth from the rotation phase
        elevation = 20 + 70 * np.sin(rotation_duration * 0.5)
        azimuth = 360
        # Zoom factor decreases from 1 to 0.3 over 5 seconds
        zoom_factor = 1 - 0.14 * (t - rotation_duration)
    else:  # Final 5 seconds: rotate from zoomed-in position
        # Use the last zoomed-in elevation and zoom factor
        elevation = 20 + 70 * np.sin((rotation_duration + t) * 0.5)
        azimuth = 360 + (t - (rotation_duration + zoom_duration)) * 36  # Rotate further
        zoom_factor = 1  # Stay zoomed in
        
    fig, ax = create_3d_plot(x, y, z, elevation, azimuth, zoom_factor)
    frame = mplfig_to_npimage(fig)
    plt.close(fig)
    return frame

# Generate 3D data
x, y, z = generate_3d_data(1000)

# Create video clip
duration = 20  # seconds (10 for rotation, 5 for zoom, 5 for final rotation)
fps = 30
video = VideoClip(make_frame, duration=duration)

# Write video file
output_file = "3d_visualization_rotation_zoom_final_rotation.mp4"
video.write_videofile(output_file, fps=fps)

print(f"Video saved as {output_file}")