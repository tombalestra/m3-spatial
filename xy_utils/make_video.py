import cv2
import os
from tqdm import tqdm
import glob
import numpy as np

def convert_jpg_sequence_to_mp4(input_dir, output_file='output.mp4', fps=30):
    """
    Convert a sequence of JPG files to an MP4 video.
    
    Args:
        input_dir (str): Directory containing the JPG files
        output_file (str): Output MP4 file path
        fps (int): Frames per second for the output video
    """
    # Get all jpg files in the directory
    jpg_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    
    if not jpg_files:
        print(f"No JPG files found in {input_dir}")
        return
    
    # Read the first image to get dimensions
    first_image = cv2.imread(jpg_files[0])
    height, width, layers = first_image.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    print(f"Converting {len(jpg_files)} images to video...")
    
    # Write each frame to video
    for jpg_file in tqdm(jpg_files):
        frame = cv2.imread(jpg_file)
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    # Example usage
    input_directory = "/disk1/checkpoint/mmm/garden_bsz1_gpu1_embTrue_clipFalse_sigFalse_dinoFalse_seemFalse_llaTrue_llvFalse_dim32_temp0.05_debug/run_0001/trace/ours_30000/llama3_mask/raw/"
    output_video = "garden_llama3_raw_mask.mp4"
    
    convert_jpg_sequence_to_mp4(
        input_dir=input_directory,
        output_file=output_video,
        fps=30  # You can adjust this value as needed
    )