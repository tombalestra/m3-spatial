from flask import Flask, request, send_file, jsonify
import torch
import numpy as np
from PIL import Image
import io
import math
import os
import sys
import cv2
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast

# Import necessary modules from camear_to_render
from scene import Scene, SceneDataset
from gaussian_renderer import GaussianModel, render_final, distributed_preprocess3dgs_and_all2all_final
from gaussian_renderer.workload_division import DivisionStrategyHistoryFinal, start_strategy_final
from gaussian_renderer.loss_distribution import load_camera_from_cpu_to_all_gpu_for_eval
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    get_combined_args,
    init_args,
)
from utils.general_utils import safe_state, set_args, init_distributed, set_log_file
from xy_utils.memory import index_to_raw
from xy_utils.visual import vpca_embeddings
from .camera_utils import pdb_to_posrot, posrot_to_pdb
import utils.general_utils as utils

app = Flask(__name__, static_url_path='', static_folder='static')

class GaussianRenderer:
    def __init__(self, dataset, pipeline, model_path):
        """Initialize the Gaussian Splatting renderer.
        
        Args:
            model_path: Path to the trained model
            iteration: Iteration of the model to load
            use_embed: Whether to use embeddings
        """
        self.model_path = model_path
        self.dataset = dataset
        self.pipeline = pipeline
                
        self.previous_cache = {
            'position': None,
            'rotation': None,
            'image': None,
            'embedding': None
        }

        args = utils.get_args()
        self.args = args

        # Initialize CUDA and distributed setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_renderer()
        
        self.height = self.scene.test_cameras[0].image_height
        self.width = self.scene.test_cameras[0].image_width

    def initialize_renderer(self):
        """Initialize the Gaussian Splatting renderer."""
        print("Initializing Gaussian model and scene...")
        with torch.no_grad():
            self.gaussians = GaussianModel(self.dataset.sh_degree, self.dataset.emb_degree, self.args.use_embed)
            self.scene = Scene(self.args, self.gaussians, load_iteration=-1, shuffle=False, _eval=True)
            self.scene.load_weights(self.model_path)
        
        # Setup background
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        
        # Setup background
        self.bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device=self.device)
        print("Renderer initialization complete")

    def create_camera_from_posrot(self, position, rotation, width, height):
        """Create a camera object from position and rotation."""
        # Convert position and rotation to PDB format
        pos_rot_dict = {
            'x': position[0],
            'y': position[1],
            'z': position[2],
            'pitch': rotation[0],
            'yaw': rotation[1],
            'roll': rotation[2]
        }
        
        # Use the saved FoV values if available, otherwise use defaults
        fovX = self.fovX if hasattr(self, 'fovX') else 0.8
        fovY = self.fovY if hasattr(self, 'fovY') else 0.8 * height / width
        
        # Convert to PDB camera format
        pdb_camera = posrot_to_pdb(pos_rot_dict, {'fovX': fovX, 'fovY': fovY})
        
        # Create a camera object with the correct properties based on the Camera class signature
        from scene.cameras import Camera
        
        # Create camera with the required parameters
        camera = Camera(
            colmap_id=-1,  # Using -1 as this is not from COLMAP
            R=np.array(pdb_camera['R']),
            T=np.array(pdb_camera['T']),
            FoVx=pdb_camera['FoVx'],
            FoVy=pdb_camera['FoVy'],
            image=None,  # No image for rendering
            embeddings=None,  # No embeddings for rendering
            gt_alpha_mask=None,  # No alpha mask for rendering
            image_name=f"render_view_{position}_{rotation}",
            uid=0,  # Using 0 as default UID for rendered views
            trans=np.array(pdb_camera['trans']),
            scale=pdb_camera['scale']
        )
        return camera
    
    def check_cache(self, position, rotation):
        """Check if the current render parameters are the same as the previous render."""
        prev = self.previous_cache
        if prev['position'] is None or prev['rotation'] is None:
            return False

        prev_position = np.array(prev['position'])
        prev_rotation = np.array(prev['rotation']) 

        # check distance
        position_distance = np.linalg.norm(np.array(position) - prev_position)
        rotation_distance = np.linalg.norm(np.array(rotation) - prev_rotation)
        
        # error rate 1e-6
        if position_distance < 1e-6 and rotation_distance < 1e-6:
            return True
        return False

    def render_view(self, camera_position, camera_rotation, mode='rgb', feature_index=0):
        """Render a view from the specified camera position and rotation."""
        # print(f"Rendering view: pos={camera_position}, rot={camera_rotation}, mode={mode}, feature_index={feature_index}")
        
        is_cache = self.check_cache(camera_position, camera_rotation)
                
        # Update dimensions if needed
        width = self.width
        height = self.height
        
        with torch.no_grad():
            if not is_cache:
                # Create camera from position and rotation
                camera = self.create_camera_from_posrot(camera_position, camera_rotation, width, height)
                
                # Create batched camera list
                batched_cameras = [camera]
                _dataset = SceneDataset(batched_cameras)
                
                # Create strategy history and division
                strategy_history = DivisionStrategyHistoryFinal(
                    _dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
                )
                batched_strategies, gpuid2tasks = start_strategy_final(
                    batched_cameras, strategy_history
                )
                load_camera_from_cpu_to_all_gpu_for_eval(
                    batched_cameras, batched_strategies, gpuid2tasks
                )
                
                # Preprocess Gaussians and render
                batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
                    batched_cameras,
                    self.gaussians,
                    self.pipeline,
                    self.background,
                    batched_strategies=batched_strategies,
                    mode="test",
                )
                
                # Render the image
                batched_image, batched_embeddings, _ = render_final(
                    batched_screenspace_pkg, 
                    batched_strategies, 
                    use_embed=self.args.use_embed
                )
                
                # Extract image and embeddings
                image = batched_image[0]
                # Clamp image values
                image = torch.clamp(image, 0.0, 1.0)
                embedding = batched_embeddings[0] if batched_embeddings else None
                self.previous_cache = {
                    'position': camera_position,
                    'rotation': camera_rotation,
                    'image': image,
                    'embedding': embedding
                }
            else:
                # Use cached image and embeddings
                image = self.previous_cache['image']
                embedding = self.previous_cache['embedding']
            
            # If feature mode and embeddings are available, process them
            if mode != 'rgb' and embedding is not None:
                model_names = ["clip", "siglip", "dinov2", "seem", "llama3", "llamav"]
                
                # Use feature_index to select which model to visualize (if valid)
                if 1 <= feature_index <= len(model_names):
                    selected_model = model_names[feature_index - 1]
                    
                    # with autocast(enabled=True, dtype=torch.float16):
                    # Check if the selected model is available
                    if getattr(self.args, f"use_{selected_model}", False):
                        emb_proj = self.scene.emb_proj_ops[selected_model]
                        emb_mem = self.scene.emb_mem_ops[selected_model]
                        pred = embedding[getattr(self.args, f'{selected_model}_bit')[0]:getattr(self.args, f'{selected_model}_bit')[1],:,:]
                        
                        c,h,w = pred.shape
                        embedding_resized = F.interpolate(
                            pred[None,], (h//4, w//4), mode='bilinear', align_corners=True
                        )[0]
                        raw_feature = index_to_raw(embedding_resized, emb_proj, emb_mem, _eval=True, _temp=args.softmax_temp).float()
                        image = vpca_embeddings(raw_feature.permute(2,0,1).cpu())
                        return TF.to_pil_image(image)
                
                # If no specific model was selected or the selected model is not available,
                # fall back to RGB mode
                return TF.to_pil_image(image)
            else:
                return TF.to_pil_image(image)

# Global renderer instance
renderer = None

@app.route('/render', methods=['POST'])
def render():
    global renderer
    
    if renderer is None:
        return jsonify({"error": "Renderer not initialized"}), 500
    
    data = request.json
    camera_position = data['position']
    camera_rotation = data['rotation']
    mode = data['mode']
    feature_index = data.get('featureIndex', 0)  # Get feature index with default 0

    # Render the image
    with torch.no_grad():
        img = renderer.render_view(
            camera_position,
            camera_rotation,
            mode,
            feature_index=feature_index  # Pass feature index to render_view
        )
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return send_file(img_byte_arr, mimetype='image/png')        

@app.route('/config', methods=['GET'])
def get_config():
    global renderer
    
    if renderer is None:
        return jsonify({"error": "Renderer not initialized"}), 500
    
    return jsonify({
        'width': renderer.width,
        'height': renderer.height
    })

@app.route('/initial_camera', methods=['GET'])
def get_initial_camera():
    global renderer
    
    if renderer is None:
        return jsonify({"error": "Renderer not initialized"}), 500

    init_camera = True
    # Get the first camera from the dataset if available
    if init_camera:
        first_camera = renderer.scene.getTestCameras()[0]
        
        # Save the FoV values from the first camera to the renderer
        renderer.fovX = first_camera.FoVx
        renderer.fovY = first_camera.FoVy
        
        # Convert camera to position/rotation format
        camera_data = pdb_to_posrot({
            'R': first_camera.R.tolist(),
            'T': first_camera.T.tolist(),
            'FoVx': first_camera.FoVx,
            'FoVy': first_camera.FoVy,
            'scale': first_camera.scale
        })
        
        # Extract position and rotation
        position = list(camera_data['position'].values())
        rotation = list(camera_data['rotation'].values())
        
        return jsonify({
            'position': position,
            'rotation': rotation,
            'fovX': first_camera.FoVx,
            'fovY': first_camera.FoVy
        })
    else:
        # Fallback to default values if no cameras are available
        return jsonify({
            'position': [0, 0, 5],
            'rotation': [0, -90, 0],
            'fovX': 0.8,
            'fovY': 0.8 * renderer.height / renderer.width
        })

@app.route('/')
def index():
    return app.send_static_file('index.html')

def initialize_renderer(dataset, pipeline, model_path):
    """Initialize the global renderer instance."""
    global renderer
    renderer = GaussianRenderer(dataset, pipeline, model_path)    

if __name__ == '__main__':
    import argparse    
    parser = argparse.ArgumentParser(description="Gaussian Splatting Web Server")
    parser.add_argument("--port", default=6037, type=int, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--web_render", action="store_true")
    
    # Add the same arguments as in camear_to_render.py
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    
    # Parse arguments
    args = get_combined_args(parser)
    if args.skip_train:
        args.num_train_cameras = 0
    if args.skip_test:
        args.num_test_cameras = 0
    init_distributed(args)

    log_file = open(
        args.model_path
        + f"/render_ws={utils.DEFAULT_GROUP.size()}_rk_{utils.DEFAULT_GROUP.rank()}.log",
        "w",
    )
    set_log_file(log_file)
    init_args(args)
    
    # Initialize renderer
    initialize_renderer(lp.extract(args), pp.extract(args), args.model_path)
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=False)