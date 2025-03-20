import torch
import torch.nn.functional as F
import os
import cv2
import torchvision
from argparse import ArgumentParser
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
    print_all_args,
    get_combined_args,
    init_args,
)
from utils.general_utils import safe_state, set_args, init_distributed, set_log_file
from xy_utils.memory import index_to_raw
from xy_utils.visual import vpca_embeddings
import utils.general_utils as utils


def render_single_view_feature(
    dataset,
    pipeline,
    model_path,
    iteration,
    camera_idx,
    output_dir="./feature_output"
):
    """
    Render feature embeddings for a single camera view
    
    Args:
        model_path: Path to the trained model
        iteration: Iteration of the model to load
        camera_idx: Index of the camera to render (1-indexed as in the original code)
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        # Initialize model and scene
        args = utils.get_args()
        gaussians = GaussianModel(dataset.sh_degree, dataset.emb_degree, args.use_embed)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False, _eval=True)
        scene.load_weights(model_path)
        
        # Setup background
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Get the specified camera
        camera = scene.getTestCameras()[camera_idx]  # Convert to 0-indexed
        
        print(camera)
        batched_cameras = [camera]
        _dataset = SceneDataset(batched_cameras)
        strategy_history = DivisionStrategyHistoryFinal(
            _dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
        )
        batched_strategies, gpuid2tasks = start_strategy_final(
            batched_cameras, strategy_history
        )
        load_camera_from_cpu_to_all_gpu_for_eval(
            batched_cameras, batched_strategies, gpuid2tasks
        )

        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
            batched_cameras,
            gaussians,
            pipeline,
            background,
            batched_strategies=batched_strategies,
            mode="test",
        )
        batched_image, batched_embeddings, _ = render_final(batched_screenspace_pkg, batched_strategies, use_embed=args.use_embed)
        image = batched_image[0]
        embedding = batched_embeddings[0]

        # Part to complete
        image = torch.clamp(image, 0.0, 1.0)
        torchvision.utils.save_image(
            image,
            os.path.join(output_dir, "image_ren.png"),
        )

        # process embeddings
        model_names = ["clip", "siglip", "dinov2", "seem", "llama3", "llamav"]
        for model in model_names:
            if not getattr(args, "use_{}".format(model)):
                continue
            emb_proj = scene.emb_proj_ops[model]
            emb_mem = scene.emb_mem_ops[model]
            pred = embedding[getattr(args, f'{model}_bit')[0]:getattr(args, f'{model}_bit')[1],:,:]
            
            cv2.imwrite(
                os.path.join(output_dir, model + "_index.png"),
                vpca_embeddings(pred.cpu()),
            )

            c,h,w = pred.shape
            embedding_resized = F.interpolate(
                pred[None,], (h//4, w//4), mode='bilinear', align_corners=True
            )[0]
            raw_feature = index_to_raw(embedding_resized, emb_proj, emb_mem, _eval=True, _temp=args.softmax_temp).float()
            cv2.imwrite(
                os.path.join(output_dir, model + "_raw.png"),
                vpca_embeddings(raw_feature.permute(2,0,1).cpu()),
            )


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser(description="Single view feature rendering")
    
    # Add the same arguments as in the original render_metrics.py script
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    
    # Add standard arguments from the original script
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--generate_num", default=-1, type=int)
    parser.add_argument("--sample_freq", default=-1, type=int)
    parser.add_argument("--distributed_load", action="store_true")
    parser.add_argument("--l", default=-1, type=int)
    parser.add_argument("--r", default=-1, type=int)
    
    # Add our new single-view specific arguments
    parser.add_argument("--camera_idx", type=int, required=True, help="Camera index to render")
    parser.add_argument("--output_dir", default="./feature_output", help="Output directory")
    
    # Get combined args as in the original script
    args = get_combined_args(parser)    
    init_distributed(args)

    log_file = open(
        args.model_path
        + f"/render_ws={utils.DEFAULT_GROUP.size()}_rk_{utils.DEFAULT_GROUP.rank()}.log",
        "w",
    )
    set_log_file(log_file)

    init_args(args)
    if args.skip_train:
        args.num_train_cameras = 0
    if args.skip_test:
        args.num_test_cameras = 0
    # Set up global args
    set_args(args)

    # Call the rendering function
    render_single_view_feature(
        lp.extract(args),
        pp.extract(args),
        args.model_path,
        args.iteration,
        args.camera_idx,
        output_dir=args.output_dir
    )