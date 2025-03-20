#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu
from scene import Scene, SceneDataset
from gaussian_renderer import (
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
)
from utils.general_utils import (
    safe_state,
    set_args,
    init_distributed,
    set_log_file,
    set_cur_iter,
)
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer.loss_distribution import load_camera_from_cpu_to_all_gpu_for_eval
from gaussian_renderer.workload_division import (
    start_strategy_final,
    DivisionStrategyHistoryFinal,
)
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)

import utils.general_utils as utils
from xy_utils.visual import vpca_embeddings
from xy_utils.memory import index_to_raw

from .utils import extract_feature


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def render_set(iteration, views, gaussians, pipeline, scene, background, temperature, index, query):
    dataset = SceneDataset(views)
    set_cur_iter(iteration)
    strategy_history = DivisionStrategyHistoryFinal(
        dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )

    batched_cameras = dataset.get_batched_cameras_from_idx([index])
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

    for camera_id, (image, gt_camera, embedding) in enumerate(
        zip(batched_image, batched_cameras, batched_embeddings)
    ):  
        # process image
        image = torch.clamp(image, 0.0, 1.0)
        gt_image = torch.clamp(gt_camera.original_image / 255.0, 0.0, 1.0)

        # process embeddings
        emb_proj = scene.emb_proj_ops['llama3']
        emb_mem = scene.emb_mem_ops['llama3']
        gt_feature = gt_camera.original_embeddings_backup['llama3'].cuda() # [h,w,c]
        
        embedding = embedding[args.llama3_bit[0]:args.llama3_bit[1]]
        embedding_resized = F.interpolate(
            embedding[None,], (gt_feature.shape[0], gt_feature.shape[1]), mode='bicubic', align_corners=True
        )[0]
        raw_feature = index_to_raw(embedding_resized, emb_proj, emb_mem, _eval=True, _temp=temperature).float()
        text_features = extract_feature(query)[-12].mean(dim=0, keepdim=True)
        _,_h,_w = gt_image.shape

        norm_text_features = (text_features / text_features.norm(dim=1, keepdim=True)).float()
        norm_raw_feature = (raw_feature / (raw_feature.norm(dim=-1, keepdim=True) + 1e-6)).float()
        norm_gt_feature = (gt_feature / (gt_feature.norm(dim=-1, keepdim=True) + 1e-6)).float()

        gt_logits = norm_gt_feature @ norm_text_features.T
        raw_logits = norm_raw_feature @ norm_text_features.T
        
        gt_logits = F.interpolate(
            gt_logits.permute(2,0,1)[None,], (_h, _w), mode='bilinear', align_corners=True
        )[0]
        raw_logits = F.interpolate(
            raw_logits.permute(2,0,1)[None,], (_h, _w), mode='bilinear', align_corners=True
        )[0]

        gt_logits = gt_logits[0:1].float()
        data = gt_logits.cpu().squeeze().numpy()
        vmin = gt_logits.min().item()
        vmax = gt_logits.max().item()
        normalized_data = ((data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        cmap = plt.get_cmap('coolwarm')
        colored_data = cmap(normalized_data)
        colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel
        gt_logits_img = Image.fromarray(colored_data)
        gt_logits_img.save("gt_logits.png")

        raw_logits = raw_logits[0:1].float()
        data = raw_logits.cpu().squeeze().numpy()
        vmin = raw_logits.min().item()
        vmax = raw_logits.max().item()
        normalized_data = ((data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        cmap = plt.get_cmap('coolwarm')
        colored_data = cmap(normalized_data)
        colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel
        raw_logits_img = Image.fromarray(colored_data)
        raw_logits_img.save("raw_logits.png")
        
        cv2.imwrite(
            "feature.png",
            vpca_embeddings(embedding.cpu()),
        )
        cv2.imwrite("image.png", image.cpu().numpy().transpose(1,2,0) * 255)

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    index: int,
    query: str,
):
    with torch.no_grad():
        args = utils.get_args()
        gaussians = GaussianModel(dataset.sh_degree, dataset.emb_degree, args.use_embed)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False, _eval=True)
        scene.load_weights(args.load_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        render_set(
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipeline,
            scene,
            background,
            args.softmax_temp,
            index,
            query,
        )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--generate_num", default=-1, type=int)
    parser.add_argument("--sample_freq", default=-1, type=int)
    parser.add_argument("--text", default="train", type=str)
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--distributed_load", action="store_true")  # TODO: delete this.
    parser.add_argument("--l", default=-1, type=int)
    parser.add_argument("--r", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    init_distributed(args)
    # This script only supports single-gpu rendering.
    # I need to put the flags here because the render() function need it.
    # However, disable them during render.py because they are only needed during training.
    
    
    log_file = open(
        args.model_path
        + f"/render_ws={utils.DEFAULT_GROUP.size()}_rk_{utils.DEFAULT_GROUP.rank()}.log",
        "w",
    )
    set_log_file(log_file)

    ## Prepare arguments.
    # Check arguments
    init_args(args)
    if args.skip_train:
        args.num_train_cameras = 0
    if args.skip_test:
        args.num_test_cameras = 0
    # Set up global args
    set_args(args)

    print_all_args(args, log_file)

    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(
        lp.extract(args),
        args.iteration,
        pp.extract(args),
        args.index,
        args.text,
    )