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
import torch
import json
import torch.distributed as dist
import torch.nn.functional as F
from skimage.filters import threshold_otsu
from scene import Scene, SceneDataset
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import (
    # preprocess3dgs_and_all2all,
    # render
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
)
import torchvision
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
from utils.loss_utils import l2_loss, cosine_loss
from xy_utils.memory import index_to_raw
from xy_utils.visual import vpca_embeddings

from evaluators import GroundingEvaluator
from transformers import AutoProcessor


def load_seem(embed_info, image_root, image_name):
    image_info = embed_info[image_name]
    segment_info = image_info["segment_info"]

    embed_pth = os.path.join(image_root, "/".join(image_info["emb_pth"].split('/')[-3:]))
    embeddings = torch.load(embed_pth)['pixel_embeds']
    n,c = embeddings.shape
    down_rate = 1
    
    mask_pth = os.path.join(image_root, "/".join(image_info["mask_pth"].split('/')[-3:]))
    mask = cv2.imread(mask_pth, cv2.IMREAD_GRAYSCALE)
    h,w = mask.shape
    mask = cv2.resize(mask, (w//down_rate, h//down_rate), interpolation=cv2.INTER_NEAREST)
    mask = torch.from_numpy(mask)
    valid_mask = mask != 255
    
    local_ids = [x['local_id'] for x in segment_info]
    gt_embeddings = torch.zeros((h//down_rate,w//down_rate,c)).type_as(embeddings)
    
    for local_id in local_ids:
        gt_embeddings[mask==local_id] = embeddings[local_id]

    output = {"seem": {
        "queries": embeddings,
        "masks": mask,
        "height": image_info["height"],
        "width": image_info["width"],
        "emb_height": h,
        "emb_width": w,
    }}
    return output

def render_set(model_path, name, iteration, views, gaussians, pipeline, scene, background, annotations, data_pth, masks_pth, temperature):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    emb_path = os.path.join(model_path, name, "ours_{}".format(iteration), "embedding")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(emb_path, exist_ok=True)
    
    dataset = SceneDataset(views)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_cur_iter(iteration)
    generated_cnt = 0

    num_cameras = len(views)
    strategy_history = DivisionStrategyHistoryFinal(
        dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )
    progress_bar = tqdm(
        range(1, num_cameras + 1),
        desc="Rendering progress",
        disable=(utils.LOCAL_RANK != 0),
    )
    
    evaluator_siglip = GroundingEvaluator("clip", None)
    evaluator_mmm = GroundingEvaluator("mmm", None)
    evaluator_siglip.reset()
    evaluator_mmm.reset()

    l2_dist = []
    cosine_dist = []
    for idx in range(1, num_cameras + 1, args.bsz):
        progress_bar.update(args.bsz)

        num_camera_to_load = min(args.bsz, num_cameras - idx + 1)
        batched_cameras = dataset.get_batched_cameras(num_camera_to_load)
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
            # name_set = torch.load("/data/xueyanz/data/3dgs/train/train_subset_names.da")
            # if gt_camera.image_name not in name_set:
            #     continue
            
            actual_idx = idx + camera_id
            if args.sample_freq != -1 and actual_idx % args.sample_freq != 0:
                continue
            if generated_cnt == args.generate_num:
                break

            if args.l != -1 and args.r != -1:
                if actual_idx < args.l or actual_idx >= args.r:
                    continue
            
            generated_cnt += 1

            if (
                image is None or len(image.shape) == 0
            ):  # The image is not rendered locally.
                image = torch.zeros(
                    gt_camera.original_image.shape, device="cuda", dtype=torch.float32
                )

            # process image
            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(gt_camera.original_image / 255.0, 0.0, 1.0)

            # process embeddings
            emb_proj = scene.emb_proj_ops['seem']
            emb_mem = scene.emb_mem_ops['seem']
            gt_feature = gt_camera.original_embeddings_backup['seem'].cuda() # [h,w,c]
            
            # process gt embeddings
            seem_gt = load_seem(annotations, data_pth, gt_camera.image_name)
            
            _,h,w = embedding.shape
            embedding = embedding[args.seem_bit[0]:args.seem_bit[1]]
            embedding_resized = F.interpolate(
                embedding[None,], (gt_feature.shape[0], gt_feature.shape[1]), mode='bicubic', align_corners=True
            )[0]
            # embedding_resized = F.interpolate(
            #     embedding[None,], (embedding.shape[1]//4, embedding.shape[2]//4), mode='bicubic', align_corners=True
            # )[0]
            # embedding_resized = embedding_resized[args.clip_bit[0]:args.clip_bit[1]]
            # embedding_resized = embedding
            raw_feature = index_to_raw(embedding_resized, emb_proj, emb_mem, _eval=True, _temp=temperature).float()
            
            # Process GT Masks
            visual_queries = seem_gt['seem']['queries'].cuda()
            gt_id = seem_gt['seem']['masks']
            gt_masks = torch.stack([gt_id==i for i in range(visual_queries.shape[0])]).cuda()
            _,_h,_w = gt_masks.shape

            norm_visual_queries = (visual_queries / visual_queries.norm(dim=1, keepdim=True)).float()
            norm_raw_feature = (raw_feature / (raw_feature.norm(dim=-1, keepdim=True) + 1e-6)).float()
            norm_gt_feature = (gt_feature / (gt_feature.norm(dim=-1, keepdim=True) + 1e-6)).float()

            gt_logits = norm_gt_feature @ norm_visual_queries.T
            raw_logits = norm_raw_feature @ norm_visual_queries.T
            
            gt_logits = F.interpolate(
                gt_logits.permute(2,0,1)[None,], (_h, _w), mode='bilinear', align_corners=True
            )[0]
            # Compute Threshold Cut for GT
            gt_thres = []
            for i in range(gt_logits.shape[0]):
                gt_thres += [threshold_otsu(gt_logits[i:i+1].cpu().numpy())]
            gt_thres = torch.tensor(gt_thres).to(gt_logits.device)[:,None,None]
            gt_logits = gt_logits > gt_thres
            
            raw_logits = F.interpolate(
                raw_logits.permute(2,0,1)[None,], (_h, _w), mode='bilinear', align_corners=True
            )[0]
            raw_thres = []
            for i in range(raw_logits.shape[0]):
                raw_thres += [threshold_otsu(raw_logits[i:i+1].cpu().numpy())]
            raw_thres = torch.tensor(raw_thres).to(raw_logits.device)[:,None,None]
            raw_logits = raw_logits > raw_thres

            # import matplotlib.pyplot as plt
            # gt_logits = gt_logits[0:1].float()
            # plt.imshow(gt_logits.cpu().squeeze().numpy(), cmap='coolwarm', vmin=gt_logits.min(), vmax=gt_logits.max())
            # plt.colorbar(label='Logits')
            # plt.savefig('test.png')
            # plt.close()
            
            # raw_logits = raw_logits[0:1].float()
            # plt.imshow(raw_logits.cpu().squeeze().numpy(), cmap='coolwarm', vmin=raw_logits.min(), vmax=raw_logits.max())
            # plt.colorbar(label='Logits')
            # plt.savefig('test2.png')
            # plt.close()
            # import pdb; pdb.set_trace()
            
            evaluator_siglip.process(
                [{"groundings": {"masks": gt_masks}}],
                [{"grounding_mask": gt_logits}],
            )
            
            evaluator_mmm.process(
                [{"groundings": {"masks": gt_masks}}],
                [{"grounding_mask": raw_logits}],
            )
            
            embedding_resized = F.interpolate(
                embedding[None,:], (gt_feature.shape[0], gt_feature.shape[1]), mode='bilinear', align_corners=True
            )[0]
            raw_feature = index_to_raw(embedding_resized, emb_proj, emb_mem, _eval=True, _temp=temperature).float()

            l2_dist += [l2_loss(raw_feature, gt_feature)]
            cosine_dist += [cosine_loss(raw_feature, gt_feature, dim=-1)]
            
            # Compute grounding metric
            torchvision.utils.save_image(
                image,
                os.path.join(render_path, "{0:05d}".format(actual_idx) + ".png"),
            )
            torchvision.utils.save_image(
                gt_image,
                os.path.join(gts_path, "{0:05d}".format(actual_idx) + ".png"),
            )
            cv2.imwrite(
                os.path.join(emb_path, "{0:05d}".format(actual_idx) + ".png"),
                vpca_embeddings(embedding),
            )

            # release memory usage
            gt_camera.original_image = None
            gt_camera.original_embeddings_backup = None

        if generated_cnt == args.generate_num:
            break
    
    results_siglip = evaluator_siglip.evaluate()
    results_mmm = evaluator_mmm.evaluate()
    print("SigLIP performance", results_siglip)
    print("MMM performance", results_mmm)
    print(f"[{name}]", f"l2_dist: {sum(l2_dist)/len(l2_dist)}, cosine_dist: {sum(cosine_dist)/len(cosine_dist)}")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    data_pth: str,
    annot_pth: str,
    masks_pth: str,
):
    # Prepare annotation
    annotations = json.load(open(annot_pth))
    image_id_to_annotation = {}
    for image_annot in annotations['images']:
        image_id_to_annotation[image_annot['image_id']] = image_annot
    
    with torch.no_grad():
        args = utils.get_args()
        gaussians = GaussianModel(dataset.sh_degree, dataset.emb_degree, args.use_embed)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False, _eval=True)
        scene.load_weights(args.load_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            render_set(
                args.load_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                scene,
                background,
                image_id_to_annotation,
                data_pth,
                masks_pth,
                args.softmax_temp,
            )

        if not skip_test:
            render_set(
                args.load_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                scene,
                background,
                image_id_to_annotation,
                data_pth,
                masks_pth,
                args.softmax_temp,
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
    parser.add_argument("--annot_name", default=-1, type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--generate_num", default=-1, type=int)
    parser.add_argument("--sample_freq", default=-1, type=int)
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
        args.skip_train,
        args.skip_test,
        args.source_path,
        os.path.join(args.source_path, args.annot_name),
        os.path.join(args.source_path, args.annot_name.split('.')[0] + '_mask'),
    )