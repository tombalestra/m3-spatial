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

from evaluators import RetrievalEvaluator
from transformers import AutoProcessor
from .modeling_siglip import SiglipModel


def render_set(model_path, name, iteration, views, gaussians, pipeline, scene, background, siglip_model, annotations, coco_info, temperature):
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
    
    evaluator_siglip = RetrievalEvaluator("siglip", None)
    evaluator_mmm = RetrievalEvaluator("mmm", None)
    evaluator_siglip.reset()
    evaluator_mmm.reset()
    
    model, processor = siglip_model
    coco_image_embs, coco_text_embs = load_coco_emb(coco_info)

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
            emb_proj = scene.emb_proj_ops['siglip']
            emb_mem = scene.emb_mem_ops['siglip']
            gt_feature = gt_camera.original_embeddings_backup['siglip'].cuda() # [h,w,c]
            
            _,h,w = embedding.shape
            embedding = embedding[args.siglip_bit[0]:args.siglip_bit[1]]
            embedding_resized = F.interpolate(
                embedding[None,], (gt_feature.shape[0], gt_feature.shape[1]), mode='bicubic', align_corners=True
            )[0]
            raw_feature = index_to_raw(embedding_resized, emb_proj, emb_mem, _eval=True, _temp=temperature).float()
            raw_feature = raw_feature.reshape(1, -1, raw_feature.shape[-1])

            # Process Language Embedding
            image_id = gt_camera.image_name
            if image_id not in annotations:
                continue
            annot = annotations[image_id]
            caption = annot['caption']
            inputs = processor(text=[caption], padding="max_length", return_tensors="pt")
            inputs.to("cuda")
            with torch.autocast(device_type='cuda'):
                text_feature = model.get_text_features(**inputs)[None]

            gt_feature = gt_feature.reshape(1, -1, gt_feature.shape[-1])
            n_ex = coco_image_embs.shape[0]
            samples = torch.randint(0, n_ex, (1999,))
            sample_coco_image_embs = coco_image_embs[samples]
            sample_coco_text_embs = coco_text_embs[samples]
            
            raw_coco_image_embs = torch.cat([raw_feature.half(), sample_coco_image_embs.cuda()], dim=0)
            gt_coco_image_embs = torch.cat([gt_feature.half(), sample_coco_image_embs.cuda()], dim=0)
            sample_coco_text_embs = torch.cat([text_feature.half(), sample_coco_text_embs.cuda()], dim=0)
            
            raw_coco_image_embs = raw_coco_image_embs / (raw_coco_image_embs.norm(dim=-1, keepdim=True) + 1e-6)            
            gt_coco_image_embs = gt_coco_image_embs / (gt_coco_image_embs.norm(dim=-1, keepdim=True) + 1e-6)
            sample_coco_text_embs = sample_coco_text_embs / (sample_coco_text_embs.norm(dim=-1, keepdim=True) + 1e-6)
            
            text_feature = text_feature / (text_feature.norm(dim=-1, keepdim=True) + 1e-6)
            raw_feature = raw_feature / (raw_feature.norm(dim=-1, keepdim=True) + 1e-6)
            gt_feature = gt_feature / (gt_feature.norm(dim=-1, keepdim=True) + 1e-6)
            
            raw_logits_per_text = torch.einsum('nbc,mdc->nbmd', sample_coco_text_embs, raw_feature.half())
            raw_i2t = raw_logits_per_text[:,0,0,:].max(dim=-1).values
            
            gt_logits_per_text = torch.einsum('nbc,mdc->nbmd', sample_coco_text_embs, gt_feature.half())
            gt_i2t = gt_logits_per_text[:,0,0,:].max(dim=-1).values

            raw_logits_per_image = torch.einsum('nbc,mdc->nbmd', text_feature.half(), raw_coco_image_embs)
            raw_t2i = raw_logits_per_image[0,0,:,:].max(dim=-1).values

            gt_logits_per_image = torch.einsum('nbc,mdc->nbmd', text_feature.half(), gt_coco_image_embs)
            gt_t2i = gt_logits_per_image[0,0,:,:].max(dim=-1).values

            evaluator_siglip.process(
                [{"i2t": gt_i2t, "t2i": gt_t2i}],
            )
            
            evaluator_mmm.process(
                [{"i2t": raw_i2t, "t2i": raw_t2i}],
            )
            
            # release memory usage
            gt_camera.original_image = None
            gt_camera.original_embeddings_backup = None

        if generated_cnt == args.generate_num:
            break
    
    results_siglip = evaluator_siglip.evaluate()
    results_mmm = evaluator_mmm.evaluate()
    print("CLIP performance", results_siglip)
    print("MMM performance", results_mmm)

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    annot_pth: str,
    coco_info: str,
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
        
        # Prepare model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384", device_map='cuda')
        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

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
                (model, processor),
                image_id_to_annotation,
                args.coco_info,
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
                (model, processor),
                image_id_to_annotation,
                args.coco_info,
                args.softmax_temp,
            )

def load_coco_emb(embed_info_path):
    """
    Load COCO embeddings from saved files.
    
    Args:
        embed_info_path (str): Path to the JSON file containing embedding information
        
    Returns:
        tuple: (coco_image_embs, coco_text_embs) - stacked tensor of image and text embeddings
    """
    # Load embedding info
    with open(embed_info_path, 'r') as f:
        info = json.load(f)
    
    images_info = info['images']
    
    # Lists to store embeddings
    image_embeddings = []
    text_embeddings = []
    
    # Load embeddings with progress bar
    for img_info in tqdm(images_info, desc="Loading embeddings"):
        emb_path = img_info['emb_pth']
        
        if not os.path.exists(emb_path):
            print(f"Warning: Embedding file not found: {emb_path}")
            continue
            
        try:
            embeddings = torch.load(emb_path)
            image_embeddings.append(embeddings['pixel_embeds'])
            text_embeddings.append(embeddings['text_embeds'])
        except Exception as e:
            print(f"Error loading {emb_path}: {str(e)}")
            continue
    
    # Stack all embeddings
    if len(image_embeddings) == 0:
        raise ValueError("No valid embeddings found!")
    
    coco_image_embs = torch.stack(image_embeddings, dim=0)
    coco_text_embs = torch.stack(text_embeddings, dim=0)
    return coco_image_embs, coco_text_embs

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
    parser.add_argument("--coco_info", default=-1, type=str)
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
        os.path.join(args.source_path, args.annot_name),
        args.coco_info,
    )