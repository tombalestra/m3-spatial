import os
import glob
import json
import cv2
import base64
import random
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn.functional as F

# seem
from detectron2.structures import ImageList
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data import detection_utils as d2_utils

from vlcore.trainer.utils.misc import move_batch_to_device, cast_batch_to_half
from vlcore.xy_utils.image2html.visualizer import VL
from vlcore.utils.arguments import load_opt_from_config_file
from vlcore.utils.distributed import init_distributed 
from vlcore.utils.constants import COCO_PANOPTIC_CLASSES
from vlcore.modeling.BaseModel import BaseModel
from vlcore.modeling import build_model

metadata = MetadataCatalog.get('coco_2017_train_panoptic')

# local
from lmm.llama.semantic_sam.tasks import inference_semsam_m2m_auto
from lmm.llama.api.gpt4v import call_gpt4o
from lmm.llama.utils import parse_segment_data, extract_feature
from lmm.seem.grid_sample import create_circular_grid_masks
from lmm.seem.matrix_nms import matrix_nms, resolve_mask_conflicts

from .prompt import system_msg2, user_msg2
from .utils import add_image_marker


def build_transform_gen(min_scale, max_scale=None):
    augmentation = []
    augmentation.extend([
        T.ResizeShortestEdge(
            min_scale, max_size=max_scale
        ),
    ])
    return augmentation

data_root = "/data/xueyanz/data/3dgs/garden"
test_images = torch.load(os.path.join(data_root, "test_names.da"))
input_folder = os.path.join(data_root, "images")

marked_folder = os.path.join(data_root, ".marked_images_seem")
masks_folder = os.path.join(data_root, ".dict_masks_seem")

mask_folder = os.path.join(data_root, "vlm_info_seem_test_mask")
json_pth = os.path.join(data_root, "vlm_info_seem_test.json")
device = "cuda"

if not os.path.exists(marked_folder):
    os.makedirs(marked_folder)
if not os.path.exists(masks_folder):
    os.makedirs(masks_folder)
if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

image_pths = sorted(glob.glob(os.path.join(input_folder, "*.JPG")))

device = "cuda" if torch.cuda.is_available() else "cpu"
seem_cfg = "/data/xueyanz/code/vlcore_v2.0/vlcore/configs/seem/davitd5_unicl_lang_v1.yaml"
seem_ckpt = "/data/xueyanz/checkpoints/seem/seem_davit_d5.pt"
opt_seem = load_opt_from_config_file(seem_cfg)
opt_seem = init_distributed(opt_seem)
opt_seem['MODEL']['ENCODER']['NAME'] = 'transformer_encoder_deform'

model = BaseModel(opt_seem, build_model(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()
model.model.metadata = metadata

if json_pth is not None and os.path.exists(json_pth):
    info = json.load(open(json_pth))
else:
    info = {"information": '''
            1. global_id refers to the mark, this is corresponding to the segment_id. \n
            2. 0 in the mask indicates the background, and the other number indicates the mark. \n
            ''',
            "images": []}

marked_image_list = []
masks_dict_list = []
image_pth_list = []
image_pth_to_hw = {}

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
        for idx, image_pth in enumerate(image_pths):
            dataset_dict = {}
            image = d2_utils.read_image(image_pth, format="RGB")
            d2_utils.check_image_size(dataset_dict, image)
            tfm_gens = build_transform_gen(min_scale=640, max_scale=1333)
            image_ori, _ = T.apply_transform_gens(tfm_gens, image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
            dataset_dict.update({"image": images})
            batched_inputs = [dataset_dict]

            batched_inputs = move_batch_to_device(batched_inputs, 'cuda')
            batched_inputs = cast_batch_to_half(batched_inputs)
            
            images = [x["image"].to(model.model.device) for x in batched_inputs] # x["image"] is a tensor with shape [3, h, w] on cuda.
            images = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images]
            images = ImageList.from_tensors(images, model.model.size_divisibility)

            queries_grounding = None
            features = model.model.backbone(images.tensor)
            mask_features, _, multi_scale_features = model.model.sem_seg_head.pixel_decoder.forward_features(features)

            _, _, h, w = images.tensor.shape
            all_spatial_samples = create_circular_grid_masks(h, w, dot_spacing=60, dot_radius=12)
            interval = 10

            acc_masks = []
            acc_scores = []
            acc_labels = []
            for i in range(0, len(all_spatial_samples), interval):
                spatial_samples = all_spatial_samples[i:i+interval]
            
                extra = {}
                pos_masks = [spatial_samples.to(model.model.device)]
                pos_masks = ImageList.from_tensors(pos_masks, model.model.size_divisibility).tensor.unbind(0)

                neg_masks = [(spatial_samples.to(model.model.device) & False)]
                neg_masks = ImageList.from_tensors(neg_masks, model.model.size_divisibility).tensor.unbind(0)
                extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})
                
                queries_grounding = None
                results = model.model.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='seg')

                v_emb = results['pred_smaskembs']
                pred_smasks = results['pred_smasks']

                s_emb = results['pred_pspatials']
                diag_mask = ~(torch.eye(model.model.sem_seg_head.predictor.attention_data.extra['spatial_query_number'], device=s_emb.device).repeat_interleave(model.model.sem_seg_head.predictor.attention_data.extra['sample_size'],dim=0)).bool()
                offset = torch.zeros_like(diag_mask, device=s_emb.device).float()
                offset.masked_fill_(diag_mask, float("-inf"))

                pred_logits = v_emb @ s_emb.transpose(1,2) + offset[None,]
                bs,_,ns = pred_logits.shape
                _,_,h,w = pred_smasks.shape

                logits_idx_y = pred_logits.max(dim=1)[1]
                logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)[:,None].repeat(1, logits_idx_y.shape[1])
                logits_idx = torch.stack([logits_idx_x, logits_idx_y]).view(2,-1).tolist()

                pred_stexts = results['pred_stexts']
                pred_object_embs = pred_stexts[logits_idx].reshape(bs,ns,-1)
                pred_object_probs = results['pred_slogits'][logits_idx].softmax(dim=-1)
                pred_object_probs[:,-1] *= 0.001
                pred_object_probs, pred_object_ids = pred_object_probs.max(dim=-1)
                outputs = {"embeddings" : pred_object_embs[0]}
                pred_masks = F.interpolate(results['prev_mask'], size=images.tensor.shape[-2:], mode='bilinear', align_corners=False)[:,:,:batched_inputs[0]['image'].shape[-2], :batched_inputs[0]['image'].shape[-1]]
                
                acc_masks += [pred_masks]
                acc_scores += [pred_object_probs]
                acc_labels += [pred_object_ids]

            acc_masks = torch.cat(acc_masks, dim=1)[0]
            acc_scores = torch.cat(acc_scores, dim=0)
            acc_ids = torch.cat(acc_labels, dim=0)
            
            sorted_indices = torch.argsort(acc_scores, descending=True)
            
            _h, _w = h//4, w//4
            _acc_masks = F.interpolate(acc_masks[None], size=(_h, _w), mode='bilinear', align_corners=False)[0]
            _acc_masks = _acc_masks[sorted_indices] > 0
            acc_masks = acc_masks[sorted_indices]
            acc_scores = acc_scores[sorted_indices]
            acc_ids = acc_ids[sorted_indices]

            final_score = matrix_nms(_acc_masks.cuda(), acc_ids.cuda(), acc_scores.cuda())
            keep = final_score > 0.60
            pred_masks = acc_masks[keep] > 0.0
            pred_scores = final_score[keep]
            pred_ids = acc_ids[keep]
            pred_masks = resolve_mask_conflicts(pred_masks, pred_scores)

            # visualization
            # Need to uncomment: outputs.update(self.update_spatial_results(outputs))
            image = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()[:,:,::-1]
            visual_masks = pred_masks.float()
            visual_masks = visual_masks.cpu().numpy()
            visual_mask = VL.overlay_all_masks_to_image(image, visual_masks)
            cv2.imwrite("mask.png", visual_mask)
            import pdb; pdb.set_trace()

            image_pth_list += [image_pth]
            image_ori = Image.open(image_pth).convert('RGB')
            width, height = image_ori.size
            image_pth_to_hw[image_pth] = {"height": height, "width": width}

            if image_pth.replace("images", marked_folder.split('/')[-1]) in glob.glob(os.path.join(marked_folder, "*.JPG")):
                print("skip image {}".format(image_pth))
                continue

            marked_image, masks_dict = inference_semsam_m2m_auto(model_semsam, image_ori, [semsam_layer], 640, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])

            cv2.imwrite(os.path.join(marked_folder, image_pth.split("/")[-1]), marked_image[:,:,::-1])        
            masks_dict_pth = os.path.join(masks_folder, image_pth.split("/")[-1].replace(".JPG", ".pth"))
            torch.save(masks_dict, masks_dict_pth)

            marked_image_list += [marked_image]
            masks_dict_list += [masks_dict]
        print(image_pth)


_masks_folder = []
for mask_pth in sorted(glob.glob(os.path.join(masks_folder, "*.pth"))):
    image_name = mask_pth.split("/")[-1].split(".")[0]
    if image_name in test_images:
        _masks_folder += [mask_pth]

_marked_folder = []
for mask_pth in sorted(glob.glob(os.path.join(marked_folder, "*.JPG"))):
    image_name = mask_pth.split("/")[-1].split(".")[0]
    if image_name in test_images:
        _marked_folder += [mask_pth]
        
_image_pth_list = []
for mask_pth in image_pth_list:
    image_name = mask_pth.split("/")[-1].split(".")[0]
    if image_name in test_images:
        _image_pth_list += [mask_pth]

# Load masks_dict_list and marked_image_list
masks_dict_list = [torch.load(pth) for pth in sorted(_masks_folder)]
marked_image_list = [cv2.imread(pth) for pth in sorted(_marked_folder)]
json_file_name_list = [image_info["file_name"] for image_info in info["images"]]

global_id = 0
for idx, (masks_dict, marked_image, image_pth) in enumerate(zip(masks_dict_list, marked_image_list, sorted(_image_pth_list))):
    if image_pth.split("/")[-1] in json_file_name_list:
        print('skip image for gpt {}'.format(image_pth))
        continue

    trail = 0
    while trail < 3:
        # random draw two examples from encoded_image_list except idx.
        input_image_list = [marked_image]
        
        encoded_image_list = []
        for jdx in range(len(input_image_list)):
            image = add_image_marker(input_image_list[jdx], text=f"Image{jdx+1}")
            cv2.imwrite("marked_image.png", image)
            encoded_image = base64.b64encode(open("marked_image.png", 'rb').read()).decode('ascii')
            encoded_image_list += [encoded_image]
        
        output_label = call_gpt4o(system_msg2, [user_msg2], encoded_image_list)
        output_label = output_label.replace('json\n', '').replace('\\n', '\n').replace("```", "")

        try:
            segment_data = json.loads(output_label)
        except:
            trail += 1
            continue

        count_segment = 0
        acc_masks = []
        acc_ids = []
        segment_all_info = []
        local_2_global = {}
        for key, value in segment_data['Grounding'].items():
            try:
                mark = int(key)
                mask = torch.from_numpy(masks_dict[mark]).unsqueeze(0)

                segment_all_info += [{
                    "local_id": mark,
                    "global_id": global_id,
                    "sentence": value,
                }]
                acc_masks += [mask]
                acc_ids += [mark]
                local_2_global[mark] = global_id
                global_id += 1
                count_segment += 1
            except:
                print("Error in segment")
                continue
        
        if count_segment > 0:
            break
        else:
            trail += 1
            continue
    
    acc_masks = torch.cat(acc_masks, dim=0).float()
    acc_ids = torch.tensor(acc_ids)[:, None, None]
    acc_masks = acc_masks * acc_ids
    acc_masks = acc_masks.max(dim=0)[0].byte().cpu().numpy()
    
    mask_pth = os.path.join(mask_folder, image_pth.split("/")[-1].replace(".JPG", ".png"))
    cv2.imwrite(mask_pth, acc_masks)
    
    try:
        image_info = {
            "file_name": image_pth.split("/")[-1],
            "image_id": image_pth.split("/")[-1].split(".")[0],
            "height": image_pth_to_hw[image_pth]["height"],
            "width": image_pth_to_hw[image_pth]["width"],
            "segment_info": segment_all_info,
            "mask_pth": mask_pth,
        }
    except:
        print("Error in image_info")
        continue

    info["images"].append(image_info)
    json.dump(info, open(json_pth, "w"))
    print(idx, image_pth)