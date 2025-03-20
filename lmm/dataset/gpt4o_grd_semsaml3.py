import os
import glob
import json
import cv2
import base64
import random
from PIL import Image
from collections import defaultdict

import torch

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file

# local
from lmm.llama.semantic_sam.tasks import inference_semsam_m2m_auto
from lmm.llama.api.gpt4v import call_gpt4o
from lmm.llama.utils import parse_segment_data, extract_feature

from .prompt import system_msg, user_msg
from .utils import add_image_marker


data_root = "/data/xueyanz/data/tandt/train"
input_folder = os.path.join(data_root, "images")
marked_folder = os.path.join(data_root, ".marked_images")
masks_folder = os.path.join(data_root, ".dict_masks")
mask_folder = os.path.join(data_root, "marked_masks2")
json_pth = os.path.join(data_root, "vlm_info2.json")
device = "cuda"

if not os.path.exists(marked_folder):
    os.makedirs(marked_folder)
if not os.path.exists(masks_folder):
    os.makedirs(masks_folder)
if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

image_pths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))

device = "cuda" if torch.cuda.is_available() else "cpu"
semsam_cfg = "/data/xueyanz/code/GPT4-V-Bench/semantic_sam/configs/semantic_sam_only_sa-1b_swinL.yaml"
semsam_ckpt = "/data/xueyanz/code/GPT4-V-Bench/swinl_only_sam_many2many.pth"
opt_semsam = load_opt_from_config_file(semsam_cfg)
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
semsam_layer = 3

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
for idx, image_pth in enumerate(image_pths):
    with torch.no_grad():
        image_pth_list += [image_pth]
        
        image_ori = Image.open(image_pth).convert('RGB')
        width, height = image_ori.size
        image_pth_to_hw[image_pth] = {"height": height, "width": width}

        if image_pth.replace("images", ".marked_images") in glob.glob(os.path.join(marked_folder, "*.jpg")):
            print("skip image {}".format(image_pth))
            continue

        marked_image, masks_dict = inference_semsam_m2m_auto(model_semsam, image_ori, [semsam_layer], 640, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])
        cv2.imwrite(os.path.join(marked_folder, image_pth.split("/")[-1]), marked_image[:,:,::-1])
        masks_dict_pth = os.path.join(masks_folder, image_pth.split("/")[-1].replace(".jpg", ".pth"))
        torch.save(masks_dict, masks_dict_pth)

        marked_image_list += [marked_image]
        masks_dict_list += [masks_dict]
    print(image_pth)

# Load masks_dict_list and marked_image_list
masks_dict_list = [torch.load(pth) for pth in sorted(glob.glob(os.path.join(masks_folder, "*.pth")))]
marked_image_list = [cv2.imread(pth) for pth in sorted(glob.glob(os.path.join(marked_folder, "*.jpg")))]
json_file_name_list = [image_info["file_name"] for image_info in info["images"]]

global_id = 0
for idx, (masks_dict, marked_image, image_pth) in enumerate(zip(masks_dict_list, marked_image_list, image_pth_list)):
    if image_pth.split("/")[-1] in json_file_name_list:
        print('skip image for gpt {}'.format(image_pth))
        continue

    trail = 0
    while trail < 3:
        # random draw two examples from encoded_image_list except idx.
        sampled_indices = random.sample([i for i in range(len(marked_image_list)) if i != idx], 2)
        negative_images = [marked_image_list[i] for i in sampled_indices]
        input_image_list = [marked_image] + negative_images
        
        encoded_image_list = []
        for jdx in range(len(input_image_list)):
            image = add_image_marker(input_image_list[jdx], text=f"Image{jdx+1}")
            cv2.imwrite("marked_image.png", image)
            encoded_image = base64.b64encode(open("marked_image.png", 'rb').read()).decode('ascii')
            encoded_image_list += [encoded_image]
        
        output_label = call_gpt4o(system_msg, [user_msg], encoded_image_list)
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
        for key, value in segment_data['Long Grounding'].items():
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
        
        segment_large_info = []
        for key, value in segment_data['Short Grounding'].items():
            try:
                mark = int(key)

                if mark not in acc_ids:
                    mask = torch.from_numpy(masks_dict[mark]).unsqueeze(0)
                    acc_masks += [mask]
                    acc_ids += [mark]
                    local_2_global[mark] = global_id

                    global_id += 1
                    count_segment += 1

                segment_large_info += [{
                    "local_id": mark,
                    "global_id": local_2_global[mark],
                    "sentence": value,
                }]
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
    
    mask_pth = os.path.join(mask_folder, image_pth.split("/")[-1].replace(".jpg", ".png"))
    cv2.imwrite(mask_pth, acc_masks)
    
    try:
        image_info = {
            "file_name": image_pth.split("/")[-1],
            "image_id": image_pth.split("/")[-1].split(".")[0],
            "height": image_pth_to_hw[image_pth]["height"],
            "width": image_pth_to_hw[image_pth]["width"],
            "segment_info_long": segment_all_info,
            "segment_info_short": segment_large_info,
            "caption_long": segment_data["Caption Long"],
            "caption_short": segment_data["Caption Short"],
            "mask_pth": mask_pth,
        }
    except:
        print("Error in image_info")
        continue

    info["images"].append(image_info)
    json.dump(info, open(json_pth, "w"))
    print(idx, image_pth)