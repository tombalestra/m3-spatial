import os
import glob
import json

from PIL import Image
import cv2
import base64
from collections import defaultdict

import torch

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file

from .semantic_sam.tasks import inference_semsam_m2m_auto
from .api.doubao import call_gpt4o
from .api.prompt import system_labeling, material_message
from .utils import parse_segment_data, extract_feature

data_root = "/disk1/data/m3/data_v2/tabletop_v2"
input_folder = os.path.join(data_root, "images")
output_folder = os.path.join(data_root, "llama3")
embed_folder = os.path.join(output_folder, "embeds")
mask_folder = os.path.join(output_folder, "masks")
json_pth = os.path.join(data_root, "llama3_info.json")
device = "cuda"

if not os.path.exists(embed_folder):
    os.makedirs(embed_folder)

if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

image_pths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))

device = "cuda" if torch.cuda.is_available() else "cpu"
semsam_cfg = "/home/xueyan/code/som/GPT4-V-Bench/semantic_sam/configs/semantic_sam_only_sa-1b_swinL.yaml"
semsam_ckpt = "/home/xueyan/code/som/swinl_only_sam_many2many.pth"
opt_semsam = load_opt_from_config_file(semsam_cfg)
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
semsam_layer = 3

if json_pth is not None and os.path.exists(json_pth):
    info = json.load(open(json_pth))
else:
    info = {"information": '''
            1. global_id refers to the mark, this is corresponding to the segment_id, and the embedding index. \n
            2. 0 in the mask indicates the background, and the other number indicates the mark. \n
            ''',
            "images": []}

global_id = 0
for image_pth in image_pths:
    print(image_pth)
    json_file_name_list = [image_info["file_name"] for image_info in info["images"]]
    with torch.no_grad():
        if image_pth.split("/")[-1] in json_file_name_list:
            continue
        
        image_ori = Image.open(image_pth).convert('RGB')
        width, height = image_ori.size

        marked_image, masks_dict = inference_semsam_m2m_auto(model_semsam, image_ori, [semsam_layer], 640, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])

        cv2.imwrite("marked_image.png", marked_image[:,:,::-1])
        encoded_image = base64.b64encode(open("marked_image.png", 'rb').read()).decode('utf-8') #.decode('ascii') for gpt4o

        trail = 0
        while trail < 3:
            output_label = call_gpt4o(system_labeling, [material_message], [encoded_image])
            segment_data = parse_segment_data(output_label)
            
            if segment_data is False:
                trail += 1
                continue
            
            embeddings = {}
            count_segment = 0
            acc_masks = []
            acc_ids = []
            segment_info = []
            for segment in segment_data:
                try:
                    mark = int(segment["mark"])
                    _property = segment["segment_property"]
                    _value = segment["property_value"]
                    mask = torch.from_numpy(masks_dict[mark]).unsqueeze(0)
                    if mark not in embeddings:
                        embeddings[mark] = {}
                    embeddings[mark][_property] = extract_feature(_value)
                    segment_info += [{
                        "local_id": mark,
                        "global_id": global_id,
                        "segment_property": _property,
                        "property_value": _value,
                    }]
                    acc_masks += [mask]
                    acc_ids += [mark]
                    global_id += 1
                    count_segment += 1
                except:
                    print("Error in segment")
                    continue
            
            if count_segment > 0:
                break
            else:
                trail += 1
        
        acc_masks = torch.cat(acc_masks, dim=0).float()
        acc_ids = torch.tensor(acc_ids)[:, None, None]
        acc_masks = acc_masks * acc_ids
        acc_masks = acc_masks.max(dim=0)[0].byte().cpu().numpy()
        
        emb_pth = os.path.join(embed_folder, image_pth.split("/")[-1].replace(".jpg", ".emb"))
        torch.save(embeddings, emb_pth)
        
        mask_pth = os.path.join(mask_folder, image_pth.split("/")[-1].replace(".jpg", ".png"))
        cv2.imwrite(mask_pth, acc_masks)
        
        image_info = {
            "file_name": image_pth.split("/")[-1],
            "image_id": image_pth.split("/")[-1].split(".")[0],
            "height": height,
            "width": width,
            "segment_info": segment_info,
            "emb_pth": emb_pth,
            "mask_pth": mask_pth,
        }
        info["images"].append(image_info)
        json.dump(info, open(json_pth, "w"))