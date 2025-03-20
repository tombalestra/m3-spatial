import os
import cv2
import torch

def load_llama3(args, embed_info, image_path):
    image_name = image_path.split("/")[-1]
    image_info = embed_info["llama3"][image_name]

    segment_info = image_info["segment_info"]
    embed_pth = os.path.join("/".join(image_path.split('/')[:-2]), "/".join(image_info["emb_pth"].split('/')[-3:]))
    embeddings = torch.load(embed_pth, map_location='cpu')
    c = 4096
    down_rate = 4
    
    mask_pth = os.path.join("/".join(image_path.split('/')[:-2]), "/".join(image_info["mask_pth"].split('/')[-3:]))
    mask = cv2.imread(mask_pth, cv2.IMREAD_GRAYSCALE)
    h,w = mask.shape
    mask = cv2.resize(mask, (w//down_rate, h//down_rate), interpolation=cv2.INTER_NEAREST)
    mask = torch.from_numpy(mask)
    valid_mask = mask != 0
    
    local_ids = [x['local_id'] for x in segment_info]
    gt_embeddings = torch.zeros((h//down_rate,w//down_rate,c)).half()
    
    for local_id in local_ids:
        _emb = embeddings[local_id]
        gt_embeddings[mask==local_id] = _emb
    
    output = {"llama3": {
        "embeddings": gt_embeddings,
        "height": image_info["height"],
        "width": image_info["width"],
        "emb_height": h,
        "emb_width": w,
    }}
    return output