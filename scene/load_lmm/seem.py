import os
import cv2
import torch

def load_seem(args, embed_info, image_path):
    image_name = image_path.split("/")[-1]
    image_info = embed_info["seem"][image_name]
    segment_info = image_info["segment_info"]

    embed_pth = os.path.join("/".join(image_path.split('/')[:-2]), "/".join(image_info["emb_pth"].split('/')[-3:]))
    embeddings = torch.load(embed_pth)['pixel_embeds']
    n,c = embeddings.shape
    down_rate = 4
    
    mask_pth = os.path.join("/".join(image_path.split('/')[:-2]), "/".join(image_info["mask_pth"].split('/')[-3:]))
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
        "embeddings": gt_embeddings,
        "height": image_info["height"],
        "width": image_info["width"],
        "emb_height": h,
        "emb_width": w,
    }}
    return output