import os
import torch

def load_llamav(args, embed_info, image_path):
    image_name = image_path.split("/")[-1]
    image_info = embed_info["llamav"][image_name]
    embed_pth = os.path.join("/".join(image_path.split('/')[:-2]), "/".join(image_info["emb_pth"].split('/')[-3:]))
    embeddings = torch.load(embed_pth)['pixel_embeds']
    output = {"llamav": {
        "embeddings": embeddings,
        "height": image_info["height"],
        "width": image_info["width"],
        "emb_height": image_info["emb_height"],
        "emb_width": image_info["emb_width"],
    }}
    return output