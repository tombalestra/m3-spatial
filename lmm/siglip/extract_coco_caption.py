from PIL import Image
import requests
import json
from transformers import AutoProcessor, AutoModel
import torch
import os
from tqdm import tqdm

from .modeling_siglip import SiglipModel


model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384", device_map='cuda')
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
patch_size = model.config.vision_config.patch_size

caption_json = "/data/xueyanz/data/coco/annotations/captions_val2017.json"
coco_caption = json.load(open(caption_json, "r"))
coco_image_root = "/data/xueyanz/data/coco/val2017"

embed_folder = "/data/xueyanz/data/3dgs/coco/siglip"
os.makedirs(embed_folder, exist_ok=True)
embed_info = "/data/xueyanz/data/3dgs/coco/siglip_info.json"

info = {"information": '''
        pixel_embeds: pixel features after the vision head, in (h, w, c). \n
        text_embeds: text features after the vision head, in (h, w, c). \n
        Note: \n
        1. All the features are before normalization. \n
        ''',
        "images": []}

# Initialize progress bar
total_annotations = len(coco_caption["annotations"])
progress_bar = tqdm(coco_caption["annotations"], 
                   total=total_annotations,
                   desc="Processing COCO images",
                   unit="image")

image_id_list = []
with torch.no_grad():
    for annot in progress_bar:
        image_id = annot['image_id']
        caption = annot['caption']
        
        if image_id in image_id_list:
            continue
        
        image_name = f"{image_id:012d}.jpg"
        image_path = f"{coco_image_root}/{image_name}"
        emb_pth = f"{embed_folder}/{image_name.split('.')[0]}.emb"
        
        if os.path.exists(emb_pth):
            print(f"Embedding for {image_name} already exists")
            image_id_list += [image_id]
            continue
        
        # Update progress bar description with current image
        progress_bar.set_description(f"Processing {image_name}")
        
        image = Image.open(image_path)
        width, height = image.size
        try:
            inputs = processor(text=[caption], images=image, padding="max_length", return_tensors="pt")
        except:
            print(f"Error processing {image_name}")
            continue
        inputs.to("cuda")

        _height, _width = inputs.pixel_values.shape[-2:]
        emb_height, emb_width = _height // patch_size, _width // patch_size
        with torch.autocast(device_type='cuda'):
            outputs = model(**inputs)
        
        pixel_embeds = outputs.vision_model_output.pooler_output
        text_embeds = outputs.text_model_output.pooler_output
        
        embeddings = {
            "pixel_embeds": pixel_embeds.cpu(),
            "text_embeds": text_embeds.cpu(),
        }
        torch.save(embeddings, emb_pth)

        image_info = {
            "file_name": image_name,
            "image_id": image_id,
            "height": height,
            "width": width,
            "emb_height": emb_height,
            "emb_width": emb_width,
            "emb_pth": emb_pth,
        }
        info["images"].append(image_info)
        image_id_list += [image_id]


json.dump(info, open(embed_info, "w"))