import os
import glob
import json
from tqdm import tqdm
from PIL import Image
import torch

from transformers import AutoModel, AutoProcessor, AutoImageProcessor
from transformers.image_utils import load_image
from transformers.models.siglip.tokenization_siglip import SiglipTokenizer
from .modeling_siglip2 import SiglipModel
from .utils import visualize_feature, visualize_feature_kmeans, visualize_feature_gmm, visualize_mask_logits
from ..dinov2.utils import visualize_feature

data_root = "/disk1/data/m3/data_v2/tabletop_v2"
input_folder = os.path.join(data_root, "images")
output_folder = os.path.join(data_root, "siglip2")
embed_folder = os.path.join(output_folder, "embeds")
json_pth = os.path.join(data_root, "siglip2_info.json")
device = "cuda"

if not os.path.exists(embed_folder):
    os.makedirs(embed_folder)

image_pths = sorted(glob.glob(os.path.join(input_folder, "*.[jJ][pP][gG]")))

# Initialize model and processor
print("Loading model and processor...")
model = SiglipModel.from_pretrained("google/siglip2-so400m-patch16-512",
                                    device_map=device,)
processor = AutoImageProcessor.from_pretrained("google/siglip2-so400m-patch16-512", cache_dir="/home/xueyan/.cache/huggingface/hub")
patch_size = model.config.vision_config.patch_size

info = {"information": '''
        pixel_embeds: pixel features after the vision head, in (h, w, c). \n
        last_hidden_state: pixel features before the vision head, in (h, w, c). \n
        Note: \n
        1. All the features are before normalization. \n
        ''',
        "images": []}

print(f"Processing {len(image_pths)} images...")
for image_pth in tqdm(image_pths, desc="Processing images", unit="image"):
    with torch.no_grad():
        image = Image.open(image_pth).convert('RGB')
        width, height = image.size
        inputs = processor(text=["713"], images=image, padding="max_length", return_tensors="pt")
        inputs.to(device)
        _height, _width = inputs.pixel_values.shape[-2:]
        emb_height, emb_width = _height // patch_size, _width // patch_size

        with torch.autocast(device_type='cuda'):
            image_embeddings = model.get_image_features(**inputs)

        last_hidden_state = image_embeddings.last_hidden_state
        pooler_output = image_embeddings.pooler_output
        
        output = {
            "pixel_embeds": pooler_output.cpu(),
            "last_hidden_state": last_hidden_state.cpu(),
        }
        emb_pth = os.path.join(embed_folder, os.path.basename(image_pth).replace(".jpg", ".emb")).replace(".JPG", ".emb")
        torch.save(output, emb_pth)
        
        image_info = {
            "file_name": image_pth.split("/")[-1],
            "image_id": image_pth.split("/")[-1].split(".")[0],
            "height": height,
            "width": width,
            "emb_height": emb_height,
            "emb_width": emb_width,
            "emb_pth": emb_pth,
        }
        info["images"].append(image_info)
        # visualize_mask_logits(last_hidden_state, image, size=(_height//patch_size, _width//patch_size), filename="siglip_logits.png", alpha=0.5)

print("Saving metadata to JSON...")
json.dump(info, open(json_pth, "w"))
print("Processing complete!")