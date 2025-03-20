import torch
import torch.nn.functional as F
import clip
from PIL import Image

from .load import load
from ..siglip.utils import visualize_mask_logits


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load("ViT-L/14@336px", device=device)

image_ori = Image.open("/data/xueyanz/data/tandt/train/images/00001.jpg")
image = preprocess(image_ori).unsqueeze(0).to(device)

coco_classes = [x.replace('-other','').replace('-merged','').replace('-stuff','') for x in COCO_PANOPTIC_CLASSES]
text = clip.tokenize(coco_classes).to(device)

idx = coco_classes.index("train")

with torch.no_grad():
    image_features, pixel_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    pixel_features = pixel_features.view(1, 24, 24, -1).permute(0, 3, 1, 2)
    pixel_features = F.interpolate(pixel_features, size=(image_ori.size[1], image_ori.size[0]), mode='bilinear', align_corners=False)[0]
    pixel_features = pixel_features.permute(1, 2, 0)
    pixel_features = pixel_features.view(-1, pixel_features.size(-1))

    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    pixel_features = pixel_features / pixel_features.norm(dim=1, keepdim=True)
    
    logits_per_pixel = pixel_features @ text_features.T
    logits_per_pixel = logits_per_pixel.softmax(dim=-1)
    
    mask_logits = logits_per_pixel.view(image_ori.size[1], image_ori.size[0], -1).cpu()
    visualize_mask_logits(mask_logits[:,:,idx], image_ori, filename="mask_logits.png")