from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

from .modeling_siglip import SiglipModel


model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# url = "/data/xueyanz/data/tandt/train/images/00001.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a photo of 2 cats", "a photo of 2 dogs"]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
logits_per_image = torch.sigmoid(logits_per_image) # these are the probabilities

probs = logits_per_image.max(dim=0, keepdim=True)[0]
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")