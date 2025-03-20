from PIL import Image
import requests

from transformers import CLIPProcessor
from .modeling_clip import CLIPModel
from ..siglip.utils import visualize_mask_logits

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
mask_embeds = outputs.image_embeds[1]
text_embeds = outputs.text_embeds

mask_embeds = mask_embeds / mask_embeds.norm(dim=-1, keepdim=True)
mask_embeds = mask_embeds.reshape(24, 24, 768)
mask_logits = mask_embeds @ text_embeds.T
mask_logits = mask_logits.softmax(dim=-1)
mask_logits = mask_logits[:, :, 0]

visualize_mask_logits(mask_logits.detach(), image, size=(24, 24), filename="mask_logits.png")
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities