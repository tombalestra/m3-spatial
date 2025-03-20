import requests
from PIL import Image

from transformers import AutoImageProcessor
from .modeling_dinov2 import Dinov2Model
from .utils import visualize_feature

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = "/data/xueyanz/data/tandt/train/images/00001.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(url)
width, height = image.size

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
# _height, _width = 384, int(width * 384 / height)
_height, _width = 384, 384
# processor.crop_size = {'height': _height, 'width': _width}
model = Dinov2Model.from_pretrained('facebook/dinov2-base')
patch_size = model.config.patch_size

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

embeddings = outputs.last_hidden_state
emb_height, emb_width = _height//patch_size, _width//patch_size
cls_, fea_ = embeddings[:, 0], embeddings[:, 1:].reshape(1, emb_height, emb_width, -1)

visualize_feature(fea_.detach(), size=(emb_height, emb_width), filename="dinov2_logits.png")