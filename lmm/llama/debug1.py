from PIL import Image
import cv2
import base64

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file

from .semantic_sam.tasks import inference_semsam_m2m_auto


semsam_cfg = "/data/xueyanz/code/GPT4-V-Bench/semantic_sam/configs/semantic_sam_only_sa-1b_swinL.yaml"
semsam_ckpt = "/data/xueyanz/code/GPT4-V-Bench/swinl_only_sam_many2many.pth"
opt_semsam = load_opt_from_config_file(semsam_cfg)
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()

image_pth = "/data/xueyanz/data/tandt/train/images/00001.jpg"
image1 = Image.open(image_pth)

text_size, hole_scale, island_scale=640,100,100
text, text_part, text_thresh = '','','0.0'
semantic = False
marked_image = inference_semsam_m2m_auto(model_semsam, image1, [3], text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])

cv2.imwrite("marked_image.png", marked_image[:,:,::-1])
encoded_image = base64.b64encode(open("marked_image.png", 'rb').read()).decode('ascii')

from .api.gpt4v import call_gpt4o
from .api.prompt import system_message, user_message

text = call_gpt4o(system_message, [user_message], [encoded_image])
print(text)