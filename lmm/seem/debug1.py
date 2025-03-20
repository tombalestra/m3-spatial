import os
import time
import glob
import base64

from PIL import Image
import numpy as np
import cv2

import torch
from pycocotools import mask as maskUtils

from vlcore.utils.arguments import load_opt_from_config_file
from vlcore.utils.distributed import init_distributed 
from vlcore.utils.constants import COCO_PANOPTIC_CLASSES
from vlcore.modeling.BaseModel import BaseModel
from vlcore.modeling import build_model
from vlcore.xy_utils.label_sam2.infer_seem import inference_seem_pano


seem_cfg = "/data/xueyanz/code/vlcore_v2.0/vlcore/configs/seem/davitd5_unicl_lang_v1.yaml"
seem_ckpt = "/data/xueyanz/checkpoints/seem/seem_davit_d5.pt"

opt_seem = load_opt_from_config_file(seem_cfg)
opt_seem = init_distributed(opt_seem)
opt_seem['MODEL']['ENCODER']['NAME'] = 'transformer_encoder_deform'

model_seem = BaseModel(opt_seem, build_model(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

image_pth = "/data/xueyanz/data/tandt/train/images/00001.jpg"
image1 = Image.open(image_pth)

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
        marked_image1, masks1 = inference_seem_pano(model_seem, image1, anno_mode=["Mask", "Mark"], image_name="image1", alpha=0.5)

cv2.imwrite("image1.jpg", marked_image1[:,:,::-1])
exit()