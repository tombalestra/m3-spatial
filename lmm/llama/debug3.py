from PIL import Image
import cv2
import base64

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file

from .semantic_sam.tasks import inference_semsam_m2m_auto
from .api.gpt4v import call_gpt4o
from .api.prompt import system_labeling, material_message


def parse_segment_data(data):
    # Split the data into individual dictionary strings
    dict_strings = data.split("},\n{")
    
    # Clean up the first and last elements
    dict_strings[0] = dict_strings[0].lstrip("[{\n")
    dict_strings[-1] = dict_strings[-1].rstrip("\n}]")
    
    result = []
    for dict_string in dict_strings:
        # Replace escaped newlines with actual newlines
        dict_string = dict_string.replace("\\n", "\n")
        
        # Parse the dictionary string
        dict_data = {}
        for line in dict_string.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().strip('"')
                value = value.strip().strip(',').strip('"')
                dict_data[key] = value
        
        result.append(dict_data)
    
    return result

semsam_cfg = "/data/xueyanz/code/GPT4-V-Bench/semantic_sam/configs/semantic_sam_only_sa-1b_swinL.yaml"
semsam_ckpt = "/data/xueyanz/code/GPT4-V-Bench/swinl_only_sam_many2many.pth"
opt_semsam = load_opt_from_config_file(semsam_cfg)
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()

image_pth = "/data/xueyanz/data/tandt/train/images/00001.jpg"
image1 = Image.open(image_pth)

marked_image, masks_dict = inference_semsam_m2m_auto(model_semsam, image1, [3], 640, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])

cv2.imwrite("marked_image.png", marked_image[:,:,::-1])
encoded_image = base64.b64encode(open("marked_image.png", 'rb').read()).decode('ascii')

output_label = call_gpt4o(system_labeling, [material_message], [encoded_image])
segment_data = parse_segment_data(output_label)
import pdb; pdb.set_trace()