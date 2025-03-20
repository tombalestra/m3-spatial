import os
import glob
import json
from PIL import Image
import torch
from transformers import AutoProcessor
from lmm.siglip.utils import visualize_feature
from .modeling_mllama import MllamaForConditionalGeneration

data_root = "/disk1/data/m3/data_v2/tabletop_v2"
input_folder = os.path.join(data_root, "images")
output_folder = os.path.join(data_root, "llamav")
embed_folder = os.path.join(output_folder, "embeds")
json_pth = os.path.join(data_root, "llamav_info.json")

if not os.path.exists(embed_folder):
    os.makedirs(embed_folder)

image_pths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

info = {
    "information": '''
    pixel_embeds: pixel features of last hidden states, in (h, w, c).
    image_embeds: image features of last hidden states, in (4, c).
    Note:
    1. All the features are before normalization.
    ''',
    "images": []
}

for image_pth in image_pths:
    with torch.no_grad():
        image_ori = Image.open(image_pth).convert('RGB')
        width, height = image_ori.size
        
        # Resize image to 1120x1120 as in the example
        image = image_ori.resize((1120, 1120))
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image:"}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors="pt").to(model.device)
        
        vision_outputs = model.vision_model(
            pixel_values=inputs.pixel_values,
            aspect_ratio_ids=inputs.aspect_ratio_ids,
            aspect_ratio_mask=inputs.aspect_ratio_mask,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        cross_attention_states = vision_outputs[0]
        cross_attention_states = model.multi_modal_projector(cross_attention_states).reshape(
            -1, cross_attention_states.shape[-2], model.hidden_size
        )

        cls_ = cross_attention_states[:, :1, :]
        img_ = cross_attention_states[:, 1:, :]
        img_ = img_.reshape(4, 40, 40, 4096)

        fea_ = torch.zeros(80, 80, 4096)
        fea_[:40, :40] = img_[0, :]
        fea_[:40, 40:] = img_[1, :]
        fea_[40:, :40] = img_[2, :]
        fea_[40:, 40:] = img_[3, :]

        output = {
            "pixel_embeds": fea_.cpu(),
            "image_embeds": cls_[:,0].cpu()
        }
        
        emb_pth = os.path.join(embed_folder, image_pth.split("/")[-1].replace(".jpg", ".emb"))
        torch.save(output, emb_pth)
        
        image_info = {
            "file_name": image_pth.split("/")[-1],
            "image_id": image_pth.split("/")[-1].split(".")[0],
            "height": height,
            "width": width,
            "emb_height": 80,
            "emb_width": 80,
            "emb_pth": emb_pth,
        }
        info["images"].append(image_info)
        visualize_feature(fea_.float().cpu(), size=(80, 80), filename="llama_vision_logits.png")

json.dump(info, open(json_pth, "w"))