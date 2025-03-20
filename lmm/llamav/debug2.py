import requests
import torch
from PIL import Image
from transformers import AutoProcessor
from lmm.siglip.utils import visualize_feature
from .modeling_mllama import MllamaForConditionalGeneration

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# PIL image resize to 1120, 1120
image = image.resize((1120, 1120))

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    # get vision tokens from vision model
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
    ) # Act as FPN.

    x = cross_attention_states[:,1:,:]
    x = x.reshape(4, 40, 40, 4096)
    # Create the final output tensor
    output = torch.zeros(80, 80, 4096)
    # Assign the quadrants
    output[:40, :40] = x[0, :]  # Top-left (0)
    output[:40, 40:] = x[1, :]  # Top-right (1)
    output[40:, :40] = x[2, :]  # Bottom-left (2)
    output[40:, 40:] = x[3, :]  # Bottom-right (3)
    visualize_feature(output.float().cpu(), size=(80, 80))
    import pdb; pdb.set_trace()