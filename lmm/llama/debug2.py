import torch
from transformers import AutoTokenizer
from .llm.llama3_1 import LlamaForCausalLM_Path
from .llm.prompt import system_message


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = LlamaForCausalLM_Path.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto", torch_dtype=torch.float16)

def call(message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0]
    return response

if __name__ == "__main__":
    response = call("Hello World.")
    print(tokenizer.decode(response, skip_special_tokens=True))