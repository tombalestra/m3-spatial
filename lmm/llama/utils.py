import json
import torch
from transformers import AutoTokenizer
from .llm.llama3_1 import LlamaForCausalLM_Path
from .llm.prompt import system_message

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = LlamaForCausalLM_Path.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="cuda", torch_dtype=torch.float16)

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

def delete_model():
    global model, tokenizer
    del model, tokenizer
    torch.cuda.empty_cache()

def compute_phrase_index(phrase1, phrase2):
    '''
    Start to build the replacing token: it will return the start and end index of the second message.
    '''
    phrase_ids1 = tokenizer.apply_chat_template(
        phrase1,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)[0]

    phrase_ids2 = tokenizer.apply_chat_template(
        phrase2,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)[0]

    def pad_or_cut_end(tensor1, tensor2):
        if tensor1.shape[0] > tensor2.shape[0]:
            tensor1 = tensor1[:tensor2.shape[0]]
        elif tensor1.shape[0] < tensor2.shape[0]:
            tensor1 = torch.cat([tensor1, -torch.ones(tensor2.shape[0] - tensor1.shape[0], dtype=tensor1.dtype, device=tensor1.device)])
        return tensor1
    
    def pad_or_cut_start(tensor1, tensor2):
        if tensor1.shape[0] > tensor2.shape[0]:
            tensor1 = tensor1[-tensor2.shape[0]:]
        elif tensor1.shape[0] < tensor2.shape[0]:
            tensor1 = torch.cat([-torch.ones(tensor2.shape[0] - tensor1.shape[0], dtype=tensor1.dtype, device=tensor1.device), tensor1])
        return tensor1

    phrase_ids1_ = pad_or_cut_end(phrase_ids1, phrase_ids2)
    _phrase_ids1 = pad_or_cut_start(phrase_ids1, phrase_ids2)

    # left align for start
    start = (phrase_ids1_ - phrase_ids2).nonzero()[0,0].item()
    # right align for end
    end = (_phrase_ids1 - phrase_ids2).nonzero()[-1,0].item() + 1
    return start, end

def extract_feature(message, layer=6):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message},
    ]

    _messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": ""},
    ]
    
    start_idx, end_idx = compute_phrase_index(_messages, messages)

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    
    with torch.no_grad():
        outputs = model.forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            cache_position=cache_position,        
        )
    
    hidden_states = torch.cat(outputs.hidden_states, dim=0) # 33, input_emb, hidden states, norm(last_states)
    embeddings = hidden_states[:, start_idx:end_idx]
    return embeddings

def parse_segment_data(input_string):
    try:
        input_string = input_string.replace('json\n', '').replace('\\n', '\n').replace("```", "")
        # Parse the input string as JSON
        parsed_data = json.loads(input_string)
        
        # Check if the parsed data is a list
        if isinstance(parsed_data, list):
            # Check if all elements in the list are dictionaries
            if all(isinstance(item, dict) for item in parsed_data):
                return parsed_data
            else:
                return False
        else:
            return False
    except json.JSONDecodeError:
        return False