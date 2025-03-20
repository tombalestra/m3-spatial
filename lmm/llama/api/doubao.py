import os
from openai import OpenAI

# Initialize Doubao client
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY"),
)

def call_gpt4o(system, _inputs, _images):
    # Prepare the messages with system prompt, text inputs and images
    messages = [
        {
            "role": "system", 
            "content": system 
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text} for text in _inputs
            ] + [
                {
                    "type": "image_url",
                    "image_url": {
                         "url":  f"data:image/png;base64,{img}"
                    }
                } for img in _images
            ]
        }
    ]
    # Call the API
    response = client.chat.completions.create(
        model="doubao-1-5-vision-pro-32k-250115",
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    
    return response.choices[0].message.content
