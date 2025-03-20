import os
import requests
import base64

import torch

from mimetypes import guess_type
from openai import AzureOpenAI


api_base = ""
api_key = os.getenv("")
deployment_name = 'gpt-4o' # gpt4v (not functioned), dalle3, gpt-4
api_version = '' # this might change in the future


client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

def call_gpt4o(system, _inputs, _images):
    messages = [
            {
                "role": "system", 
                "content": system 
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    } for text in _inputs
                ] + [
                    {
                        "type": "image",
                        "image": img,
                    } for img in _images
                ]
            }
    ]

    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content