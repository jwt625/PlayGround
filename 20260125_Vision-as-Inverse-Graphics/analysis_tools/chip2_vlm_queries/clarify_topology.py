#!/usr/bin/env python3
"""Clarify the exact topology of chip2.png"""

import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

api_key = os.getenv('VLLM_API_KEY')
client = OpenAI(api_key=api_key, base_url='http://192.222.54.152:8000/v1')

image_b64 = encode_image('source/chip2.png')

prompt = '''Look at this PIC layout very carefully. I need to understand the exact topology:

1. Are the 3 rings DIRECTLY coupled to a single continuous horizontal bus waveguide that runs underneath/beside them?
   OR
2. Are the rings connected in series with separate waveguide segments between each ring?

3. Where exactly are the metal electrodes positioned relative to each ring?

4. How are the output grating couplers connected to the circuit?

Please be very specific about the waveguide routing and connectivity.'''

messages = [{
    'role': 'user',
    'content': [
        {'type': 'text', 'text': prompt},
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_b64}'}}
    ]
}]

response = client.chat.completions.create(
    model='Qwen3-VL-32B-Instruct',
    messages=messages,
    max_tokens=1500,
    temperature=0.3
)

print('='*80)
print('TOPOLOGY CLARIFICATION:')
print('='*80)
print(response.choices[0].message.content)

