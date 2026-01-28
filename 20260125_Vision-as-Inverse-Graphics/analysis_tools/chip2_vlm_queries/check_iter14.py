import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('VLLM_API_KEY'),
    base_url="http://192.222.54.152:8000/v1"
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Encode iteration 14 image
iter14_b64 = encode_image('output/chip2_iterations/iteration_14/layout_iter14.png')

prompt = """Describe what you see in this photonic chip layout image in detail.

Count and describe:
1. How many ring resonators do you see?
2. How many electrode rings are over EACH resonator? (count carefully for one resonator)
3. How many grating couplers do you see? Where are they located?
4. Do you see any routing traces from electrodes to pads?
5. Do you see any contact pads?
6. What is the overall structure - is there a bus waveguide? Where does it go?

Be very specific with counts and locations."""

response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{iter14_b64}"}
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }],
    max_tokens=1500,
    temperature=0.1
)

answer = response.choices[0].message.content
print("="*80)
print("ITERATION 14 LAYOUT ANALYSIS:")
print("="*80)
print(answer)

