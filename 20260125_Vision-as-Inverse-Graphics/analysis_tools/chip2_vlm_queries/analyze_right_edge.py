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

right_edge_b64 = encode_image('source/regions/right_edge.png')

prompt = """This is a cropped region from the RIGHT EDGE of a photonic chip (70-100% of the full width).

Focus ONLY on this region. Tell me:

1. Are there grating couplers visible? (Grating couplers look like arrays of parallel lines/teeth, often at an angle)
   - If YES: How many? What orientation? Describe their appearance in detail. Count them carefully.
   - If NO: What structures ARE visible instead?

2. Are there electrode pads visible? (Metal contact pads, usually rectangular or T-shaped)
   - If YES: How many? What size? What positions?

3. What waveguide structures do you see?

4. List EVERY distinct structure type you can identify and COUNT each type.

Be very specific. Count carefully."""

response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{right_edge_b64}"}
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }],
    max_tokens=2000,
    temperature=0.1
)

answer = response.choices[0].message.content
print("="*80)
print("RIGHT EDGE REGION ANALYSIS:")
print("="*80)
print(answer)

