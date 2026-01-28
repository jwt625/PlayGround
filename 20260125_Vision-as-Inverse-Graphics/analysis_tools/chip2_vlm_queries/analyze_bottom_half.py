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

bottom_b64 = encode_image('source/regions/bottom_half.png')

prompt = """This is the BOTTOM HALF of a photonic chip layout.

I specifically need to know about GRATING COUPLERS.

Grating couplers are optical input/output structures that look like:
- Arrays of parallel lines/teeth (like a comb or grating)
- Often at an angle (e.g., 45 degrees)
- Usually periodic with regular spacing
- May appear as striped patterns
- Often located at chip edges for fiber coupling

Please:
1. Scan this entire bottom half region carefully
2. Are there ANY grating couplers visible? Look at ALL edges (left, right, bottom)
3. If YES: 
   - How many grating couplers?
   - Where exactly are they located?
   - What orientation/angle?
   - Describe their appearance (number of teeth, spacing, etc.)
4. If NO: State clearly "No grating couplers visible"

Be very careful and thorough. Grating couplers are critical optical I/O components."""

response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{bottom_b64}"}
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
print("BOTTOM HALF - GRATING COUPLER SEARCH:")
print("="*80)
print(answer)

