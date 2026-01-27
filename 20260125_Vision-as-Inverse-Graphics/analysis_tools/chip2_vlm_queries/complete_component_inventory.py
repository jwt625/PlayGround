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

target_b64 = encode_image('source/chip2.png')

prompt = """Look at this photonic chip layout image (chip2.png).

I need a COMPLETE INVENTORY of every component type on this chip.

For EACH component type, tell me:
1. Component name (e.g., "ring resonator", "grating coupler", "waveguide", "electrode pad", etc.)
2. EXACT quantity (count them carefully)
3. Orientation (e.g., "horizontal", "vertical", "rotated 90°", "facing left/right/up/down")
4. Location (e.g., "left edge", "right edge", "center", "top-right corner", with approximate coordinates if possible)
5. Size estimate in pixels (width x height, or diameter for circular components)

List EVERY distinct component type you can identify. Include:
- Ring resonators
- Waveguides (straight sections, bends)
- Grating couplers
- Electrode rings
- Electrode pads
- Routing traces
- Splitters/combiners
- Any other structures

For each component type, provide a table format like:

**Component: [Name]**
- Quantity: [number]
- Orientation: [description]
- Locations: [list each instance with position]
- Size: [width x height in pixels]

Be exhaustive and precise. Count everything."""

response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{target_b64}"}
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }],
    max_tokens=3000,
    temperature=0.1
)

answer = response.choices[0].message.content
print("="*80)
print("COMPLETE COMPONENT INVENTORY:")
print("="*80)
print(answer)

