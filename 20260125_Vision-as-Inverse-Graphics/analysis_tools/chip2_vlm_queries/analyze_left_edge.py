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

left_edge_b64 = encode_image('source/regions/left_edge.png')

prompt = """This is a cropped region from the LEFT EDGE of a photonic chip (0-20% of the full width).

Focus ONLY on this region. Tell me:

1. Are there grating couplers visible? (Grating couplers look like arrays of parallel lines/teeth, often at an angle)
   - If YES: How many? What orientation? Describe their appearance in detail.
   - If NO: What structures ARE visible instead?

2. What waveguide structures do you see?
   - How many waveguides?
   - What direction do they run?
   - Where do they go (toward the center of the chip)?

3. Are there any input/output ports visible?

4. Describe every distinct structure you can identify in this left edge region.

Be very specific about what you actually see, not what you expect to see."""

response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{left_edge_b64}"}
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
print("LEFT EDGE REGION ANALYSIS:")
print("="*80)
print(answer)

