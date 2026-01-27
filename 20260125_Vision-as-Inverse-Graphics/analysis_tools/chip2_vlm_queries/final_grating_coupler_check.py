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

prompt = """Look at this FULL photonic chip layout image very carefully.

CRITICAL QUESTION: Are there ANY grating couplers ANYWHERE in this image?

Grating couplers look like this:
- Periodic array of parallel lines (like teeth of a comb)
- Often at an angle
- Regular spacing between lines
- Used for optical fiber coupling

Please scan the ENTIRE image systematically:
1. Top edge - any grating couplers?
2. Bottom edge - any grating couplers?
3. Left edge - any grating couplers?
4. Right edge - any grating couplers?
5. Anywhere in the middle - any grating couplers?

For EACH location where you find a grating coupler:
- Describe its exact position (e.g., "bottom left corner", "right edge middle")
- Count how many teeth/lines it has
- Describe its orientation
- Estimate its size in pixels

If you find ZERO grating couplers, state: "ZERO grating couplers found in this image."

Be absolutely certain. This is critical."""

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
    max_tokens=2000,
    temperature=0.1
)

answer = response.choices[0].message.content
print("="*80)
print("FINAL GRATING COUPLER CHECK - FULL IMAGE:")
print("="*80)
print(answer)

