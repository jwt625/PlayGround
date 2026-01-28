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

prompt = """Look at this photonic chip layout image (chip2.png) very carefully.

I need to understand the EXACT waveguide routing topology. Please answer these specific questions:

1. Starting from the LEFT side: Is there an input grating coupler? If yes, where exactly?

2. Trace the main bus waveguide: Does it run horizontally? Does it pass under/near the 3 rings? Does it continue all the way to the right edge?

3. At the RIGHT side: How many grating couplers are there? Are they arranged in a vertical array?

4. CRITICAL: How does the bus waveguide connect to the output grating couplers? Is there:
   a) A splitter (like 1x2 or 1x5 MMI)?
   b) Multiple separate waveguides from each ring to individual GCs?
   c) Some other routing pattern?

5. Do the rings have individual output waveguides going to separate GCs? Or do they all share the same bus?

6. Describe the complete optical path from input to output.

Be very specific about the routing topology."""

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
print("TARGET TOPOLOGY ANALYSIS:")
print("="*80)
print(answer)

