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

I need you to describe the ELECTRODE ROUTING in extreme detail. The electrodes are the metal structures (likely the concentric rings around the resonators).

Please answer:

1. How many concentric electrode rings are there around EACH resonator? Count them carefully.

2. Starting from one electrode ring, describe EXACTLY how the metal trace routes to the pad:
   - Does it go straight up? Or does it bend?
   - How wide does the trace appear to be (in pixels or relative to the ring)?
   - How long is the trace?
   - What direction does it go?

3. Are there connections BETWEEN the electrode rings of different resonators? If yes, describe the path.

4. Where are the electrode pads located? Top? Bottom? Left? Right? What are their positions relative to the rings?

5. Do all three resonators have the same electrode routing pattern, or are they different?

6. Describe the electrode routing for the LEFTMOST ring in step-by-step detail: "The trace starts at [location], goes [direction] for [distance], then [bends/continues], and ends at [location]."

Be extremely specific. Use measurements in pixels if needed."""

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
print("ELECTRODE ROUTING DETAILED DESCRIPTION:")
print("="*80)
print(answer)

