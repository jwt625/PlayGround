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
iter9_b64 = encode_image('output/chip2_iterations/iteration_09/layout_iter09.png')

prompt = """Compare these two images:
LEFT: Target design (chip2.png)
RIGHT: Generated design (iteration 9 - achieved 75% accuracy, the best so far)

This iteration 9 achieved 75% accuracy. I need to understand what it got RIGHT so I can build on it.

Please tell me:
1. What are the TOP 3 things that iteration 9 got CORRECT compared to the target?
2. What is the MAIN structural similarity between them?
3. If you had to keep the good parts of iteration 9 and fix only the most critical issues, what would you change?

Focus on what WORKS in iteration 9, not just what's wrong."""

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
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{iter9_b64}"}
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
print("WHY ITERATION 9 WORKED (75% accuracy):")
print("="*80)
print(answer)

