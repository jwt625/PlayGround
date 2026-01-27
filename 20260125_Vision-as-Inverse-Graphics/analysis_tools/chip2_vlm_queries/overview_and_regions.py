import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

client = OpenAI(
    api_key=os.getenv('VLLM_API_KEY'),
    base_url="http://192.222.54.152:8000/v1"
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

target_b64 = encode_image('source/chip2.png')

# First, get overview and region recommendations
prompt_overview = """Look at this photonic chip layout image.

I need you to help me analyze it systematically by dividing it into regions.

Please:
1. Describe the overall layout structure (what's on the left, center, right, top, bottom)
2. Identify distinct functional regions that should be analyzed separately
3. For each region, provide bounding box coordinates as percentages (e.g., "left region: 0-30% width, 0-100% height")
4. Suggest which regions I should crop and analyze in detail

For example:
- "Left edge region (0-15% width): Contains input structures"
- "Center region (15-70% width): Contains main optical components"
- "Right edge region (70-100% width): Contains output structures"

Be specific about what functional areas exist and how to divide them for detailed analysis."""

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
                "text": prompt_overview
            }
        ]
    }],
    max_tokens=2000,
    temperature=0.1
)

answer = response.choices[0].message.content
print("="*80)
print("OVERVIEW AND REGION RECOMMENDATIONS:")
print("="*80)
print(answer)
print("\n" + "="*80)

# Now create crops based on typical PIC layout
img = Image.open('source/chip2.png')
width, height = img.size

# Create typical regions
regions = {
    'left_edge': (0, 0, int(width*0.2), height),
    'center': (int(width*0.2), 0, int(width*0.7), height),
    'right_edge': (int(width*0.7), 0, width, height),
    'top_half': (0, 0, width, int(height*0.5)),
    'bottom_half': (0, int(height*0.5), width, height)
}

os.makedirs('source/regions', exist_ok=True)

for region_name, bbox in regions.items():
    cropped = img.crop(bbox)
    cropped.save(f'source/regions/{region_name}.png')
    print(f"Created crop: source/regions/{region_name}.png (bbox: {bbox})")

