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

# Encode both images
target_b64 = encode_image('source/chip2.png')
current_b64 = encode_image('output/chip2_iterations/iteration_09/layout_iter09.png')

# Detailed comparison queries
queries = [
    {
        "name": "overall_differences",
        "prompt": """Compare these two photonic chip layouts in detail. 
        
LEFT IMAGE: Target design
RIGHT IMAGE: Current generated design

List EVERY specific difference you can see between them. Be extremely detailed about:
1. Component counts (how many of each type in each image?)
2. Spatial arrangement differences
3. Missing components in the right image
4. Extra components in the right image
5. Size/dimension differences
6. Structural differences in how components are connected

Be specific with locations (left/right/top/bottom/center) and counts."""
    },
    {
        "name": "electrode_structure",
        "prompt": """Focus ONLY on the electrode structures (the circular/ring patterns over the ring resonators).

LEFT IMAGE: Target design
RIGHT IMAGE: Current generated design

Describe in detail:
1. How many electrode rings are visible over each resonator in the LEFT image?
2. How many electrode rings are visible over each resonator in the RIGHT image?
3. What is the pattern/structure of electrodes in the LEFT image? (concentric rings? other shapes?)
4. What is the pattern/structure of electrodes in the RIGHT image?
5. Are there any routing traces connecting electrodes to pads in the LEFT image? Where?
6. Are there any routing traces connecting electrodes to pads in the RIGHT image?
7. What specific electrode features are MISSING in the right image?"""
    },
    {
        "name": "waveguide_topology",
        "prompt": """Focus ONLY on the waveguide structures (bus waveguides, routing, connections).

LEFT IMAGE: Target design
RIGHT IMAGE: Current generated design

Describe in detail:
1. How is the main bus waveguide structured in the LEFT image?
2. How is the main bus waveguide structured in the RIGHT image?
3. How do the rings connect to the bus in the LEFT image?
4. How do the rings connect to the bus in the RIGHT image?
5. What happens at the end of the bus in the LEFT image? (splitter? branches?)
6. What happens at the end of the bus in the RIGHT image?
7. What specific waveguide routing is MISSING or WRONG in the right image?"""
    },
    {
        "name": "grating_couplers",
        "prompt": """Focus ONLY on the grating couplers (the structures for fiber coupling).

LEFT IMAGE: Target design
RIGHT IMAGE: Current generated design

Count and describe:
1. How many grating couplers in the LEFT image? Where are they located?
2. How many grating couplers in the RIGHT image? Where are they located?
3. What is the spacing between grating couplers in the LEFT image?
4. What is the spacing between grating couplers in the RIGHT image?
5. How are they arranged (vertical array? horizontal? other pattern)?
6. What is WRONG about the grating couplers in the right image?"""
    },
    {
        "name": "dimensional_analysis",
        "prompt": """Estimate the actual dimensions and spacings.

LEFT IMAGE: Target design
RIGHT IMAGE: Current generated design

For the LEFT image, estimate:
1. Ring resonator radius (in micrometers if possible)
2. Spacing between adjacent rings
3. Coupling gap between ring and bus waveguide
4. Waveguide widths
5. Electrode ring spacing

For the RIGHT image, estimate the same parameters.

What dimensional mismatches do you see?"""
    }
]

results = {}

for query in queries:
    print(f"\n{'='*80}")
    print(f"Query: {query['name']}")
    print(f"{'='*80}")
    
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
                    "image_url": {"url": f"data:image/png;base64,{current_b64}"}
                },
                {
                    "type": "text",
                    "text": query['prompt']
                }
            ]
        }],
        max_tokens=2000,
        temperature=0.1
    )
    
    answer = response.choices[0].message.content
    results[query['name']] = answer
    
    print(answer)
    print()

# Save results
output_path = 'output/chip2_iterations/detailed_comparison_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print(f"Results saved to: {output_path}")
print(f"{'='*80}")

