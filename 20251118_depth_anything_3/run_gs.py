#!/usr/bin/env python3
"""
Run DA3 with 3DGS export enabled
"""
import sys
from pathlib import Path
from depth_anything_3.api import DepthAnything3

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_gs.py <images_dir> [export_dir]")
        sys.exit(1)
    
    images_dir = Path(sys.argv[1])
    export_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("workspace/gallery/scene")
    
    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist")
        sys.exit(1)
    
    # Get all images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(sorted(images_dir.glob(ext)))
    
    print(f"Found {len(image_paths)} images")
    print(f"Loading DA3 NESTED-GIANT-LARGE model...")
    
    # Load model
    model = DepthAnything3.from_pretrained(
        'depth-anything/DA3NESTED-GIANT-LARGE',
        cache_dir='./models'
    ).to('cuda').eval()
    
    print(f"Running inference with 3DGS export...")
    
    # Run inference with 3DGS enabled
    prediction = model.inference(
        image=[str(p) for p in image_paths],
        export_dir=str(export_dir),
        export_format="npz-glb-gs_ply-gs_video",
        infer_gs=True,  # Enable 3DGS branch
        process_res=504,
        align_to_input_ext_scale=True,
        conf_thresh_percentile=40.0,
        num_max_points=1000000,
        show_cameras=True,
    )
    
    print(f"\n✓ Done!")
    print(f"✓ Output: {export_dir}")
    print(f"✓ 3DGS file: {export_dir}/gs_ply/0000.ply")
    print(f"✓ Video: {export_dir}/gs_video/")

if __name__ == "__main__":
    main()

