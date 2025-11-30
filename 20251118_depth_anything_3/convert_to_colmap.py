#!/usr/bin/env python3
"""
Convert DA3 outputs to COLMAP format for traditional 3DGS training
"""
import numpy as np
import struct
from pathlib import Path

def write_cameras_binary(cameras, output_path):
    """Write COLMAP cameras.bin"""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('Q', len(cameras)))
        for camera_id, (model, width, height, params) in cameras.items():
            f.write(struct.pack('I', camera_id))
            f.write(struct.pack('i', model))  # PINHOLE = 1
            f.write(struct.pack('Q', width))
            f.write(struct.pack('Q', height))
            f.write(struct.pack(f'{len(params)}d', *params))

def write_images_binary(images, output_path):
    """Write COLMAP images.bin"""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('Q', len(images)))
        for image_id, (qvec, tvec, camera_id, name) in images.items():
            f.write(struct.pack('I', image_id))
            f.write(struct.pack('4d', *qvec))
            f.write(struct.pack('3d', *tvec))
            f.write(struct.pack('I', camera_id))
            f.write(name.encode('utf-8') + b'\x00')
            f.write(struct.pack('Q', 0))  # No 2D points

def write_points3D_binary(output_path):
    """Write empty COLMAP points3D.bin (we'll use depth maps instead)"""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('Q', 0))  # No 3D points

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])

def main():
    # Load DA3 results
    data = np.load('workspace/gallery/scene/exports/npz/results.npz')
    extrinsics = data['extrinsics']  # (N, 3, 4)
    intrinsics = data['intrinsics']  # (N, 3, 3)
    
    N = len(extrinsics)
    height, width = data['depth'].shape[1:3]
    
    print(f"Converting {N} images to COLMAP format...")
    print(f"Image size: {width}x{height}")
    
    # Create output directory
    output_dir = Path('colmap_output/sparse/0')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Symlink to original images folder
    images_dir = Path('colmap_output/input')
    if images_dir.exists():
        images_dir.unlink()
    images_dir.symlink_to(Path('images').resolve())

    # Get image filenames
    image_files = sorted(Path('images').glob('*.jpg')) + sorted(Path('images').glob('*.JPG')) + \
                  sorted(Path('images').glob('*.jpeg')) + sorted(Path('images').glob('*.JPEG')) + \
                  sorted(Path('images').glob('*.png')) + sorted(Path('images').glob('*.PNG'))
    
    # Prepare cameras (assume all images use same camera)
    cameras = {}
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    cx = intrinsics[0, 0, 2]
    cy = intrinsics[0, 1, 2]
    cameras[1] = (1, width, height, [fx, fy, cx, cy])  # PINHOLE model
    
    # Prepare images
    images = {}
    for i in range(N):
        R = extrinsics[i, :, :3]  # 3x3 rotation
        t = extrinsics[i, :, 3]   # 3x1 translation
        qvec = rotation_matrix_to_quaternion(R)
        images[i+1] = (qvec, t, 1, image_files[i].name)
    
    # Write COLMAP files
    write_cameras_binary(cameras, output_dir / 'cameras.bin')
    write_images_binary(images, output_dir / 'images.bin')
    write_points3D_binary(output_dir / 'points3D.bin')

    # Write project.ini
    with open(output_dir / 'project.ini', 'w') as f:
        f.write(f"log_to_stderr=0\n")
        f.write(f"log_level=2\n")
        f.write(f"database_path={output_dir.parent.parent / 'database.db'}\n")
        f.write(f"image_path={images_dir}\n")

    print(f"\n✓ COLMAP format created in: colmap_output/")
    print(f"  - sparse/0/cameras.bin")
    print(f"  - sparse/0/images.bin")
    print(f"  - sparse/0/points3D.bin")
    print(f"  - sparse/0/project.ini")
    print(f"  - input/ -> images/ ({N} images)")
    print(f"\nNow you can train 3DGS with:")
    print(f"  python train.py -s colmap_output/ --eval")

if __name__ == "__main__":
    main()

