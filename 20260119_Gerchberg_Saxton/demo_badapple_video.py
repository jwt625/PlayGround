"""
Bad Apple!! Full Video Demo - GS Phase Mask Rendering

Renders the full Bad Apple video with side-by-side intensity and phase mask.
Outputs frames for ffmpeg to combine with audio.

Output: frames/*.png -> badapple_gs_demo.mp4
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gs_algorithms import weighted_gs


# --- Configuration ---
GRID_SIZE = 256          # SLM resolution
GS_ITERATIONS = 50       # GS iterations per frame
FPS = 10                 # Output frame rate (full video ~219s -> ~2190 frames)
VIDEO_PATH = "badapple.mp4"
FRAMES_DIR = "frames"


def create_gaussian_input(size: int, sigma: float = None) -> np.ndarray:
    """Create Gaussian input beam amplitude."""
    if sigma is None:
        sigma = size / 4
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


def extract_frames(video_path: str, target_size: int, target_fps: int) -> list[np.ndarray]:
    """Extract all frames from video, resize and convert to binary."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps
    
    print(f"Video: {video_fps:.1f} FPS, {total_frames} frames, {video_duration:.1f}s duration")
    
    frame_skip = max(1, int(video_fps / target_fps))
    max_frames = int(video_duration * target_fps)
    
    print(f"Extracting: ~{max_frames} frames at {target_fps} FPS")
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
            _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            amplitude = binary.astype(np.float64) / 255.0
            frames.append(amplitude)
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames


def render_sidebyside_frame(intensity: np.ndarray, phase: np.ndarray, 
                             frame_idx: int, n_frames: int,
                             vmin_int: float, vmax_int: float) -> np.ndarray:
    """Render side-by-side frame: intensity (left) + phase (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    
    # Left: Intensity
    axes[0].imshow(intensity, cmap='hot', vmin=vmin_int, vmax=vmax_int)
    axes[0].set_title(f"Reconstructed Intensity", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Right: Phase
    axes[1].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"Phase Mask", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    fig.suptitle(f"Bad Apple!! GS Demo - Frame {frame_idx+1}/{n_frames}", 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3]
    plt.close(fig)
    return img.copy()


def main():
    """Run full Bad Apple video demo."""
    print("Bad Apple!! Full Video GS Demo")
    print("=" * 50)
    
    # Create frames directory
    os.makedirs(FRAMES_DIR, exist_ok=True)
    
    # Extract frames
    target_frames = extract_frames(VIDEO_PATH, GRID_SIZE, FPS)
    n_frames = len(target_frames)
    
    # Setup
    input_amp = create_gaussian_input(GRID_SIZE)
    
    # Estimate intensity range from sample frames
    sample_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    print(f"\nEstimating intensity range from {len(sample_indices)} samples...")
    
    sample_intensities = []
    for idx in sample_indices:
        result = weighted_gs(input_amp, target_frames[idx], GS_ITERATIONS, None)
        sample_intensities.append(result.reconstructed.ravel())
    
    all_samples = np.concatenate(sample_intensities)
    vmin_int = np.percentile(all_samples, 5)
    vmax_int = np.percentile(all_samples, 95)
    print(f"Intensity range (5-95%): [{vmin_int:.2f}, {vmax_int:.2f}]")
    
    # Process all frames
    print(f"\nProcessing {n_frames} frames...")
    current_phase = np.random.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))
    
    for i, target_amp in enumerate(target_frames):
        result = weighted_gs(input_amp, target_amp, GS_ITERATIONS, current_phase)
        current_phase = result.phase_mask
        
        frame_img = render_sidebyside_frame(
            result.reconstructed, result.phase_mask,
            i, n_frames, vmin_int, vmax_int
        )
        
        # Save as PNG
        frame_path = os.path.join(FRAMES_DIR, f"frame_{i:05d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))
        
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Frame {i+1}/{n_frames}")
    
    print(f"\nFrames saved to {FRAMES_DIR}/")
    print(f"\nTo create final video, run:")
    print(f"  ./make_video.sh")


if __name__ == "__main__":
    main()

