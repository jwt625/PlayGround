"""
Bad Apple!! Demo - GS Phase Mask Rendering

Renders the Bad Apple shadow art video using Weighted GS-computed phase masks.
Each frame becomes a target intensity pattern, and the phase mask is computed via WGS.

Output: badapple_phase.gif, badapple_intensity.gif
"""

import numpy as np
import cv2
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gs_algorithms import weighted_gs


# --- Configuration ---
GRID_SIZE = 256          # SLM resolution
GS_ITERATIONS = 50       # GS iterations per frame
FPS = 15                 # Output GIF frame rate
DURATION_SECONDS = 30    # Duration to process (seconds)
VIDEO_PATH = "badapple.mp4"


def create_gaussian_input(size: int, sigma: float = None) -> np.ndarray:
    """Create Gaussian input beam amplitude."""
    if sigma is None:
        sigma = size / 4
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


def extract_frames(video_path: str, target_size: int, target_fps: int, max_seconds: float) -> list[np.ndarray]:
    """
    Extract frames from video, resize and convert to binary.
    
    Returns list of binary target amplitude arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps
    
    print(f"Video: {video_fps:.1f} FPS, {total_frames} frames, {video_duration:.1f}s duration")
    
    # Calculate frame sampling
    frame_skip = max(1, int(video_fps / target_fps))
    max_frames = int(min(max_seconds, video_duration) * target_fps)
    
    print(f"Extracting: {max_frames} frames at {target_fps} FPS (skip every {frame_skip})")
    
    frames = []
    frame_idx = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to target size
            resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
            # Threshold to binary (Bad Apple is already high contrast)
            _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            # Normalize to [0, 1] amplitude
            amplitude = binary.astype(np.float64) / 255.0
            frames.append(amplitude)
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames


def render_frame(data: np.ndarray, title: str, cmap: str, vmin=None, vmax=None) -> np.ndarray:
    """Render a single frame as RGB array."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=64)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3]
    plt.close(fig)
    return img.copy()


def main():
    """Run Bad Apple GS demo."""
    print("Bad Apple!! GS Phase Mask Demo")
    print("=" * 40)

    # Extract frames from video
    target_frames = extract_frames(VIDEO_PATH, GRID_SIZE, FPS, DURATION_SECONDS)
    n_frames = len(target_frames)

    # Setup
    input_amp = create_gaussian_input(GRID_SIZE)

    # --- Estimate intensity range from a few sample frames ---
    sample_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    print(f"\nEstimating intensity range from {len(sample_indices)} sample frames...")

    sample_intensities = []
    for idx in sample_indices:
        result = weighted_gs(input_amp, target_frames[idx], GS_ITERATIONS, None)
        sample_intensities.append(result.reconstructed.ravel())

    all_samples = np.concatenate(sample_intensities)
    vmin_intensity = np.percentile(all_samples, 5)
    vmax_intensity = np.percentile(all_samples, 95)
    print(f"Intensity range (5-95%): [{vmin_intensity:.2f}, {vmax_intensity:.2f}]")

    # --- Main pass: compute and render all frames ---
    print(f"\nProcessing {n_frames} frames...")

    phase_frames = []
    intensity_frames = []
    current_phase = np.random.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))

    for i, target_amp in enumerate(target_frames):
        result = weighted_gs(input_amp, target_amp, GS_ITERATIONS, current_phase)
        current_phase = result.phase_mask  # Warm start

        phase_img = render_frame(
            result.phase_mask, f"Phase {i+1}/{n_frames}",
            'twilight', -np.pi, np.pi
        )
        intensity_img = render_frame(
            result.reconstructed, f"Intensity {i+1}/{n_frames}",
            'hot', vmin_intensity, vmax_intensity
        )

        phase_frames.append(phase_img)
        intensity_frames.append(intensity_img)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Frame {i+1}/{n_frames}")

    # Save GIFs
    print("\nSaving GIFs...")
    imageio.mimsave('badapple_phase.gif', phase_frames, fps=FPS, loop=0)
    imageio.mimsave('badapple_intensity.gif', intensity_frames, fps=FPS, loop=0)

    print("Saved: badapple_phase.gif")
    print("Saved: badapple_intensity.gif")
    print("Done!")


if __name__ == "__main__":
    main()

