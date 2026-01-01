#!/usr/bin/env python3
"""
Script to parse MOOSE thermal simulation CSV files and generate a GIF
showing temperature (T) vs position (x) for all time steps.
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import numpy as np

def main():
    # Find all CSV files except the one at t=0
    csv_files = sorted(glob.glob('therm_step03_out_t_sampler_*.csv'))
    csv_files = [f for f in csv_files if not f.endswith('0000.csv')]
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Determine global min/max for consistent axis scaling
    all_x = []
    all_T = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if not df.empty:
            all_x.extend(df['x'].values)
            all_T.extend(df['T'].values)
    
    x_min, x_max = min(all_x), max(all_x)
    T_min, T_max = min(all_T), max(all_T)
    
    # Add some padding to the limits
    x_padding = (x_max - x_min) * 0.05
    T_padding = (T_max - T_min) * 0.05
    
    x_lim = [x_min - x_padding, x_max + x_padding]
    T_lim = [T_min - T_padding, T_max + T_padding]
    
    # Create temporary directory for frames
    os.makedirs('temp_frames', exist_ok=True)
    
    frame_files = []
    
    # Generate plots for each time step
    for idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        
        if df.empty:
            continue
        
        # Extract time step number from filename
        time_step = int(csv_file.split('_')[-1].replace('.csv', ''))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['x'], df['T'], 'b-', linewidth=2, marker='o', markersize=4)
        
        # Set labels
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('T (K)', fontsize=12)
        
        # Set consistent limits
        ax.set_xlim(x_lim)
        ax.set_ylim(T_lim)
        
        # Enable grid
        ax.grid(True, alpha=0.3)
        
        # Add time step info as text
        ax.text(0.02, 0.98, f'Time step: {time_step}', 
                transform=ax.transAxes, 
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save frame
        frame_file = f'temp_frames/frame_{idx:04d}.png'
        plt.savefig(frame_file, dpi=100)
        plt.close()
        
        frame_files.append(frame_file)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(csv_files)} files")
    
    print(f"All {len(frame_files)} frames generated")
    
    # Create GIF
    print("Creating GIF...")
    frames = [Image.open(frame) for frame in frame_files]
    
    output_file = 'temperature_evolution.gif'
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # milliseconds per frame
        loop=0
    )
    
    print(f"GIF saved as: {output_file}")
    
    # Clean up temporary frames
    print("Cleaning up temporary frames...")
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir('temp_frames')
    
    print("Done!")

if __name__ == '__main__':
    main()

