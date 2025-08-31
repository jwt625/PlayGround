#!/usr/bin/env python3
"""
YouTube Metadata Analysis and Visualization Script

This script analyzes scraped YouTube metadata and creates various visualizations
including comments vs views, video duration analysis, and upload date trends.

Features:
- Robust parsing that skips missing fields
- Maintains ordering of the original list
- Extracts exact view counts from preciseDate field
- Converts duration to seconds
- Parses upload dates from preciseDate field
- Creates scatter plots and other visualizations

Usage:
    python analyze_metadata.py [json_file_path]
"""

import json
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os
from pathlib import Path

# Set font size to 16 for all text elements
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16
})


def parse_view_count(precise_date_str):
    """
    Extract exact view count from preciseDate string.
    
    Examples:
    "12,827,138 views • Premiered Jul 11, 2025 • #Civilization #Minecraft" -> 12827138
    "222,671 views • Aug 16, 2025 • #DIY #woodworking #narwalflow" -> 222671
    """
    if not precise_date_str:
        return None
    
    # Look for pattern like "12,827,138 views" or "222K views" etc.
    view_match = re.search(r'([\d,]+(?:\.\d+)?)\s*([KMB])?\s*views', precise_date_str, re.IGNORECASE)
    if view_match:
        number_str = view_match.group(1).replace(',', '')
        multiplier_str = view_match.group(2)
        
        try:
            number = float(number_str)
            
            # Apply multiplier if present
            if multiplier_str:
                multiplier_str = multiplier_str.upper()
                if multiplier_str == 'K':
                    number *= 1000
                elif multiplier_str == 'M':
                    number *= 1000000
                elif multiplier_str == 'B':
                    number *= 1000000000
            
            return int(number)
        except ValueError:
            return None
    
    return None


def parse_comment_count(comment_count_str):
    """
    Extract comment count from commentCount string.
    
    Examples:
    "506 Comments" -> 506
    "1,964 Comments" -> 1964
    "0 Comments" -> 0
    """
    if not comment_count_str:
        return None
    
    # Look for pattern like "506 Comments" or "1,964 Comments"
    comment_match = re.search(r'([\d,]+)\s*Comments?', comment_count_str, re.IGNORECASE)
    if comment_match:
        try:
            return int(comment_match.group(1).replace(',', ''))
        except ValueError:
            return None
    
    return None


def parse_duration_to_seconds(duration_str):
    """
    Convert duration string to seconds.
    
    Examples:
    "2:34:00" -> 9240 seconds
    "22:07" -> 1327 seconds
    "0:14" -> 14 seconds
    """
    if not duration_str:
        return None
    
    # Split by colon and convert to seconds
    parts = duration_str.split(':')
    try:
        if len(parts) == 3:  # H:M:S
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # M:S
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 1:  # Just seconds
            return int(parts[0])
    except ValueError:
        return None
    
    return None


def parse_upload_date(precise_date_str):
    """
    Extract upload date from preciseDate string.
    
    Examples:
    "12,827,138 views • Premiered Jul 11, 2025 • #Civilization #Minecraft" -> datetime(2025, 7, 11)
    "222,671 views • Aug 16, 2025 • #DIY #woodworking #narwalflow" -> datetime(2025, 8, 16)
    """
    if not precise_date_str:
        return None
    
    # Look for date patterns like "Jul 11, 2025" or "Aug 16, 2025"
    date_patterns = [
        r'([A-Za-z]{3})\s+(\d{1,2}),\s+(\d{4})',  # Jul 11, 2025
        r'([A-Za-z]{3})\s+(\d{1,2})\s+(\d{4})',   # Jul 11 2025
        r'(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})',   # 11 Jul 2025
    ]
    
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    for pattern in date_patterns:
        match = re.search(pattern, precise_date_str, re.IGNORECASE)
        if match:
            try:
                if pattern == date_patterns[0] or pattern == date_patterns[1]:  # Month Day Year
                    month_str, day_str, year_str = match.groups()
                    month = month_map.get(month_str.lower()[:3])
                    day = int(day_str)
                    year = int(year_str)
                else:  # Day Month Year
                    day_str, month_str, year_str = match.groups()
                    month = month_map.get(month_str.lower()[:3])
                    day = int(day_str)
                    year = int(year_str)
                
                if month:
                    return datetime(year, month, day)
            except (ValueError, TypeError):
                continue
    
    return None


def load_and_parse_metadata(json_file_path):
    """
    Load JSON file and parse metadata into a structured format.
    
    Returns:
        pandas.DataFrame: Parsed metadata with columns for analysis
    """
    print(f"Loading metadata from: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} videos")
    
    # Parse each video's metadata
    parsed_data = []
    for i, video in enumerate(data):
        parsed_video = {
            'index': i,  # Maintain original ordering
            'video_id': video.get('videoId'),
            'title': video.get('title'),
            'channel': video.get('channel'),
            'url': video.get('url'),
        }
        
        # Parse view count from preciseDate
        precise_date = video.get('preciseDate', '')
        parsed_video['view_count'] = parse_view_count(precise_date)
        
        # Parse comment count
        comment_count_str = video.get('commentCount', '')
        parsed_video['comment_count'] = parse_comment_count(comment_count_str)
        
        # Parse duration to seconds
        duration_str = video.get('duration', '')
        parsed_video['duration_seconds'] = parse_duration_to_seconds(duration_str)
        
        # Parse upload date
        parsed_video['upload_date'] = parse_upload_date(precise_date)
        
        # Store raw fields for reference
        parsed_video['raw_precise_date'] = precise_date
        parsed_video['raw_comment_count'] = comment_count_str
        parsed_video['raw_duration'] = duration_str
        
        parsed_data.append(parsed_video)
    
    df = pd.DataFrame(parsed_data)
    
    # Print parsing statistics
    print(f"\nParsing Statistics:")
    print(f"View counts parsed: {df['view_count'].notna().sum()}/{len(df)} ({df['view_count'].notna().mean()*100:.1f}%)")
    print(f"Comment counts parsed: {df['comment_count'].notna().sum()}/{len(df)} ({df['comment_count'].notna().mean()*100:.1f}%)")
    print(f"Durations parsed: {df['duration_seconds'].notna().sum()}/{len(df)} ({df['duration_seconds'].notna().mean()*100:.1f}%)")
    print(f"Upload dates parsed: {df['upload_date'].notna().sum()}/{len(df)} ({df['upload_date'].notna().mean()*100:.1f}%)")
    
    return df


def create_comments_vs_views_plot(df, output_dir):
    """Create scatter plot of comments vs views."""
    # Filter data where both comments and views are available
    plot_data = df.dropna(subset=['comment_count', 'view_count'])

    if len(plot_data) == 0:
        print("No data available for comments vs views plot")
        return

    plt.figure(figsize=(12, 8))

    # Create scatter plot
    plt.scatter(plot_data['view_count'], plot_data['comment_count'],
                alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    # Use log scale for better visualization
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('View Count (log scale)')
    plt.ylabel('Comment Count (log scale)')
    plt.title(f'Comments vs Views\n({len(plot_data)} videos with both metrics)')
    plt.grid(True, alpha=0.3)

    # Add trend line
    if len(plot_data) > 1:
        # Filter out zero values for log calculation
        valid_data = plot_data[(plot_data['view_count'] > 0) & (plot_data['comment_count'] > 0)]

        if len(valid_data) > 1:
            log_views = np.log10(valid_data['view_count'])
            log_comments = np.log10(valid_data['comment_count'])

            # Fit linear regression on log scale
            coeffs = np.polyfit(log_views, log_comments, 1)
            trend_line = np.poly1d(coeffs)

            # Calculate R-squared
            log_comments_pred = trend_line(log_views)
            ss_res = np.sum((log_comments - log_comments_pred) ** 2)
            ss_tot = np.sum((log_comments - np.mean(log_comments)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            x_trend = np.logspace(np.log10(valid_data['view_count'].min()),
                                 np.log10(valid_data['view_count'].max()), 100)
            y_trend = 10 ** trend_line(np.log10(x_trend))

            plt.plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2,
                    label=f'Trend (slope: {coeffs[0]:.3f}, R²: {r_squared:.3f})')
            plt.legend()

            print(f"Trend analysis:")
            print(f"  Slope: {coeffs[0]:.3f} (log-log scale)")
            print(f"  R-squared: {r_squared:.3f}")
            print(f"  Data points used: {len(valid_data)}")

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'comments_vs_views.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()  # Close the figure instead of showing it


def create_duration_analysis_plots(df, output_dir):
    """Create plots analyzing video duration."""
    duration_data = df.dropna(subset=['duration_seconds'])

    if len(duration_data) == 0:
        print("No duration data available")
        return

    # Convert seconds to minutes for better readability
    duration_data = duration_data.copy()
    duration_data['duration_minutes'] = duration_data['duration_seconds'] / 60

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Duration histogram
    ax1.hist(duration_data['duration_minutes'], bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Duration (minutes)')
    ax1.set_ylabel('Number of Videos')
    ax1.set_title(f'Video Duration Distribution\n({len(duration_data)} videos)')
    ax1.grid(True, alpha=0.3)

    # 2. Duration vs Views
    views_duration_data = duration_data.dropna(subset=['view_count'])
    if len(views_duration_data) > 0:
        ax2.scatter(views_duration_data['duration_minutes'], views_duration_data['view_count'],
                   alpha=0.6, s=20)
        ax2.set_xlabel('Duration (minutes)')
        ax2.set_ylabel('View Count')
        ax2.set_yscale('log')
        ax2.set_title('Duration vs Views')
        ax2.grid(True, alpha=0.3)

    # 3. Duration vs Comments
    comments_duration_data = duration_data.dropna(subset=['comment_count'])
    if len(comments_duration_data) > 0:
        ax3.scatter(comments_duration_data['duration_minutes'], comments_duration_data['comment_count'],
                   alpha=0.6, s=20, color='orange')
        ax3.set_xlabel('Duration (minutes)')
        ax3.set_ylabel('Comment Count')
        ax3.set_yscale('log')
        ax3.set_title('Duration vs Comments')
        ax3.grid(True, alpha=0.3)

    # 4. Duration statistics by ranges
    duration_ranges = [(0, 5), (5, 15), (15, 30), (30, 60), (60, float('inf'))]
    range_labels = ['0-5 min', '5-15 min', '15-30 min', '30-60 min', '60+ min']
    range_counts = []

    for min_dur, max_dur in duration_ranges:
        if max_dur == float('inf'):
            count = len(duration_data[duration_data['duration_minutes'] >= min_dur])
        else:
            count = len(duration_data[(duration_data['duration_minutes'] >= min_dur) &
                                    (duration_data['duration_minutes'] < max_dur)])
        range_counts.append(count)

    ax4.bar(range_labels, range_counts, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Duration Range')
    ax4.set_ylabel('Number of Videos')
    ax4.set_title('Videos by Duration Range')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'duration_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_upload_date_analysis(df, output_dir):
    """Create plots analyzing upload dates over time."""
    date_data = df.dropna(subset=['upload_date'])

    if len(date_data) == 0:
        print("No upload date data available")
        return

    date_data = date_data.copy()
    date_data = date_data.sort_values('upload_date')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # 1. Videos over time
    date_data['year_month'] = date_data['upload_date'].dt.to_period('M')
    monthly_counts = date_data.groupby('year_month').size()

    ax1.plot(monthly_counts.index.to_timestamp(), monthly_counts.values, marker='o', linewidth=2)
    ax1.set_xlabel('Upload Date')
    ax1.set_ylabel('Number of Videos')
    ax1.set_title(f'Videos Uploaded Over Time\n({len(date_data)} videos with dates)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # 2. Average views over time (if available)
    views_date_data = date_data.dropna(subset=['view_count'])
    if len(views_date_data) > 0:
        monthly_views = views_date_data.groupby('year_month')['view_count'].mean()

        ax2.plot(monthly_views.index.to_timestamp(), monthly_views.values,
                marker='s', linewidth=2, color='orange')
        ax2.set_xlabel('Upload Date')
        ax2.set_ylabel('Average View Count')
        ax2.set_yscale('log')
        ax2.set_title('Average Views by Upload Month')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'upload_date_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_chronological_order_plot(df, output_dir):
    """Create plot showing rolling most recent upload date as you go through liked videos chronologically."""
    date_data = df.dropna(subset=['upload_date']).copy()

    if len(date_data) == 0:
        print("No upload date data available for chronological order plot")
        return

    # Since the original list is from newest liked (index 0) to oldest liked (highest index),
    # we need to flip it to process chronologically (oldest liked first)
    max_index = len(df) - 1
    date_data['chronological_order'] = max_index - date_data['index']

    # Sort by chronological order (oldest liked first)
    date_data = date_data.sort_values('chronological_order')

    # Calculate rolling most recent upload date
    date_data['rolling_most_recent'] = date_data['upload_date'].cummax()

    plt.figure(figsize=(15, 10))

    # Create scatter plot with correct axes
    plt.scatter(date_data['rolling_most_recent'], date_data['upload_date'],
                alpha=0.6, s=20, edgecolors='black', linewidth=0.3)

    # Add diagonal line for reference (y=x) only where both axes overlap
    x_min, x_max = date_data['rolling_most_recent'].min(), date_data['rolling_most_recent'].max()
    y_min, y_max = date_data['upload_date'].min(), date_data['upload_date'].max()

    # Find the overlapping range for the diagonal line
    diag_min = max(x_min, y_min)
    diag_max = min(x_max, y_max)

    if diag_min <= diag_max:  # Only draw if there's an overlap
        plt.plot([diag_min, diag_max], [diag_min, diag_max], 'r--', alpha=0.5, linewidth=1)

    plt.xlabel('Rolling Most Recent Upload Date')
    plt.ylabel('Video Upload Date')
    # plt.title(f'Video Upload Date vs Rolling Most Recent Upload Date\n'
    #           f'({len(date_data)} videos with upload dates)\n'
    #           f'Shows when videos were uploaded vs your chronological "freshness" progress')
    plt.grid(True, alpha=1)
    plt.legend()

    # Add some statistics
    date_range = date_data['upload_date'].max() - date_data['upload_date'].min()
    oldest_video = date_data['upload_date'].min()
    newest_video = date_data['upload_date'].max()

    # Add text box with statistics
    # stats_text = f'Video Upload Date Range:\n{oldest_video.strftime("%Y-%m-%d")} to {newest_video.strftime("%Y-%m-%d")}\n({date_range.days} days span)'
    # plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
    #          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Rotate y-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'chronological_order.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print some insights
    print(f"Rolling upload date analysis:")
    print(f"  Total videos with dates: {len(date_data)}")
    print(f"  Oldest video uploaded: {oldest_video.strftime('%Y-%m-%d')}")
    print(f"  Newest video uploaded: {newest_video.strftime('%Y-%m-%d')}")
    print(f"  Upload date span: {date_range.days} days")

    # Analyze the rolling most recent pattern
    final_most_recent = date_data['rolling_most_recent'].iloc[-1]
    print(f"  Most recent video you've liked (upload date): {final_most_recent.strftime('%Y-%m-%d')}")

    # Find points where you "went backwards" in time (liked older videos)
    backwards_points = date_data[date_data['upload_date'] < date_data['rolling_most_recent']]
    print(f"  Times you liked older videos: {len(backwards_points)}/{len(date_data)} ({len(backwards_points)/len(date_data)*100:.1f}%)")

    # Find the biggest "backwards jumps"
    if len(backwards_points) > 0:
        backwards_points['time_gap'] = backwards_points['rolling_most_recent'] - backwards_points['upload_date']
        biggest_jump = backwards_points.loc[backwards_points['time_gap'].idxmax()]
        gap_days = biggest_jump['time_gap'].days
        print(f"  Biggest backwards jump: {gap_days} days")
        print(f"    Video: '{biggest_jump['title'][:50]}...' from {biggest_jump['upload_date'].strftime('%Y-%m-%d')}")
        print(f"    (You had been liking videos up to {biggest_jump['rolling_most_recent'].strftime('%Y-%m-%d')})")

    return date_data


def create_upload_date_vs_index_plot(df, output_dir):
    """Create plot showing upload date vs flipped index (chronological order of liking)."""
    date_data = df.dropna(subset=['upload_date']).copy()

    if len(date_data) == 0:
        print("No upload date data available for upload date vs index plot")
        return

    # Since the original list is from newest liked (index 0) to oldest liked (highest index),
    # we need to flip it so that x-axis represents chronological order of liking
    # Original index 0 (newest liked) becomes the highest number
    # Original highest index (oldest liked) becomes 0
    max_index = len(df) - 1
    date_data['chronological_order'] = max_index - date_data['index']

    # Sort by chronological order for plotting
    date_data = date_data.sort_values('chronological_order')

    plt.figure(figsize=(15, 10))

    # Create scatter plot
    plt.scatter(date_data['chronological_order'], date_data['upload_date'],
                alpha=0.6, s=20, edgecolors='black', linewidth=0.3)

    plt.xlabel('Index')
    plt.ylabel('Video Upload Date')
    # plt.title(f'Video Upload Date vs Chronological Order of Liking\n'
    #           f'({len(date_data)} videos with upload dates)\n'
    #           f'Shows when videos were uploaded vs the order you liked them')
    plt.grid(True, alpha=1)

    # Add some statistics
    date_range = date_data['upload_date'].max() - date_data['upload_date'].min()
    oldest_video = date_data['upload_date'].min()
    newest_video = date_data['upload_date'].max()

    # Add text box with statistics
    # stats_text = f'Video Upload Date Range:\n{oldest_video.strftime("%Y-%m-%d")} to {newest_video.strftime("%Y-%m-%d")}\n({date_range.days} days span)'
    # plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
    #          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'upload_date_vs_index.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print some insights
    print(f"Upload date vs index analysis:")
    print(f"  Total videos with dates: {len(date_data)}")
    print(f"  Oldest video uploaded: {oldest_video.strftime('%Y-%m-%d')}")
    print(f"  Newest video uploaded: {newest_video.strftime('%Y-%m-%d')}")
    print(f"  Upload date span: {date_range.days} days")

    # Find some interesting patterns
    recent_likes_old_videos = date_data[
        (date_data['chronological_order'] > len(date_data) * 0.8) &  # Recently liked (right 20% of plot)
        (date_data['upload_date'] < date_data['upload_date'].quantile(0.2))  # But old videos (bottom 20% of plot)
    ]

    if len(recent_likes_old_videos) > 0:
        print(f"  Found {len(recent_likes_old_videos)} recently liked videos that are quite old")
        oldest_recent = recent_likes_old_videos.loc[recent_likes_old_videos['upload_date'].idxmin()]
        print(f"    Oldest recently liked: '{oldest_recent['title'][:50]}...' from {oldest_recent['upload_date'].strftime('%Y-%m-%d')}")

    return date_data


def main():
    parser = argparse.ArgumentParser(description='Analyze YouTube metadata and create visualizations')
    parser.add_argument('json_file', nargs='?', 
                       help='Path to JSON file with scraped metadata')
    
    args = parser.parse_args()
    
    # Find JSON file if not provided
    if not args.json_file:
        # Look for JSON files in current directory and subdirectories
        current_dir = Path('.')
        json_files = list(current_dir.glob('**/*scraped_metadata*.json'))
        
        if not json_files:
            print("No scraped metadata JSON files found. Please provide a file path.")
            return
        elif len(json_files) == 1:
            json_file = json_files[0]
            print(f"Found metadata file: {json_file}")
        else:
            print("Multiple metadata files found:")
            for i, f in enumerate(json_files):
                print(f"  {i+1}: {f}")
            choice = input("Enter number to select file (or press Enter for most recent): ")
            if choice.strip():
                json_file = json_files[int(choice) - 1]
            else:
                # Use most recent file
                json_file = max(json_files, key=lambda f: f.stat().st_mtime)
                print(f"Using most recent: {json_file}")
    else:
        json_file = Path(args.json_file)
    
    if not json_file.exists():
        print(f"File not found: {json_file}")
        return
    
    # Load and parse data
    df = load_and_parse_metadata(json_file)
    
    # Create output directory for plots
    output_dir = Path('analysis_output')
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_comments_vs_views_plot(df, output_dir)
    create_duration_analysis_plots(df, output_dir)
    create_upload_date_analysis(df, output_dir)
    create_upload_date_vs_index_plot(df, output_dir)
    create_chronological_order_plot(df, output_dir)

    # Save processed data for further analysis
    csv_path = output_dir / 'processed_metadata.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved processed data: {csv_path}")

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total videos analyzed: {len(df)}")

    if 'view_count' in df.columns and df['view_count'].notna().any():
        views_data = df['view_count'].dropna()
        print(f"Views - Min: {views_data.min():,}, Max: {views_data.max():,}, Median: {views_data.median():,.0f}")

    if 'comment_count' in df.columns and df['comment_count'].notna().any():
        comments_data = df['comment_count'].dropna()
        print(f"Comments - Min: {comments_data.min():,}, Max: {comments_data.max():,}, Median: {comments_data.median():.0f}")

    if 'duration_seconds' in df.columns and df['duration_seconds'].notna().any():
        duration_data = df['duration_seconds'].dropna()
        print(f"Duration - Min: {duration_data.min()/60:.1f} min, Max: {duration_data.max()/60:.1f} min, Median: {duration_data.median()/60:.1f} min")

    print(f"\nAnalysis complete! Check the '{output_dir}' directory for outputs.")


if __name__ == '__main__':
    main()
