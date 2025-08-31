#!/usr/bin/env python3
"""
Analyze scraped metadata JSON files to identify videos with errors or missing data.
Generates a youtube_liked.json compatible output for re-processing failed videos.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime


class ScrapedMetadataAnalyzer:
    """Analyzes scraped metadata to identify videos that need re-processing."""
    
    def __init__(self):
        self.error_videos: List[Dict] = []
        self.missing_description_videos: List[Dict] = []
        self.all_failed_videos: List[Dict] = []
        
    def load_scraped_metadata(self, file_path: str) -> List[Dict]:
        """Load scraped metadata from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Loaded {len(data)} videos from {file_path}")
            return data
            
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            raise
    
    def analyze_video_metadata(self, video: Dict) -> Dict:
        """Analyze a single video's metadata for errors or missing data."""
        issues = []
        video_id = video.get('videoId', 'unknown')

        # Check for error field - this is the primary indicator of scraping failure
        if 'error' in video:
            issues.append(f"scraping_error: {video['error']}")

        # Check for missing or empty description
        description = video.get('description', '')
        if not description or description.strip() == '':
            issues.append("missing_description")

        return {
            'video_id': video_id,
            'issues': issues,
            'has_issues': len(issues) > 0,
            'original_video': video
        }
    
    def analyze_all_videos(self, scraped_data: List[Dict]) -> Dict:
        """Analyze all videos and categorize issues."""
        analysis_results = {
            'total_videos': len(scraped_data),
            'videos_with_errors': 0,
            'videos_with_missing_descriptions': 0,
            'videos_with_scraping_errors': 0,
            'successful_videos': 0,
            'issue_breakdown': {},
            'failed_videos': []
        }

        issue_counts = {}

        for video in scraped_data:
            result = self.analyze_video_metadata(video)

            if result['has_issues']:
                analysis_results['videos_with_errors'] += 1
                analysis_results['failed_videos'].append(result)

                # Count specific issue types
                for issue in result['issues']:
                    if issue not in issue_counts:
                        issue_counts[issue] = 0
                    issue_counts[issue] += 1

                    # Update specific counters
                    if 'scraping_error' in issue:
                        analysis_results['videos_with_scraping_errors'] += 1
                    elif 'missing_description' in issue:
                        analysis_results['videos_with_missing_descriptions'] += 1
            else:
                analysis_results['successful_videos'] += 1

        analysis_results['issue_breakdown'] = issue_counts
        return analysis_results
    
    def convert_to_youtube_liked_format(self, failed_videos: List[Dict]) -> Dict:
        """Convert failed videos to youtube_liked.json format for re-processing."""
        youtube_liked_format = {
            "metadata": {
                "source": "Failed Video Analysis",
                "generated_at": datetime.now().isoformat(),
                "total_videos": len(failed_videos),
                "description": "Videos that failed during metadata scraping and need re-processing"
            },
            "videos": []
        }

        for failed_video in failed_videos:
            original = failed_video['original_video']

            # Extract basic info needed for youtube_liked.json format
            video_entry = {
                "videoId": original.get('videoId', ''),
                "title": original.get('title', ''),
                "channel": original.get('channel', ''),
                "url": original.get('url', ''),
                "issues": failed_video['issues']  # Add issues for reference
            }

            youtube_liked_format['videos'].append(video_entry)

        return youtube_liked_format
    
    def print_analysis_report(self, analysis: Dict) -> None:
        """Print a detailed analysis report."""
        print("\n" + "="*60)
        print("📊 SCRAPED METADATA ANALYSIS REPORT")
        print("="*60)

        print(f"📈 Total videos analyzed: {analysis['total_videos']}")
        print(f"✅ Successful videos: {analysis['successful_videos']}")
        print(f"❌ Videos with errors: {analysis['videos_with_errors']}")
        print(f"📊 Success rate: {(analysis['successful_videos']/analysis['total_videos']*100):.1f}%")

        print("\n🔍 ERROR BREAKDOWN:")
        print("-" * 40)
        print(f"🚫 Scraping errors: {analysis['videos_with_scraping_errors']}")
        print(f"📝 Missing descriptions: {analysis['videos_with_missing_descriptions']}")

        if analysis['issue_breakdown']:
            print("\n📋 DETAILED ISSUE COUNTS:")
            print("-" * 40)
            for issue, count in sorted(analysis['issue_breakdown'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {issue}: {count}")

        print("\n" + "="*60)


def main():
    """Main function to analyze scraped metadata."""
    parser = argparse.ArgumentParser(description='Analyze scraped metadata for errors and missing data')
    parser.add_argument('input_file', help='Path to scraped_metadata_*.json file')
    parser.add_argument('-o', '--output', help='Output file for failed videos (youtube_liked.json format)', 
                       default=None)
    parser.add_argument('--missing-descriptions-only', action='store_true',
                       help='Only include videos with missing descriptions')
    parser.add_argument('--scraping-errors-only', action='store_true',
                       help='Only include videos with scraping errors')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"❌ Input file not found: {args.input_file}")
        return 1
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"failed_videos_{timestamp}.json"
    
    print(f"🔍 Analyzing: {args.input_file}")
    print(f"💾 Output will be saved to: {args.output}")
    
    try:
        # Initialize analyzer
        analyzer = ScrapedMetadataAnalyzer()
        
        # Load and analyze data
        scraped_data = analyzer.load_scraped_metadata(args.input_file)
        analysis = analyzer.analyze_all_videos(scraped_data)
        
        # Print analysis report
        analyzer.print_analysis_report(analysis)
        
        # Filter failed videos based on options
        failed_videos = analysis['failed_videos']
        
        if args.missing_descriptions_only:
            failed_videos = [v for v in failed_videos if any('missing_description' in issue for issue in v['issues'])]
            print(f"\n🔍 Filtering for missing descriptions only: {len(failed_videos)} videos")

        if args.scraping_errors_only:
            failed_videos = [v for v in failed_videos if any('scraping_error' in issue for issue in v['issues'])]
            print(f"\n🔍 Filtering for scraping errors only: {len(failed_videos)} videos")
        
        if not failed_videos:
            print("\n✅ No failed videos found matching criteria!")
            return 0
        
        # Convert to youtube_liked.json format
        output_data = analyzer.convert_to_youtube_liked_format(failed_videos)
        
        # Save output
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Failed videos saved to: {args.output}")
        print(f"📊 Total failed videos for re-processing: {len(failed_videos)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
