#!/usr/bin/env python3
"""
Test script to check video playback status, autoplay behavior, and muting
after browser restarts. Helps identify issues with browser restart logic.
"""

import asyncio
import json
import sys
from pathlib import Path
from playwright.async_api import async_playwright, Page
from datetime import datetime


class VideoPlaybackTester:
    """Test video playback behavior across browser restarts."""
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.test_results = []
    
    async def get_video_status(self, page: Page) -> dict:
        """Get comprehensive video player status."""
        try:
            # Wait for video player to load
            await page.wait_for_selector('.html5-video-player', timeout=10000)
            
            # Get video element status
            video_status = await page.evaluate("""
                () => {
                    const video = document.querySelector('video');
                    const player = document.querySelector('.html5-video-player');
                    
                    if (!video) return { error: 'No video element found' };
                    
                    return {
                        // Basic video properties
                        paused: video.paused,
                        muted: video.muted,
                        volume: video.volume,
                        currentTime: video.currentTime,
                        duration: video.duration,
                        readyState: video.readyState,
                        
                        // Player state
                        playerClasses: player ? player.className : 'no-player',
                        
                        // Autoplay detection
                        autoplay: video.autoplay,
                        
                        // Audio context (if available)
                        audioContext: typeof AudioContext !== 'undefined' ? 'available' : 'not-available',
                        
                        // YouTube specific
                        ytPlayerState: window.ytplayer ? 'available' : 'not-available'
                    };
                }
            """)
            
            return video_status
            
        except Exception as e:
            return {'error': str(e)}
    
    async def check_ad_presence(self, page: Page) -> dict:
        """Check if ads are present and their status."""
        try:
            ad_status = await page.evaluate("""
                () => {
                    const adElements = [
                        '.ytp-ad-skip-button',
                        '.ytp-ad-text',
                        '.video-ads',
                        '[class*="ad-showing"]'
                    ];
                    
                    let adFound = false;
                    let adType = 'none';
                    
                    for (const selector of adElements) {
                        const element = document.querySelector(selector);
                        if (element && element.offsetParent !== null) {
                            adFound = true;
                            adType = selector;
                            break;
                        }
                    }
                    
                    return {
                        adPresent: adFound,
                        adType: adType,
                        skipButtonVisible: document.querySelector('.ytp-ad-skip-button') !== null
                    };
                }
            """)
            
            return ad_status
            
        except Exception as e:
            return {'error': str(e)}
    
    async def apply_manual_controls(self, page: Page, test_name: str) -> dict:
        """Apply manual muting and play controls after browser restart."""
        control_results = {
            'mute_attempted': False,
            'play_attempted': False,
            'mute_success': False,
            'play_success': False
        }

        try:
            # Wait for video player to be ready
            await page.wait_for_selector('.html5-video-player', timeout=10000)

            # Focus on the video player first
            await page.click('.html5-video-player')
            await page.wait_for_timeout(500)

            # Get status before manual controls
            before_status = await self.get_video_status(page)
            print(f"🎮 Before manual controls: paused={before_status.get('paused')}, muted={before_status.get('muted')}")

            # Attempt to mute with 'M' key
            print("🔇 Attempting to mute with 'M' key...")
            await page.keyboard.press('m')
            control_results['mute_attempted'] = True
            await page.wait_for_timeout(1000)

            # Check if muting worked
            after_mute_status = await self.get_video_status(page)
            control_results['mute_success'] = after_mute_status.get('muted', False)
            print(f"🔇 After mute attempt: muted={control_results['mute_success']}")

            # Attempt to play with Space key (only if paused)
            if before_status.get('paused', True):
                print("▶️ Attempting to play with Space key...")
                await page.keyboard.press(' ')
                control_results['play_attempted'] = True
                await page.wait_for_timeout(1000)

                # Check if play worked
                after_play_status = await self.get_video_status(page)
                control_results['play_success'] = not after_play_status.get('paused', True)
                print(f"▶️ After play attempt: paused={after_play_status.get('paused')}")
            else:
                print("▶️ Video already playing, skipping play attempt")

            return control_results

        except Exception as e:
            print(f"❌ Manual controls failed: {e}")
            control_results['error'] = str(e)
            return control_results

    async def test_single_video(self, page: Page, video_url: str, test_name: str, apply_manual_controls: bool = False) -> dict:
        """Test a single video's playback behavior."""
        print(f"🎬 Testing: {test_name}")
        print(f"📺 URL: {video_url}")

        start_time = datetime.now()

        try:
            # Navigate to video
            await page.goto(video_url, timeout=15000)

            # Wait for initial load
            await page.wait_for_timeout(3000)

            # Get initial status
            initial_status = await self.get_video_status(page)
            print(f"📊 Initial status: paused={initial_status.get('paused')}, muted={initial_status.get('muted')}")

            # Check for ads
            ad_status = await self.check_ad_presence(page)
            print(f"📺 Ad status: present={ad_status.get('adPresent')}, type={ad_status.get('adType')}")

            # Apply manual controls if requested (for browser restart tests)
            manual_control_results = {}
            if apply_manual_controls:
                manual_control_results = await self.apply_manual_controls(page, test_name)

            # Wait a bit more for autoplay to kick in
            await page.wait_for_timeout(5000)

            # Get final status
            final_status = await self.get_video_status(page)
            print(f"📊 Final status: paused={final_status.get('paused')}, muted={final_status.get('muted')}")

            # Calculate if video started playing
            time_progressed = final_status.get('currentTime', 0) > initial_status.get('currentTime', 0)

            result = {
                'test_name': test_name,
                'video_url': video_url,
                'timestamp': start_time.isoformat(),
                'success': True,
                'initial_status': initial_status,
                'final_status': final_status,
                'ad_status': ad_status,
                'manual_controls': manual_control_results,
                'time_progressed': time_progressed,
                'autoplay_working': not final_status.get('paused', True) or time_progressed
            }

            print(f"✅ Test completed: autoplay_working={result['autoplay_working']}")
            return result

        except Exception as e:
            error_result = {
                'test_name': test_name,
                'video_url': video_url,
                'timestamp': start_time.isoformat(),
                'success': False,
                'error': str(e)
            }
            print(f"❌ Test failed: {e}")
            return error_result
    
    async def run_browser_restart_test(self, test_videos: list) -> list:
        """Test video playback across multiple browser restarts."""
        results = []
        
        async with async_playwright() as p:
            for i, video_url in enumerate(test_videos):
                print(f"\n{'='*60}")
                print(f"🔄 BROWSER RESTART TEST {i+1}/{len(test_videos)}")
                print(f"{'='*60}")
                
                # Launch fresh browser for each test
                browser = await p.firefox.launch(
                    headless=self.headless,
                    args=['--mute-audio', '--disable-audio-output']
                )
                
                try:
                    # Create context with session if available
                    session_file = Path("sessions/browser_context.json")
                    if session_file.exists():
                        context = await browser.new_context(storage_state=str(session_file))
                        print("📁 Loaded saved session")
                    else:
                        context = await browser.new_context()
                        print("🆕 Created new session")
                    
                    page = await context.new_page()

                    # Test this video with manual controls after browser restart
                    result = await self.test_single_video(page, video_url, f"Browser Restart {i+1}", apply_manual_controls=True)
                    results.append(result)
                    
                finally:
                    # Clean shutdown
                    try:
                        await context.close()
                        await browser.close()
                        print("🔄 Browser closed cleanly")
                    except Exception as e:
                        print(f"⚠️ Browser close error: {e}")
                
                # Pause between tests
                if i < len(test_videos) - 1:
                    print("⏸️ Pausing 3 seconds between tests...")
                    await asyncio.sleep(3)
        
        return results
    
    async def run_single_browser_test(self, test_videos: list) -> list:
        """Test video playback in a single browser session."""
        results = []
        
        async with async_playwright() as p:
            print(f"\n{'='*60}")
            print(f"🔄 SINGLE BROWSER SESSION TEST")
            print(f"{'='*60}")
            
            browser = await p.firefox.launch(
                headless=self.headless,
                args=['--mute-audio', '--disable-audio-output']
            )
            
            try:
                # Create context with session if available
                session_file = Path("sessions/browser_context.json")
                if session_file.exists():
                    context = await browser.new_context(storage_state=str(session_file))
                    print("📁 Loaded saved session")
                else:
                    context = await browser.new_context()
                    print("🆕 Created new session")
                
                page = await context.new_page()
                
                # Test all videos in same browser (no manual controls needed)
                for i, video_url in enumerate(test_videos):
                    result = await self.test_single_video(page, video_url, f"Same Browser {i+1}", apply_manual_controls=False)
                    results.append(result)
                    
                    # Pause between videos
                    if i < len(test_videos) - 1:
                        print("⏸️ Pausing 2 seconds between videos...")
                        await asyncio.sleep(2)
                
            finally:
                try:
                    await context.close()
                    await browser.close()
                    print("🔄 Browser closed cleanly")
                except Exception as e:
                    print(f"⚠️ Browser close error: {e}")
        
        return results
    
    def print_summary(self, restart_results: list, single_results: list):
        """Print test summary comparing both approaches."""
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")
        
        # Analyze restart results
        restart_success = sum(1 for r in restart_results if r.get('autoplay_working', False))
        restart_total = len(restart_results)
        
        # Analyze single browser results
        single_success = sum(1 for r in single_results if r.get('autoplay_working', False))
        single_total = len(single_results)
        
        print(f"🔄 Browser Restart Approach:")
        print(f"   ✅ Autoplay working: {restart_success}/{restart_total} ({restart_success/restart_total*100:.1f}%)")
        
        print(f"🔄 Single Browser Approach:")
        print(f"   ✅ Autoplay working: {single_success}/{single_total} ({single_success/single_total*100:.1f}%)")
        
        # Detailed breakdown
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 40)
        
        for i, (restart, single) in enumerate(zip(restart_results, single_results)):
            print(f"Video {i+1}:")

            # Restart results with manual controls
            restart_manual = restart.get('manual_controls', {})
            mute_success = restart_manual.get('mute_success', False)
            play_success = restart_manual.get('play_success', False)

            print(f"  Restart: {'✅' if restart.get('autoplay_working') else '❌'} "
                  f"(muted: {restart.get('final_status', {}).get('muted', 'unknown')}, "
                  f"manual_mute: {'✅' if mute_success else '❌'}, "
                  f"manual_play: {'✅' if play_success else '❌'})")

            print(f"  Single:  {'✅' if single.get('autoplay_working') else '❌'} "
                  f"(muted: {single.get('final_status', {}).get('muted', 'unknown')})")


async def main():
    """Main test function."""
    # Test videos (mix of short and long videos)
    test_videos = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (short)
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # Me at the zoo (short)
        "https://www.youtube.com/watch?v=9bZkp7q19f0"   # Gangnam Style (longer)
    ]
    
    print("🎬 Video Playback Test Suite")
    print("Testing autoplay and muting behavior across browser restarts")
    print(f"📺 Testing {len(test_videos)} videos")
    
    tester = VideoPlaybackTester(headless=False)  # Use visible browser for testing
    
    # Run both test approaches
    print("\n🔄 Running browser restart tests...")
    restart_results = await tester.run_browser_restart_test(test_videos)
    
    print("\n🔄 Running single browser tests...")
    single_results = await tester.run_single_browser_test(test_videos)
    
    # Print summary
    tester.print_summary(restart_results, single_results)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"video_playback_test_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'browser_restart_results': restart_results,
            'single_browser_results': single_results,
            'test_timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
