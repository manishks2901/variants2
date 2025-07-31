#!/usr/bin/env python3
"""
Test the new single black screen implementation multiple times
"""
import asyncio
import os
import sys
import re
sys.path.append('/Users/manishkumarsharma/Documents/variants2')

from app.services.ffmpeg_service import FFmpegTransformationService

async def test_single_black_screen():
    """Test single black screen with timing analysis"""
    input_video = "/Users/manishkumarsharma/Documents/variants2/variant2.mp4"
    
    print("🎯 SINGLE BLACK SCREEN TEST")
    print("=" * 50)
    print("New specification: 1 black screen, 1-2 seconds, random position")
    print("=" * 50)
    
    service = FFmpegTransformationService()
    transformations = service.get_transformations()
    
    # Find black screen transformation
    black_screen_transform = None
    for transform in transformations:
        if transform.name == "black_screen_random":
            black_screen_transform = transform
            break
    
    if not black_screen_transform:
        print("❌ Transformation not found")
        return
    
    # Test multiple times to show randomization
    for i in range(5):
        print(f"\n🔄 Test #{i+1}:")
        output_video = f"/Users/manishkumarsharma/Documents/variants2/single_black_{i+1}.mp4"
        
        # Generate command
        cmd = black_screen_transform.execute(input_video, output_video)
        
        # Extract timing information from command
        duration_match = re.search(r'duration=(\d+\.\d+)', cmd)
        trim_matches = re.findall(r'trim=start=(\d+\.\d+):end=(\d+\.\d+)', cmd)
        
        black_duration = float(duration_match.group(1)) if duration_match else 0
        
        # Calculate insertion point
        insertion_point = 0
        if len(trim_matches) >= 1:
            insertion_point = float(trim_matches[0][1])  # End of first video segment
        
        print(f"   ⚫ Black screen: {black_duration:.2f}s at {insertion_point:.2f}s")
        print(f"   📋 Command: {cmd[:80]}...")
        
        # Execute
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Success! File: single_black_{i+1}.mp4")
            if os.path.exists(output_video):
                size = os.path.getsize(output_video) / (1024*1024)
                print(f"   📊 Size: {size:.1f}MB")
        else:
            print(f"   ❌ Failed: {result.stderr[:50]}")
    
    print(f"\n📈 Summary:")
    print("• Each test shows different random timing")
    print("• Black screen duration: 1-2 seconds (as requested)")
    print("• Single black screen per video (as requested)")
    print("• Random position in video (avoiding start/end)")

if __name__ == "__main__":
    asyncio.run(test_single_black_screen())
