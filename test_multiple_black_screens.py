#!/usr/bin/env python3
"""
Test multiple black screens in a single video
"""
import asyncio
import os
import sys
sys.path.append('/Users/manishkumarsharma/Documents/variants2')

from app.services.ffmpeg_service import FFmpegTransformationService

async def test_multiple_black_screens():
    """Test multiple black screen insertions"""
    input_video = "/Users/manishkumarsharma/Documents/variants2/variant2.mp4"
    output_video = "/Users/manishkumarsharma/Documents/variants2/test_multiple_black.mp4"
    
    print("🎬 Testing multiple black screen insertions...")
    
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
    
    # Run it multiple times to see different patterns
    for i in range(3):
        print(f"\n🔄 Test run {i+1}/3:")
        test_output = f"/Users/manishkumarsharma/Documents/variants2/test_black_{i+1}.mp4"
        
        cmd = black_screen_transform.execute(input_video, test_output)
        print(f"📋 Command: {cmd[:100]}...")  # Show first 100 chars
        
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Success! Created: {test_output}")
            if os.path.exists(test_output):
                size = os.path.getsize(test_output) / (1024*1024)
                print(f"📊 File size: {size:.1f}MB")
        else:
            print(f"❌ Failed: {result.stderr[:100]}")

if __name__ == "__main__":
    asyncio.run(test_multiple_black_screens())
