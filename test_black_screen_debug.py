#!/usr/bin/env python3
"""
Test script to debug black screen transformation
"""
import asyncio
import os
import sys
sys.path.append('/Users/manishkumarsharma/Documents/variants2')

from app.services.ffmpeg_service import FFmpegTransformationService

async def test_black_screen_debug():
    """Test the black screen transformation with debugging"""
    input_video = "/Users/manishkumarsharma/Documents/variants2/variant2.mp4"
    output_video = "/Users/manishkumarsharma/Documents/variants2/debug_black_screen.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return
    
    print("🔍 Testing black screen transformation...")
    
    # Create service instance
    service = FFmpegTransformationService()
    
    try:
        # Test the black screen command generation
        cmd = await service.black_screen_random(input_video, output_video)
        print(f"📋 Generated command:\n{cmd}\n")
        
        # Get video info first
        video_info = await service.get_video_info(input_video)
        print(f"📹 Video info: {video_info}")
        
        # Execute the command
        print("🎬 Executing transformation...")
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Black screen transformation completed successfully!")
            if os.path.exists(output_video):
                print(f"📁 Output file created: {output_video}")
                # Get output video info
                output_info = await service.get_video_info(output_video)
                print(f"📹 Output video info: {output_info}")
            else:
                print("❌ Output file not created")
        else:
            print(f"❌ Command failed with return code: {result.returncode}")
            print(f"📝 Error: {result.stderr}")
            print(f"📝 Output: {result.stdout}")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_black_screen_debug())
