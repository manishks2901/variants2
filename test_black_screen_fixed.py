#!/usr/bin/env python3
"""
Test script to debug black screen transformation - proper version
"""
import asyncio
import os
import sys
sys.path.append('/Users/manishkumarsharma/Documents/variants2')

from app.services.ffmpeg_service import FFmpegTransformationService

async def test_black_screen_proper():
    """Test the black screen transformation properly"""
    input_video = "/Users/manishkumarsharma/Documents/variants2/variant2.mp4"
    output_video = "/Users/manishkumarsharma/Documents/variants2/debug_black_screen_fixed.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return
    
    print("🔍 Testing fixed black screen transformation...")
    
    # Create service instance
    service = FFmpegTransformationService()
    
    try:
        # Get the transformations list
        transformations = service.get_transformations()
        
        # Find the black screen transformation
        black_screen_transform = None
        for transform in transformations:
            if transform.name == "black_screen_random":
                black_screen_transform = transform
                break
        
        if not black_screen_transform:
            print("❌ Black screen transformation not found!")
            return
        
        print(f"✅ Found transformation: {black_screen_transform.name}")
        
        # Execute the transformation function
        cmd = black_screen_transform.execute(input_video, output_video)
        print(f"📋 Generated command:\n{cmd}\n")
        
        # Get video info first
        video_info = await service.get_video_info(input_video)
        print(f"📹 Input video info: duration={video_info.get('duration', 'unknown')}s")
        
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
                print(f"📹 Output video info: duration={output_info.get('duration', 'unknown')}s")
                
                # Check file size
                input_size = os.path.getsize(input_video) / (1024*1024)
                output_size = os.path.getsize(output_video) / (1024*1024)
                print(f"📊 File sizes: input={input_size:.1f}MB, output={output_size:.1f}MB")
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
    asyncio.run(test_black_screen_proper())
