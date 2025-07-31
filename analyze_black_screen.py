#!/usr/bin/env python3
"""
Detailed test to show black screen timing details
"""
import asyncio
import os
import sys
import subprocess
sys.path.append('/Users/manishkumarsharma/Documents/variants2')

from app.services.ffmpeg_service import FFmpegTransformationService

async def analyze_black_screen_output():
    """Analyze the output to show timing details"""
    input_video = "/Users/manishkumarsharma/Documents/variants2/variant2.mp4"
    output_video = "/Users/manishkumarsharma/Documents/variants2/final_black_test.mp4"
    
    print("📊 Detailed Black Screen Analysis")
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
    
    # Generate command and analyze
    cmd = black_screen_transform.execute(input_video, output_video)
    print(f"🔧 Generated FFmpeg command:")
    print(f"{cmd}\n")
    
    # Extract filter details from command
    if "-filter_complex" in cmd:
        filter_part = cmd.split("-filter_complex")[1].split("-map")[0].strip().strip('"')
        print("🎯 Filter breakdown:")
        
        # Count video and black segments
        video_segments = filter_part.count("[0:v]trim")
        black_segments = filter_part.count("color=black")
        concat_n = filter_part.split("concat=n=")[1].split(":")[0] if "concat=n=" in filter_part else "unknown"
        
        print(f"   📹 Video segments: {video_segments}")
        print(f"   ⚫ Black segments: {black_segments}")
        print(f"   🔗 Total segments to concatenate: {concat_n}")
        
        # Find black durations
        import re
        black_durations = re.findall(r'duration=(\d+\.\d+)', filter_part)
        if black_durations:
            durations = [float(d) for d in black_durations if 2.0 <= float(d) <= 3.5]  # Filter for black screen durations
            if durations:
                print(f"   ⏱️  Black screen durations: {[f'{d:.2f}s' for d in durations]}")
    
    # Execute and analyze result
    print(f"\n🎬 Executing transformation...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Transformation completed successfully!")
        
        if os.path.exists(output_video):
            # Get input and output durations
            input_info = await service.get_video_info(input_video)
            output_info = await service.get_video_info(output_video)
            
            input_duration = input_info.get('duration', 0)
            output_duration = output_info.get('duration', 0)
            
            input_size = os.path.getsize(input_video) / (1024*1024)
            output_size = os.path.getsize(output_video) / (1024*1024)
            
            print(f"\n📊 Results:")
            print(f"   ⏱️  Input duration:  {input_duration:.2f}s")
            print(f"   ⏱️  Output duration: {output_duration:.2f}s")
            print(f"   📁 Input size:  {input_size:.1f}MB")
            print(f"   📁 Output size: {output_size:.1f}MB")
            print(f"   📈 Size change: {((output_size/input_size-1)*100):+.1f}%")
            
            if black_segments > 0:
                estimated_black_time = sum([float(d) for d in black_durations if 2.0 <= float(d) <= 3.5]) if black_durations else 0
                print(f"   ⚫ Estimated black screen time: {estimated_black_time:.2f}s")
        
        print(f"\n🎯 Final output: {output_video}")
    else:
        print(f"❌ Command failed: {result.stderr[:200]}")

if __name__ == "__main__":
    asyncio.run(analyze_black_screen_output())
