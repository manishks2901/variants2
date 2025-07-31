#!/usr/bin/env python3
"""
Test script for the fixed random_frame_inserts function
"""

import sys
import os
import tempfile
import subprocess

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.ffmpeg_service import FFmpegTransformationService

def create_test_video(output_path: str, duration: int = 5):
    """Create a simple test video using FFmpeg"""
    cmd = [
        'ffmpeg', '-f', 'lavfi', 
        '-i', f'testsrc=duration={duration}:size=640x480:rate=30',
        '-f', 'lavfi',
        '-i', f'sine=frequency=440:duration={duration}:sample_rate=44100',
        '-c:v', 'libx264', '-c:a', 'aac',
        '-pix_fmt', 'yuv420p',
        '-y', output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to create test video: {result.stderr}")
    
    print(f"✅ Test video created: {output_path}")

def test_random_frame_inserts():
    """Test the random_frame_inserts function"""
    print("🧪 Testing random_frame_inserts function...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "test_input.mp4")
        output_path = os.path.join(temp_dir, "test_output.mp4")
        
        try:
            # Create test video
            print("📹 Creating test video...")
            create_test_video(input_path, duration=5)
            
            # Get all transformations and find random_frame_inserts
            transformations = FFmpegTransformationService.get_transformations()
            
            # Find the random_frame_inserts transformation
            frame_inserts = None
            for transform in transformations:
                if transform.name == 'random_frame_inserts':
                    frame_inserts = transform
                    break
            
            if not frame_inserts:
                print("❌ random_frame_inserts transformation not found!")
                return False
            
            print(f"✅ Found transformation: {frame_inserts.name}")
            
            # Execute the transformation
            print("🚀 Executing random_frame_inserts...")
            command = frame_inserts.execute(input_path, output_path)
            print(f"📝 Generated command: {command}")
            
            # Run the command
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Command executed successfully!")
                
                # Check if output file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"✅ Output file created: {output_path}")
                    print(f"📊 Output file size: {os.path.getsize(output_path)} bytes")
                    print("✅ random_frame_inserts test PASSED!")
                    return True
                else:
                    print("❌ Output file not created or empty!")
                    return False
            else:
                print(f"❌ Command failed with return code: {result.returncode}")
                print(f"📝 STDOUT: {result.stdout}")
                print(f"📝 STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            return False

if __name__ == "__main__":
    print("🎬 Random Frame Inserts Test")
    print("=" * 40)
    
    test_passed = test_random_frame_inserts()
    
    print("\n" + "=" * 40)
    if test_passed:
        print("🎉 Test PASSED!")
        sys.exit(0)
    else:
        print("💥 Test FAILED!")
        sys.exit(1)
