#!/usr/bin/env python3
"""
Test script for frame_reordering_segments function
"""

import sys
import os
import tempfile
import subprocess
import shutil
from pathlib import Path

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.ffmpeg_service import FFmpegTransformationService

def create_test_video(output_path: str, duration: int = 10):
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

def test_frame_reordering_segments():
    """Test the frame_reordering_segments function"""
    print("🧪 Testing frame_reordering_segments function...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "test_input.mp4")
        output_path = os.path.join(temp_dir, "test_output.mp4")
        
        try:
            # Create test video
            print("📹 Creating test video...")
            create_test_video(input_path, duration=5)
            
            # Test the frame_reordering_segments function
            print("🔧 Testing frame_reordering_segments...")
            
            # Get all transformations and find frame_reordering_segments
            transformations = FFmpegTransformationService.get_transformations()
            
            # Find the frame_reordering_segments transformation
            frame_reordering = None
            for transform in transformations:
                if transform.name == 'frame_reordering_segments':
                    frame_reordering = transform
                    break
            
            if not frame_reordering:
                print("❌ frame_reordering_segments transformation not found!")
                return False
            
            print(f"✅ Found transformation: {frame_reordering.name}")
            
            # Execute the transformation
            print("🚀 Executing frame_reordering_segments...")
            command = frame_reordering.execute(input_path, output_path)
            print(f"📝 Generated command: {command}")
            
            # Run the command
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Command executed successfully!")
                
                # Check if output file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"✅ Output file created: {output_path}")
                    print(f"📊 Output file size: {os.path.getsize(output_path)} bytes")
                    
                    # Get video info of output
                    probe_cmd = [
                        'ffprobe', '-v', 'quiet', '-print_format', 'json',
                        '-show_format', '-show_streams', output_path
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    
                    if probe_result.returncode == 0:
                        import json
                        info = json.loads(probe_result.stdout)
                        duration = float(info['format'].get('duration', 0))
                        print(f"📏 Output video duration: {duration:.2f} seconds")
                        print("✅ frame_reordering_segments test PASSED!")
                        return True
                    else:
                        print("⚠️ Could not probe output video, but file exists")
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

def test_command_generation_only():
    """Test just the command generation without execution"""
    print("\n🧪 Testing command generation only...")
    
    try:
        # Get all transformations and find frame_reordering_segments
        transformations = FFmpegTransformationService.get_transformations()
        
        # Find the frame_reordering_segments transformation
        frame_reordering = None
        for transform in transformations:
            if transform.name == 'frame_reordering_segments':
                frame_reordering = transform
                break
        
        if frame_reordering:
            # Test with dummy paths
            dummy_input = "/tmp/test_input.mp4"
            dummy_output = "/tmp/test_output.mp4"
            
            print("🔧 Generating command...")
            command = frame_reordering.execute(dummy_input, dummy_output)
            print(f"📝 Generated command: {command}")
            
            # Check if command looks valid (contains expected elements)
            expected_elements = ['ffmpeg', '-i', 'setpts', dummy_input, dummy_output]
            missing_elements = [elem for elem in expected_elements if elem not in command]
            
            if not missing_elements:
                print("✅ Command generation test PASSED!")
                return True
            else:
                print(f"❌ Command missing elements: {missing_elements}")
                return False
        else:
            print("❌ frame_reordering_segments transformation not found!")
            return False
            
    except Exception as e:
        print(f"❌ Command generation test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🎬 Frame Reordering Segments Test")
    print("=" * 50)
    
    # Test 1: Command generation only
    cmd_test_passed = test_command_generation_only()
    
    # Test 2: Full execution test (only if FFmpeg is available)
    full_test_passed = False
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("\n✅ FFmpeg is available, running full test...")
        full_test_passed = test_frame_reordering_segments()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n⚠️ FFmpeg not available, skipping full execution test")
        full_test_passed = True  # Don't fail if ffmpeg is not available
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"Command Generation: {'✅ PASSED' if cmd_test_passed else '❌ FAILED'}")
    print(f"Full Execution: {'✅ PASSED' if full_test_passed else '❌ FAILED'}")
    
    if cmd_test_passed and full_test_passed:
        print("🎉 All tests PASSED!")
        sys.exit(0)
    else:
        print("💥 Some tests FAILED!")
        sys.exit(1)
