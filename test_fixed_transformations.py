#!/usr/bin/env python3
"""
Test script for the fixed FFmpeg transformations
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

def test_transformation(transformation_name: str):
    """Test a specific transformation"""
    print(f"🧪 Testing {transformation_name}...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "test_input.mp4")
        output_path = os.path.join(temp_dir, "test_output.mp4")
        
        try:
            # Create test video
            create_test_video(input_path, duration=3)
            
            # Get all transformations and find the target transformation
            transformations = FFmpegTransformationService.get_transformations()
            
            # Find the transformation
            target_transform = None
            for transform in transformations:
                if transform.name == transformation_name:
                    target_transform = transform
                    break
            
            if not target_transform:
                print(f"❌ {transformation_name} transformation not found!")
                return False
            
            print(f"✅ Found transformation: {target_transform.name}")
            
            # Execute the transformation
            print(f"🚀 Executing {transformation_name}...")
            command = target_transform.execute(input_path, output_path)
            print(f"📝 Generated command: {command}")
            
            # Run the command
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Command executed successfully!")
                
                # Check if output file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"✅ Output file created")
                    print(f"📊 Output file size: {os.path.getsize(output_path)} bytes")
                    print(f"✅ {transformation_name} test PASSED!")
                    return True
                else:
                    print("❌ Output file not created or empty!")
                    return False
            else:
                print(f"❌ Command failed with return code: {result.returncode}")
                print(f"📝 STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            return False

if __name__ == "__main__":
    print("🎬 FFmpeg Transformations Fix Test")
    print("=" * 50)
    
    # Test the two functions that were causing errors
    transformations_to_test = [
        'random_frame_inserts',
        'black_screen_random'
    ]
    
    results = {}
    
    for transform_name in transformations_to_test:
        print(f"\n{'='*20} {transform_name.upper()} {'='*20}")
        results[transform_name] = test_transformation(transform_name)
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    
    all_passed = True
    for transform_name, passed in results.items():
        status = '✅ PASSED' if passed else '❌ FAILED'
        print(f"{transform_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests PASSED! Both error-causing transformations are now fixed!")
        sys.exit(0)
    else:
        print("💥 Some tests FAILED!")
        sys.exit(1)
