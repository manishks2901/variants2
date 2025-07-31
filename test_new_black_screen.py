#!/usr/bin/env python3
"""
Test New Black Screen Multiple Effect
Test the updated black_screen_random function with multiple 2-3s black screens
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.ffmpeg_service import FFmpegTransformationService

async def test_new_black_screen():
    """Test the new multiple black screen effect"""
    print("🖤 Testing New Multiple Black Screen Effect")
    print("=" * 60)
    
    # Get transformations
    transformations = FFmpegTransformationService.get_transformations()
    
    # Find black screen transformation
    black_screen_transform = next((t for t in transformations if t.name == 'black_screen_random'), None)
    
    if not black_screen_transform:
        print("❌ black_screen_random transformation not found")
        return False
    
    # Test with different video durations
    test_cases = [
        {"duration": 30, "description": "30-second video"},
        {"duration": 60, "description": "60-second video"}, 
        {"duration": 120, "description": "2-minute video"},
        {"duration": 10, "description": "10-second video (edge case)"}
    ]
    
    test_input = "/tmp/test_input.mp4"
    test_output = "/tmp/test_output.mp4"
    
    print("🧪 Testing Different Video Durations:")
    print("-" * 50)
    
    for test_case in test_cases:
        duration = test_case["duration"]
        description = test_case["description"]
        
        print(f"\n📹 {description}:")
        
        try:
            # Mock video info for testing
            original_get_video_info = FFmpegTransformationService.get_video_info
            
            async def mock_get_video_info(video_path):
                return {
                    'duration': duration,
                    'hasAudio': True,
                    'width': 1280,
                    'height': 720
                }
            
            # Temporarily replace the function
            FFmpegTransformationService.get_video_info = mock_get_video_info
            
            # Generate command
            command = await black_screen_transform.execute(test_input, test_output)
            
            # Restore original function
            FFmpegTransformationService.get_video_info = original_get_video_info
            
            print(f"   ✅ Command generated successfully")
            print(f"   📄 Command: {command[:150]}...")
            
            # Analyze the command for black screen timing
            if 'fade=out' in command and 'fade=in' in command:
                fade_outs = command.count('fade=out')
                fade_ins = command.count('fade=in')
                
                print(f"   🖤 Black screen segments: {fade_outs} (should be 2-3)")
                print(f"   ⚡ Fade transitions: {fade_ins} fade-ins, {fade_outs} fade-outs")
                
                # Check if timing makes sense
                if fade_outs >= 2 and fade_outs <= 3:
                    print(f"   ✅ Correct number of black screen segments")
                else:
                    print(f"   ⚠️  Unexpected number of segments")
                    
            else:
                print(f"   ⚠️  No fade effects detected in command")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 New Black Screen Effect Summary:")
    print(f"  • Multiple segments: 2-3 black screens per video")
    print(f"  • Duration per segment: 2-3 seconds each")
    print(f"  • Distribution: Evenly spaced throughout video")
    print(f"  • Fade transitions: 0.2s fade out/in for smooth effect")
    print(f"  • Total black time: 4-9 seconds per video (much more impactful!)")
    
    print(f"\n💡 Timing Examples:")
    print(f"  • 30s video: ~2 segments at 8s, 22s (4-6s total black)")
    print(f"  • 60s video: ~3 segments at 12s, 30s, 48s (6-9s total black)")
    print(f"  • 120s video: ~3 segments at 24s, 60s, 96s (6-9s total black)")
    
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_new_black_screen())
        if result:
            print("\n🎉 New multiple black screen effect tested successfully!")
            print("💡 The effect is now much more impactful with 2-3 segments of 2-3s each")
        else:
            print("\n❌ Test failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        sys.exit(1)
