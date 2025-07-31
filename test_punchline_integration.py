#!/usr/bin/env python3
"""
Test script for punchline generation functionality
"""

import os
import sys
import tempfile
import asyncio
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.punchline_service import VideoPunchlineGenerator

def test_punchline_service():
    """Test the punchline service initialization and configuration"""
    print("🔍 Testing Punchline Service...")
    
    # Initialize the service
    service = VideoPunchlineGenerator()
    
    # Check if API keys are available
    is_available = service.is_available()
    print(f"📊 Punchline Service Available: {is_available}")
    
    if is_available:
        print("✅ API keys are configured correctly")
        print(f"🎨 Available variants: {list(service.variants.keys())}")
        
        # Show variant styles
        for variant_name, style in service.variants.items():
            print(f"  {variant_name}: {style}")
    else:
        print("❌ API keys not configured. Please check your .env file:")
        print("   - ELEVENLABS_API_KEY")
        print("   - GROQ_API_KEY")
    
    return is_available

async def test_video_info(test_video_path):
    """Test video information extraction"""
    if not os.path.exists(test_video_path):
        print(f"❌ Test video not found: {test_video_path}")
        return False
    
    print(f"\n🎬 Testing video info extraction for: {test_video_path}")
    
    service = VideoPunchlineGenerator()
    try:
        video_size, duration = service.get_video_info(test_video_path)
        print(f"📐 Video size: {video_size}")
        print(f"⏱️  Duration: {duration} seconds")
        return True
    except Exception as e:
        print(f"❌ Video info extraction failed: {e}")
        return False

def find_test_video():
    """Find a test video file in the workspace"""
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    
    # Check current directory for video files
    for ext in video_extensions:
        for video_file in Path('.').glob(f'*{ext}'):
            return str(video_file)
    
    return None

async def main():
    """Main test function"""
    print("🚀 Punchline Generation Test Suite")
    print("=" * 50)
    
    # Test 1: Service initialization
    service_available = test_punchline_service()
    
    # Test 2: Find test video
    print("\n🔍 Looking for test video files...")
    test_video = find_test_video()
    
    if test_video:
        print(f"📹 Found test video: {test_video}")
        
        # Test 3: Video info extraction
        video_info_success = await test_video_info(test_video)
        
        if service_available and video_info_success:
            print(f"\n🎯 Ready to test punchline generation!")
            print(f"💡 Try uploading {test_video} to the API with enable_punchlines=true")
        else:
            print(f"\n⚠️  Some tests failed. Check configuration before using punchlines.")
    else:
        print("❌ No test video files found in current directory")
        print("💡 Add a .mp4, .mov, .avi, or .mkv file to test video processing")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(main())
