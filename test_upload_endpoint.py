#!/usr/bin/env python3
"""
Test script for the /api/v1/upload endpoint
"""

import requests
import tempfile
import subprocess
import os
import json

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

def test_upload_endpoint():
    """Test the upload endpoint"""
    print("🧪 Testing /api/v1/upload endpoint...")
    
    # API configuration
    BASE_URL = "http://127.0.0.1:8000"
    UPLOAD_URL = f"{BASE_URL}/api/v1/upload"
    
    # Create temporary test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_video_path = temp_file.name
    
    try:
        # Create test video
        print("📹 Creating test video...")
        create_test_video(temp_video_path, duration=3)
        
        # Prepare the request
        print("📤 Uploading to API...")
        
        files = {
            'file': ('test_video.mp4', open(temp_video_path, 'rb'), 'video/mp4')
        }
        
        data = {
            'priority': 'normal',
            'variations': '1',
            'min_transformations': '5',
            'enable_punchlines': 'false',
            'punchline_variant': '1'
        }
        
        # Make the request
        response = requests.post(UPLOAD_URL, files=files, data=data, timeout=30)
        
        # Close file handle
        files['file'][1].close()
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📝 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("✅ Upload successful!")
            print(f"📋 Response: {json.dumps(response_data, indent=2)}")
            
            job_id = response_data.get('job_id')
            if job_id:
                print(f"🆔 Job ID: {job_id}")
                return job_id
            else:
                print("⚠️ No job_id in response")
                return None
                
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"📝 Error response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - Is the server running on 127.0.0.1:50501?")
        return None
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return None
    finally:
        # Cleanup
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

def test_health_check():
    """Test the health check endpoint"""
    print("\n🏥 Testing health check...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"📋 Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_punchline_status():
    """Test the punchline status endpoint"""
    print("\n🎯 Testing punchline status...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/api/v1/punchline-status", timeout=10)
        if response.status_code == 200:
            print("✅ Punchline status check passed")
            print(f"📋 Response: {response.json()}")
            return True
        else:
            print(f"❌ Punchline status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Punchline status check failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🎬 API Endpoint Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    health_ok = test_health_check()
    
    # Test 2: Punchline status
    punchline_ok = test_punchline_status()
    
    # Test 3: Upload endpoint (only if server is responding)
    upload_ok = False
    job_id = None
    if health_ok:
        job_id = test_upload_endpoint()
        upload_ok = job_id is not None
    else:
        print("\n⚠️ Skipping upload test - server not responding")
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"Health Check: {'✅ PASSED' if health_ok else '❌ FAILED'}")
    print(f"Punchline Status: {'✅ PASSED' if punchline_ok else '❌ FAILED'}")
    print(f"Upload Endpoint: {'✅ PASSED' if upload_ok else '❌ FAILED'}")
    
    if job_id:
        print(f"🆔 Created Job ID: {job_id}")
        print(f"🔗 Check status: http://127.0.0.1:8000/api/v1/jobs/{job_id}")
    
    if health_ok and upload_ok:
        print("🎉 All critical tests PASSED!")
        exit(0)
    else:
        print("💥 Some tests FAILED!")
        exit(1)
