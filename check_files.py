import os
import subprocess

def check_files():
    print("🔍 Checking files in current directory...\n")
    
    # Check test.mp4
    if os.path.exists("test.mp4"):
        size = os.path.getsize("test.mp4")
        print(f"✅ test.mp4 exists - Size: {size:,} bytes ({size/1024/1024:.2f} MB)")
    else:
        print("❌ test.mp4 not found")
    
    # Check test_output.mp4
    if os.path.exists("test_output.mp4"):
        size = os.path.getsize("test_output.mp4")
        print(f"✅ test_output.mp4 exists - Size: {size:,} bytes ({size/1024/1024:.2f} MB)")
    else:
        print("❌ test_output.mp4 not found")
    
    # Try to get video info for both files
    for filename in ["test.mp4", "test_output.mp4"]:
        if os.path.exists(filename):
            try:
                cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{filename}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    import json
                    info = json.loads(result.stdout)
                    duration = float(info['format'].get('duration', 0))
                    print(f"📹 {filename} - Duration: {duration:.2f} seconds")
                    
                    # Find video stream
                    for stream in info['streams']:
                        if stream['codec_type'] == 'video':
                            width = stream.get('width', 'unknown')
                            height = stream.get('height', 'unknown')
                            print(f"   📐 Resolution: {width}x{height}")
                            break
                else:
                    print(f"❌ Could not get info for {filename}")
            except Exception as e:
                print(f"❌ Error getting info for {filename}: {e}")

if __name__ == "__main__":
    check_files()
