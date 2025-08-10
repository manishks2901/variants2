import subprocess
import os

def test_simple_ffmpeg():
    """Test if ffmpeg is working with a simple command"""
    try:
        # First check if ffmpeg is available
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            print("Version info:")
            print(result.stdout.split('\n')[0])  # Show first line of version
        else:
            print("❌ FFmpeg version check failed")
            return False
    except Exception as e:
        print(f"❌ FFmpeg not found: {e}")
        return False
    
    # Check if test.mp4 exists
    if not os.path.exists("test.mp4"):
        print("❌ test.mp4 not found!")
        return False
    
    print("✅ test.mp4 found")
    
    # Try a simple copy operation
    try:
        print("🔄 Testing simple copy operation...")
        cmd = ['ffmpeg', '-i', 'test.mp4', '-t', '3', '-c', 'copy', 'test_simple_copy.mp4', '-y']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Simple copy test successful")
            if os.path.exists('test_simple_copy.mp4'):
                size = os.path.getsize('test_simple_copy.mp4')
                print(f"📁 Output size: {size:,} bytes")
                # Clean up
                os.remove('test_simple_copy.mp4')
            return True
        else:
            print("❌ Simple copy test failed")
            print("STDERR:", result.stderr[:500])  # Show first 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Simple copy test timed out")
        return False
    except Exception as e:
        print(f"❌ Simple copy test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running FFmpeg diagnostics...\n")
    test_simple_ffmpeg()
