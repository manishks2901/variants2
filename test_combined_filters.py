import random
import subprocess
import os

def build_combined_ffmpeg_filter() -> str:
    """
    Build a combined FFmpeg filter chain from:
    1. Slight zoom with center crop
    2. Random rotation (±5 degrees)
    3. Faint diagonal texture overlay using geq
    """
    
    # --- 1. Slight Zoom ---
    zoom_filter = "scale=2*trunc(iw*1.05/2):2*trunc(ih*1.05/2),crop=iw:ih"

    # --- 2. Random Rotation ---
    angle = random.uniform(-5, 5)
    rotate_filter = f"rotate={angle:.3f}*PI/180:fillcolor=black"

    # --- 3. Texture Overlay (faint diagonal via geq) ---
    opacity = random.uniform(0.02, 0.06)
    r_formula = f"r(X,Y)+{opacity:.3f}*abs(sin((X+Y)/20))*255"
    g_formula = f"g(X,Y)+{opacity:.3f}*abs(sin((X+Y)/20))*255"
    b_formula = f"b(X,Y)+{opacity:.3f}*abs(sin((X+Y)/20))*255"
    texture_filter = f"geq=r='{r_formula}':g='{g_formula}':b='{b_formula}'"

    # --- Combine all filters into a single -vf filter chain ---
    filter_chain = f"{zoom_filter},{rotate_filter},{texture_filter}"
    return filter_chain


def run_ffmpeg_with_combined_filters(input_path: str, output_path: str) -> None:
    """
    Build and run FFmpeg command with the combined filter chain.
    """
    filter_chain = build_combined_ffmpeg_filter()
    
    cmd = f'ffmpeg -i "{input_path}" -vf "{filter_chain}" -c:a copy -y "{output_path}"'

    print("👉 Running FFmpeg command:\n")
    print(cmd, "\n")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("✅ FFmpeg completed successfully.")
            print(f"✅ Output saved to: {output_path}")
            
            # Check if output file exists and get its size
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                print(f"📁 Output file size: {size:,} bytes ({size/1024/1024:.2f} MB)")
            else:
                print("⚠️ Output file was not created")
                
        else:
            print("❌ FFmpeg failed.")
            print(f"Return code: {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ FFmpeg command timed out after 120 seconds")
    except Exception as e:
        print(f"❌ Error running FFmpeg: {e}")


# === 🔁 Example Usage ===
if __name__ == "__main__":
    # Check if ffmpeg is available
    try:
        ffmpeg_check = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if ffmpeg_check.returncode != 0:
            print("❌ FFmpeg is not available or not working")
            exit(1)
        else:
            print("✅ FFmpeg is available")
    except Exception as e:
        print(f"❌ FFmpeg check failed: {e}")
        exit(1)
    
    # Use test.mp4 as input and create test_output.mp4
    input_video = "test.mp4"
    output_video = "test_output.mp4"
    
    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"❌ Input file '{input_video}' not found!")
        print("📁 Current directory contents:")
        for file in os.listdir("."):
            if file.endswith(('.mp4', '.mov', '.avi', '.mkv')):
                print(f"   🎥 {file}")
        exit(1)
    
    print(f"🎬 Input: {input_video}")
    print(f"📤 Output: {output_video}")
    print()
    
    run_ffmpeg_with_combined_filters(input_video, output_video)
