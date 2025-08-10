#!/usr/bin/env python3
"""
Speed test and benchmark script for the optimized video processing pipeline.
Tests the new ultra-fast processing capabilities and compares performance.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('speed_test.log')
    ]
)

def create_test_video():
    """Create a test video for speed testing"""
    test_video = "speed_test_input.mp4"
    
    if os.path.exists(test_video):
        return test_video
    
    try:
        import subprocess
        # Create a 10-second test video
        cmd = [
            "ffmpeg", "-f", "lavfi", "-i", 
            "testsrc=duration=10:size=640x480:rate=30",
            "-c:v", "libx264", "-preset", "fast",
            "-y", test_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(test_video):
            print(f"✅ Created test video: {test_video}")
            return test_video
        else:
            print(f"❌ Failed to create test video: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating test video: {e}")
        return None

def speed_test_variants(input_path: str, num_variants: int = 10):
    """Test variant creation speed with different methods"""
    if not input_path or not os.path.exists(input_path):
        print("⚠️ No valid input file for speed test")
        return None
    
    try:
        from services.ffmpeg_service import FFmpegTransformationService
    except ImportError as e:
        print(f"❌ Could not import FFmpeg service: {e}")
        return None
    
    print(f"🚀 SPEED TEST: Creating {num_variants} variants")
    print(f"📹 Input: {os.path.basename(input_path)}")
    
    results = {}
    
    # Create output directories
    base_output = "speed_test_output"
    os.makedirs(base_output, exist_ok=True)
    
    # Test 1: Standard parallel processing
    print("\n📊 Test 1: Standard parallel processing")
    standard_dir = os.path.join(base_output, "standard")
    os.makedirs(standard_dir, exist_ok=True)
    
    start_time = time.time()
    try:
        standard_results = FFmpegTransformationService.create_multiple_variants_fast(
            input_path=input_path,
            output_dir=standard_dir,
            num_variants=num_variants,
            strategy="standard",
            max_workers=None  # Auto-detect
        )
        standard_time = time.time() - start_time
        standard_success = len([r for r in standard_results if r.get('status') == 'success'])
        
        results['standard'] = {
            'time': standard_time,
            'success_count': standard_success,
            'rate': standard_success / standard_time if standard_time > 0 else 0
        }
        
        print(f"   ⏱️ Time: {standard_time:.2f}s")
        print(f"   ✅ Success: {standard_success}/{num_variants}")
        print(f"   📈 Rate: {results['standard']['rate']:.2f} variants/sec")
        
    except Exception as e:
        print(f"   ❌ Standard test failed: {e}")
        results['standard'] = {'error': str(e)}
    
    # Test 2: Ultra-fast mode
    print("\n🚀 Test 2: Ultra-fast mode")
    ultra_fast_dir = os.path.join(base_output, "ultra_fast")
    os.makedirs(ultra_fast_dir, exist_ok=True)
    
    start_time = time.time()
    try:
        ultra_fast_results = FFmpegTransformationService.create_variants_ultra_fast(
            input_path=input_path,
            output_dir=ultra_fast_dir,
            num_variants=num_variants
        )
        ultra_fast_time = time.time() - start_time
        ultra_fast_success = len([r for r in ultra_fast_results if r.get('status') == 'success'])
        
        results['ultra_fast'] = {
            'time': ultra_fast_time,
            'success_count': ultra_fast_success,
            'rate': ultra_fast_success / ultra_fast_time if ultra_fast_time > 0 else 0
        }
        
        print(f"   ⏱️ Time: {ultra_fast_time:.2f}s")
        print(f"   ✅ Success: {ultra_fast_success}/{num_variants}")
        print(f"   📈 Rate: {results['ultra_fast']['rate']:.2f} variants/sec")
        
    except Exception as e:
        print(f"   ❌ Ultra-fast test failed: {e}")
        results['ultra_fast'] = {'error': str(e)}
    
    # Test 3: Maximum workers test
    print("\n⚡ Test 3: Maximum workers")
    max_workers_dir = os.path.join(base_output, "max_workers")
    os.makedirs(max_workers_dir, exist_ok=True)
    
    optimal_workers = FFmpegTransformationService.get_optimal_worker_count()
    max_workers = min(optimal_workers * 2, 32)
    
    start_time = time.time()
    try:
        max_worker_results = FFmpegTransformationService.create_multiple_variants_fast(
            input_path=input_path,
            output_dir=max_workers_dir,
            num_variants=num_variants,
            strategy="standard",
            max_workers=max_workers,
            ultra_fast_mode=True
        )
        max_worker_time = time.time() - start_time
        max_worker_success = len([r for r in max_worker_results if r.get('status') == 'success'])
        
        results['max_workers'] = {
            'time': max_worker_time,
            'success_count': max_worker_success,
            'rate': max_worker_success / max_worker_time if max_worker_time > 0 else 0,
            'workers_used': max_workers
        }
        
        print(f"   ⏱️ Time: {max_worker_time:.2f}s")
        print(f"   ✅ Success: {max_worker_success}/{num_variants}")
        print(f"   📈 Rate: {results['max_workers']['rate']:.2f} variants/sec")
        print(f"   👥 Workers: {max_workers}")
        
    except Exception as e:
        print(f"   ❌ Max workers test failed: {e}")
        results['max_workers'] = {'error': str(e)}
    
    return results

def analyze_results(results: dict):
    """Analyze and compare speed test results"""
    print("\n" + "="*60)
    print("📊 SPEED TEST ANALYSIS")
    print("="*60)
    
    if not results:
        print("❌ No valid results to analyze")
        return
    
    # Find the fastest method
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        print("❌ All tests failed")
        return
    
    fastest_method = min(valid_results.items(), key=lambda x: x[1]['time'])
    fastest_name, fastest_data = fastest_method
    
    print(f"🏆 FASTEST METHOD: {fastest_name.upper()}")
    print(f"   ⏱️ Time: {fastest_data['time']:.2f}s")
    print(f"   📈 Rate: {fastest_data['rate']:.2f} variants/sec")
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    baseline_time = valid_results.get('standard', {}).get('time', 0)
    
    for method, data in valid_results.items():
        if 'error' not in data:
            speedup = baseline_time / data['time'] if data['time'] > 0 and baseline_time > 0 else 1
            print(f"   {method.upper()}: {data['time']:.2f}s ({speedup:.2f}x speedup)")
    
    # Speed recommendations
    print(f"\n💡 SPEED RECOMMENDATIONS:")
    if 'ultra_fast' in valid_results:
        ultra_rate = valid_results['ultra_fast']['rate']
        print(f"   🚀 Ultra-fast mode: {ultra_rate:.1f} variants/sec")
        print(f"   📊 Best for high-volume processing")
    
    if 'max_workers' in valid_results:
        max_rate = valid_results['max_workers']['rate']
        workers = valid_results['max_workers'].get('workers_used', 'unknown')
        print(f"   ⚡ Max workers ({workers}): {max_rate:.1f} variants/sec")
        print(f"   🔧 Best for systems with many CPU cores")
    
    # Calculate theoretical maximum throughput
    if valid_results:
        best_rate = max(data['rate'] for data in valid_results.values())
        hourly_throughput = best_rate * 3600
        daily_throughput = hourly_throughput * 24
        
        print(f"\n🎯 THEORETICAL MAXIMUM THROUGHPUT:")
        print(f"   📈 Per hour: {hourly_throughput:.0f} variants")
        print(f"   📅 Per day: {daily_throughput:.0f} variants")

def main():
    """Run comprehensive speed tests"""
    print("⚡ VIDEO PROCESSING SPEED TEST")
    print("="*50)
    
    # Create or find test video
    test_video = create_test_video()
    
    if not test_video:
        # Look for existing video files
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            for test_file in Path('test_results').rglob(f'*{ext}'):
                test_video = str(test_file)
                print(f"✅ Found existing video: {os.path.basename(test_video)}")
                break
            if test_video:
                break
    
    if not test_video:
        print("⚠️ No test video available")
        print("   To run speed tests:")
        print("   1. Install FFmpeg and ensure it's in PATH")
        print("   2. Place a video file in test_results/ directory")
        return False
    
    # Run speed tests with different numbers of variants
    test_sizes = [5, 10]  # Start small for testing
    
    for num_variants in test_sizes:
        print(f"\n🎬 Testing with {num_variants} variants")
        print("-" * 40)
        
        results = speed_test_variants(test_video, num_variants)
        
        if results:
            analyze_results(results)
        
        print("\n" + "="*60)
    
    print("🎉 Speed tests completed!")
    print("📁 Results saved in speed_test_output/")
    print("📝 Logs saved in speed_test.log")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
