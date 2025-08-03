#!/usr/bin/env python3
"""
Test Script for Comprehensive SSIM Reduction Strategy

This script tests the new SSIM reduction transformations implemented from the strategy table.
"""

import sys
import os
import subprocess
import asyncio
import logging
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.ffmpeg_service import FFmpegTransformationService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_individual_ssim_transformations():
    """Test each individual SSIM transformation from the strategy table"""
    
    # Test input video (use an existing test result)
    input_video = "test_results/01_micro_perspective_warp_enhanced.mp4"
    output_dir = "test_results/ssim_strategy"
    
    if not os.path.exists(input_video):
        print(f"❌ Test video {input_video} not found. Please ensure test video exists.")
        return False
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Individual high-impact SSIM transformations to test
    transformations_to_test = [
        ('high_impact_crop_zoom', 'High Impact Crop + Zoom'),
        ('high_impact_rotation', 'High Impact Rotation'),
        ('high_impact_gaussian_blur', 'High Impact Gaussian Blur'),
        ('high_impact_color_shift_hue', 'High Impact Color Shift'),
        ('high_impact_add_noise', 'High Impact Add Noise'),
        ('high_impact_contrast_brightness', 'High Impact Contrast/Brightness'),
        ('high_impact_flip_transform', 'High Impact Flip Transform'),
        ('high_impact_overlay_texture_pattern', 'High Impact Overlay Pattern'),
        ('medium_impact_trim_start_end', 'Medium Impact Trim'),
        ('medium_impact_insert_black_frame', 'Medium Impact Black Frame'),
        ('advanced_ssim_reduction_pipeline', 'Advanced SSIM Pipeline'),
        ('extreme_ssim_destroyer', 'Extreme SSIM Destroyer')
    ]
    
    successful_tests = 0
    failed_tests = 0
    
    print("🎯 TESTING INDIVIDUAL SSIM REDUCTION TRANSFORMATIONS")
    print("=" * 60)
    
    for method_name, description in transformations_to_test:
        output_file = os.path.join(output_dir, f"ssim_{method_name}.mp4")
        
        try:
            print(f"\nTesting: {description}")
            print(f"Method: {method_name}")
            
            # Get the transformation method
            if hasattr(FFmpegTransformationService, method_name):
                transform_method = getattr(FFmpegTransformationService, method_name)
                cmd = transform_method(input_video, output_file)
                
                print(f"Command: {cmd}")
                
                # Execute the transformation
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(output_file):
                    # Check file size
                    file_size = os.path.getsize(output_file)
                    print(f"✅ SUCCESS - Output: {file_size} bytes")
                    successful_tests += 1
                    
                    # Quick validation check
                    info_cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{output_file}"'
                    info_result = subprocess.run(info_cmd, shell=True, capture_output=True, text=True)
                    if info_result.returncode == 0:
                        print(f"   📊 Video validation passed")
                    else:
                        print(f"   ⚠️ Video validation failed but file exists")
                        
                else:
                    print(f"❌ FAILED - Return code: {result.returncode}")
                    if result.stderr:
                        print(f"   Error: {result.stderr}")
                    failed_tests += 1
                    
            else:
                print(f"❌ FAILED - Method {method_name} not found")
                failed_tests += 1
                
        except Exception as e:
            print(f"❌ FAILED - Exception: {str(e)}")
            failed_tests += 1
    
    print("\n" + "=" * 60)
    print(f"📊 SSIM TRANSFORMATION TEST SUMMARY")
    print(f"Total Tests: {len(transformations_to_test)}")
    print(f"✅ Successful: {successful_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success Rate: {(successful_tests / len(transformations_to_test)) * 100:.1f}%")
    
    return successful_tests > failed_tests

async def test_comprehensive_ssim_strategy():
    """Test the comprehensive SSIM strategy pipeline"""
    
    input_video = "test_results/01_micro_perspective_warp_enhanced.mp4"
    output_dir = "test_results/ssim_strategy"
    
    if not os.path.exists(input_video):
        print(f"❌ Test video {input_video} not found.")
        return False
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🎯 TESTING COMPREHENSIVE SSIM STRATEGY PIPELINE")
    print("=" * 60)
    
    strategy_levels = [
        ('medium', 'Medium Impact Strategy (2-3 high + 1-2 medium, target SSIM < 0.35)'),
        ('high', 'High Impact Strategy (3-4 high impact, target SSIM < 0.30)'),
        ('extreme', 'Extreme Impact Strategy (5-6 high impact, target SSIM < 0.25)')
    ]
    
    successful_strategies = 0
    
    for level, description in strategy_levels:
        output_file = os.path.join(output_dir, f"comprehensive_ssim_{level}.mp4")
        
        try:
            print(f"\nTesting: {description}")
            
            result = FFmpegTransformationService.apply_comprehensive_ssim_strategy(
                input_video, output_file, strategy_level=level
            )
            
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✅ SUCCESS - {result}")
                print(f"   📊 Output: {file_size} bytes")
                successful_strategies += 1
            else:
                print(f"❌ FAILED - Output file not created")
                
        except Exception as e:
            print(f"❌ FAILED - Exception: {str(e)}")
    
    print(f"\n📊 Comprehensive Strategy Success Rate: {successful_strategies}/{len(strategy_levels)}")
    return successful_strategies > 0

async def test_new_strategy_in_main_pipeline():
    """Test the new comprehensive_ssim strategy in the main transformation pipeline"""
    
    input_video = "test_results/01_micro_perspective_warp_enhanced.mp4"
    output_file = "test_results/ssim_strategy/pipeline_comprehensive_ssim.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Test video {input_video} not found.")
        return False
    
    print("\n🎯 TESTING COMPREHENSIVE SSIM IN MAIN PIPELINE")
    print("=" * 60)
    
    try:
        # Test the comprehensive_ssim strategy in the main apply_transformations function
        result = await FFmpegTransformationService.apply_transformations(
            input_video, 
            output_file, 
            strategy="comprehensive_ssim",
            variant_id="ssim_test"
        )
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ SUCCESS - Pipeline test completed")
            print(f"   📊 Output: {file_size} bytes")
            print(f"   🔧 Applied transformations: {len(result)}")
            for transform in result:
                print(f"      - {transform}")
            return True
        else:
            print(f"❌ FAILED - Pipeline output file not created")
            return False
            
    except Exception as e:
        print(f"❌ FAILED - Pipeline exception: {str(e)}")
        return False

async def main():
    """Main test function"""
    print("🎯 COMPREHENSIVE SSIM REDUCTION STRATEGY TEST SUITE")
    print("=" * 80)
    
    total_tests = 0
    passed_tests = 0
    
    # Test 1: Individual transformations
    print("\n🧪 TEST 1: Individual SSIM Transformations")
    if await test_individual_ssim_transformations():
        passed_tests += 1
    total_tests += 1
    
    # Test 2: Comprehensive strategy pipeline
    print("\n🧪 TEST 2: Comprehensive SSIM Strategy Pipeline")
    if await test_comprehensive_ssim_strategy():
        passed_tests += 1
    total_tests += 1
    
    # Test 3: Integration with main pipeline
    print("\n🧪 TEST 3: Integration with Main Pipeline")
    if await test_new_strategy_in_main_pipeline():
        passed_tests += 1
    total_tests += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 FINAL TEST SUMMARY")
    print(f"Total Test Suites: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}")
    print(f"Overall Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! SSIM reduction strategy is ready for use.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
