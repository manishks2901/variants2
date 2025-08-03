#!/usr/bin/env python3
"""
Test script to verify that at least 2 metadata transformations are applied in each variation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.ffmpeg_service import FFmpegTransformationService

def test_metadata_guarantee():
    """Test that all transformation selection methods guarantee at least 2 metadata transformations"""
    
    print("🧪 Testing Metadata Transformation Guarantee")
    print("=" * 60)
    
    # Test 1: select_random_transformations
    print("\n1️⃣ Testing select_random_transformations:")
    for i in range(5):
        selected = FFmpegTransformationService.select_random_transformations()
        metadata_count = len([t for t in selected if t.category == 'metadata'])
        print(f"   Variant {i+1}: {metadata_count} metadata transformations")
        print(f"   Metadata transforms: {[t.name for t in selected if t.category == 'metadata']}")
        assert metadata_count >= 2, f"❌ Only {metadata_count} metadata transformations in select_random_transformations"
    print("   ✅ select_random_transformations: ALL VARIANTS HAVE ≥2 METADATA TRANSFORMS")
    
    # Test 2: select_ssim_focused_transformations  
    print("\n2️⃣ Testing select_ssim_focused_transformations:")
    for i in range(5):
        selected = FFmpegTransformationService.select_ssim_focused_transformations()
        metadata_count = len([t for t in selected if t.category == 'metadata'])
        print(f"   Variant {i+1}: {metadata_count} metadata transformations")
        print(f"   Metadata transforms: {[t.name for t in selected if t.category == 'metadata']}")
        assert metadata_count >= 2, f"❌ Only {metadata_count} metadata transformations in select_ssim_focused_transformations"
    print("   ✅ select_ssim_focused_transformations: ALL VARIANTS HAVE ≥2 METADATA TRANSFORMS")
    
    # Test 3: select_fully_random_transformations
    print("\n3️⃣ Testing select_fully_random_transformations:")
    for i in range(5):
        selected_tuples = FFmpegTransformationService.select_fully_random_transformations(
            num_transformations=15, 
            variant_seed=f"test_variant_{i}"
        )
        selected = [t[0] for t in selected_tuples]  # Extract configs from tuples
        metadata_count = len([t for t in selected if t.category == 'metadata'])
        print(f"   Variant {i+1}: {metadata_count} metadata transformations")
        print(f"   Metadata transforms: {[t.name for t in selected if t.category == 'metadata']}")
        assert metadata_count >= 2, f"❌ Only {metadata_count} metadata transformations in select_fully_random_transformations"
    print("   ✅ select_fully_random_transformations: ALL VARIANTS HAVE ≥2 METADATA TRANSFORMS")
    
    # Test 4: select_mixed_transformations
    print("\n4️⃣ Testing select_mixed_transformations:")
    for i in range(5):
        selected_tuples = FFmpegTransformationService.select_mixed_transformations()
        selected = [t[0] for t in selected_tuples]  # Extract configs from tuples
        metadata_count = len([t for t in selected if t.category == 'metadata'])
        print(f"   Variant {i+1}: {metadata_count} metadata transformations")
        print(f"   Metadata transforms: {[t.name for t in selected if t.category == 'metadata']}")
        assert metadata_count >= 2, f"❌ Only {metadata_count} metadata transformations in select_mixed_transformations"
    print("   ✅ select_mixed_transformations: ALL VARIANTS HAVE ≥2 METADATA TRANSFORMS")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Every transformation selection method guarantees ≥2 metadata transformations")
    print("=" * 60)

def show_available_metadata_transformations():
    """Show all available metadata transformations"""
    print("\n📋 Available Metadata Transformations:")
    print("-" * 40)
    
    all_transforms = FFmpegTransformationService.get_transformations()
    metadata_transforms = [t for t in all_transforms if t.category == 'metadata']
    
    for i, transform in enumerate(metadata_transforms, 1):
        print(f"{i:2d}. {transform.name} (probability: {transform.probability})")
    
    print(f"\nTotal metadata transformations available: {len(metadata_transforms)}")

if __name__ == "__main__":
    show_available_metadata_transformations()
    test_metadata_guarantee()
