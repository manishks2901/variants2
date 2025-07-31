#!/usr/bin/env python3
"""
Display all available transformations in the video processing system
"""
import asyncio
import sys
sys.path.append('/Users/manishkumarsharma/Documents/variants2')

from app.services.ffmpeg_service import FFmpegTransformationService

def display_all_transformations():
    """Display all transformations organized by category"""
    
    print("🎬 VIDEO TRANSFORMATION SYSTEM")
    print("=" * 60)
    print("Total Transformations Available: 33")
    print("=" * 60)
    
    # Get all transformations
    service = FFmpegTransformationService()
    transformations = service.get_transformations()
    
    # Group by category
    categories = {}
    for transform in transformations:
        if transform.category not in categories:
            categories[transform.category] = []
        categories[transform.category].append(transform)
    
    # Display by category
    category_descriptions = {
        'visual': '🎨 VISUAL TRANSFORMATIONS - Modify colors, contrast, brightness, rotation',
        'audio': '🔊 AUDIO TRANSFORMATIONS - Change pitch, tempo, add noise, reorder segments', 
        'structural': '🏗️  STRUCTURAL TRANSFORMATIONS - Modify video length, frame rate, transitions',
        'metadata': '📋 METADATA TRANSFORMATIONS - Strip and randomize file metadata',
        'semantic': '🧠 SEMANTIC TRANSFORMATIONS - Text overlays, sync offset, embedding changes',
        'advanced': '⚙️  ADVANCED TRANSFORMATIONS - Complex filters like LUT, vignette, rotation+crop',
        'enhanced': '🚀 ENHANCED TRANSFORMATIONS - High-impact effects for better bypass (NEW!)'
    }
    
    total_count = 0
    for category_name in ['visual', 'audio', 'structural', 'metadata', 'semantic', 'advanced', 'enhanced']:
        if category_name in categories:
            transforms = categories[category_name]
            print(f"\n{category_descriptions[category_name]}")
            print(f"Count: {len(transforms)} transformations")
            print("-" * 50)
            
            for i, transform in enumerate(transforms, 1):
                probability_pct = int(transform.probability * 100)
                status = "🔥 HIGH" if transform.probability >= 0.8 else "📊 MED" if transform.probability >= 0.5 else "🔻 LOW"
                print(f"{i:2d}. {transform.name:<35} [{status}] {probability_pct:3d}%")
            
            total_count += len(transforms)
    
    print(f"\n{'='*60}")
    print(f"TOTAL TRANSFORMATIONS: {total_count}")
    print(f"{'='*60}")
    
    # Show guaranteed transformations
    guaranteed = service.get_guaranteed_transformation_names()
    print(f"\n🎯 GUARANTEED TRANSFORMATIONS (Always Applied):")
    print(f"Count: {len(guaranteed)}")
    print("-" * 30)
    for i, name in enumerate(guaranteed, 1):
        print(f"{i}. {name}")
    
    # Show enhanced transformations detail
    enhanced_transforms = categories.get('enhanced', [])
    if enhanced_transforms:
        print(f"\n🚀 ENHANCED CATEGORY DETAILS:")
        print("At least 5 enhanced transformations are applied per variant")
        print("-" * 50)
        for transform in enhanced_transforms:
            special_note = ""
            if transform.name == "black_screen_random":
                special_note = " (Single 1-2s black screen)"
            print(f"• {transform.name} - {int(transform.probability*100)}% chance{special_note}")
    
    print(f"\n📊 TRANSFORMATION SELECTION STRATEGY:")
    print("• Minimum 9 transformations per variant")
    print("• At least 5 from ENHANCED category (high-impact effects)")
    print("• Balanced selection from all categories")
    print("• Guaranteed transformations always applied")
    print("• Random selection based on probability weights")
    print("• Black screen: Single 1-2s duration at random position")

if __name__ == "__main__":
    display_all_transformations()
