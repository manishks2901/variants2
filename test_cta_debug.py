#!/usr/bin/env python3
"""
Debug script to test CTA functionality
"""

import sys
import os
sys.path.append('/Users/manishkumarsharma/Documents/variants2')

from app.services.ffmpeg_service import VideoCTAService, CTAConfig, CTAType, CTAAnimation, CTATransformation

def test_cta_generation():
    """Test CTA generation"""
    print("🧪 Testing CTA Generation...")
    
    cta_service = VideoCTAService()
    video_duration = 30.0  # 30 second video
    
    # Test basic CTA generation
    try:
        cta_transformations = cta_service.generate_cta_transformations(
            video_duration=video_duration,
            cta_density="medium"
        )
        
        print(f"✅ Generated {len(cta_transformations)} CTA transformations")
        
        for i, cta_config in enumerate(cta_transformations):
            print(f"\n📝 CTA {i+1}:")
            print(f"   Name: {cta_config['name']}")
            print(f"   Category: {cta_config['category']}")
            print(f"   Description: {cta_config['description']}")
            print(f"   Start Time: {cta_config.get('start_time', 'N/A')}")
            print(f"   Duration: {cta_config.get('duration', 'N/A')}")
            print(f"   Filter: {cta_config['filter'][:100]}{'...' if len(cta_config['filter']) > 100 else ''}")
            
            # Test CTATransformation creation
            try:
                cta_transform = CTATransformation(cta_config)
                print(f"   ✅ CTATransformation created successfully")
            except Exception as e:
                print(f"   ❌ CTATransformation failed: {e}")
        
        return cta_transformations
        
    except Exception as e:
        print(f"❌ CTA generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_cta_filter_syntax():
    """Test specific CTA filter syntax"""
    print("\n🔍 Testing CTA Filter Syntax...")
    
    cta_service = VideoCTAService()
    
    # Test different text types
    test_texts = [
        "Subscribe for More!",
        "Like & Subscribe 👍",
        "Comment below 💬",
        "Share with friends!",
        "Watch next video ➡️",
        "Amazing content! 💯",
        "Special characters: :;'\"[]{}()",
        "Unicode test: 你好世界"
    ]
    
    for text in test_texts:
        try:
            cleaned = cta_service._escape_text_for_ffmpeg(text)
            print(f"✅ '{text}' -> '{cleaned}'")
        except Exception as e:
            print(f"❌ '{text}' failed: {e}")

def test_position_parsing():
    """Test position parsing"""
    print("\n📍 Testing Position Parsing...")
    
    cta_service = VideoCTAService()
    
    test_positions = [
        ("center", 1920),
        ("50%", 1920),
        ("100px", 1920),
        ("1000", 1920),
        ("95%", 1080),
        ("invalid", 1920)
    ]
    
    for pos, dimension in test_positions:
        try:
            result = cta_service._parse_position(pos, dimension)
            print(f"✅ Position '{pos}' -> {result} (dimension: {dimension})")
        except Exception as e:
            print(f"❌ Position '{pos}' failed: {e}")

def test_text_style():
    """Test text style creation"""
    print("\n🎨 Testing Text Style Creation...")
    
    cta_service = VideoCTAService()
    
    test_cta = CTAConfig(
        text="Test CTA",
        cta_type=CTAType.OVERLAY,
        start_time=5.0,
        duration=3.0,
        position=("50%", "10%"),
        font_size=24,
        font_color="white",
        background_color="rgba(0,0,0,0.7)"
    )
    
    try:
        style = cta_service._create_text_style(test_cta)
        print(f"✅ Text style: {style}")
    except Exception as e:
        print(f"❌ Text style creation failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting CTA Debug Tests...")
    
    # Run all tests
    test_cta_generation()
    test_cta_filter_syntax()
    test_position_parsing()
    test_text_style()
    
    print("\n✨ CTA Debug Tests Complete!")
