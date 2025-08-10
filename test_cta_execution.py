#!/usr/bin/env python3
"""
Test CTA execution after fix
"""

import sys
import os
sys.path.append('/Users/manishkumarsharma/Documents/variants2')

from app.services.ffmpeg_service import VideoCTAService, CTATransformation

def test_cta_execution():
    """Test CTA execution with the fix"""
    print("🧪 Testing CTA Execution after fix...")
    
    cta_service = VideoCTAService()
    video_duration = 30.0
    
    # Generate CTA transformations
    cta_configs = cta_service.generate_cta_transformations(
        video_duration=video_duration,
        cta_density="medium"
    )
    
    print(f"✅ Generated {len(cta_configs)} CTA transformations")
    
    # Test each CTA transformation
    for i, cta_config in enumerate(cta_configs):
        print(f"\n📝 Testing CTA {i+1}: {cta_config['name']}")
        
        try:
            # Create CTATransformation object
            cta_transform = CTATransformation(cta_config)
            print(f"   ✅ CTATransformation created")
            
            # Test if execute method exists
            if hasattr(cta_transform, 'execute'):
                print(f"   ✅ execute method exists")
                
                # Test execute method
                input_path = "/tmp/test_input.mp4"
                output_path = "/tmp/test_output.mp4"
                
                command = cta_transform.execute(input_path, output_path)
                print(f"   ✅ execute method works")
                print(f"   Command: {command[:80]}...")
                
                # Verify command structure
                if 'ffmpeg' in command and input_path in command and output_path in command:
                    print(f"   ✅ Command structure is correct")
                else:
                    print(f"   ❌ Command structure is incorrect")
                    
            else:
                print(f"   ❌ execute method missing")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✨ CTA Execution Test Complete!")

if __name__ == "__main__":
    test_cta_execution()
