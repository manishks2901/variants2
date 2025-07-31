#!/usr/bin/env python3
"""
Quick test to verify the fixed function commands
"""

import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.ffmpeg_service import FFmpegTransformationService

def test_command_generation():
    """Test command generation for the problematic functions"""
    print("🧪 Testing command generation for fixed functions...")
    
    # Get all transformations
    transformations = FFmpegTransformationService.get_transformations()
    
    # Find the problematic transformations
    problematic_functions = ['random_frame_inserts', 'frame_reordering_segments']
    
    for func_name in problematic_functions:
        print(f"\n📝 Testing {func_name}...")
        
        # Find the transformation
        target_transform = None
        for transform in transformations:
            if transform.name == func_name:
                target_transform = transform
                break
        
        if target_transform:
            # Generate command with dummy paths
            dummy_input = "/tmp/test_input.mp4"
            dummy_output = "/tmp/test_output.mp4"
            
            try:
                command = target_transform.execute(dummy_input, dummy_output)
                print(f"✅ Command generated successfully:")
                print(f"   {command}")
                
                # Check for problematic patterns
                problematic_patterns = ['geq=if', 'between(t,', ':cb=128:cr=128']
                issues_found = []
                
                for pattern in problematic_patterns:
                    if pattern in command:
                        issues_found.append(pattern)
                
                if issues_found:
                    print(f"❌ Found problematic patterns: {issues_found}")
                else:
                    print(f"✅ No problematic patterns found")
                    
            except Exception as e:
                print(f"❌ Error generating command: {str(e)}")
        else:
            print(f"❌ Transformation {func_name} not found!")

if __name__ == "__main__":
    test_command_generation()
