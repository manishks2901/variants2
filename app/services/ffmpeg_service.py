"""
FFmpeg Video Transformation Service - BALANCED HIGH VARIATION STRATEGY

This service provides comprehensive video transformations for Instagram copyright bypass
with BALANCED VARIATION designed to reduce total risk score to ~50 while preserving originality.

BALANCED HIGH VARIATION STRATEGY:
- Targeting MSE 20-50, SSIM 0.30-0.35, balanced correlation scores
- Moderate transformation aggressiveness to preserve video quality
- 16-24 transformations per variant (balanced for quality + variation)
- Conservative probability for aggressive effects (0.3-0.4)
- Standard probability for core transformations (0.5-0.6)
- PRESERVES VIDEO ORIGINALITY AND WATCHABILITY

KEY BALANCE PRINCIPLES:
✅ Variation: Sufficient to break detection algorithms
✅ Quality: Maintains original video content integrity  
✅ Watchability: Videos remain natural and viewable
✅ Effectiveness: Still achieves ~50 total risk target
✅ Safety: No extreme distortions that damage content

NEW TEMPORAL FEATURES:
- Random geometric warps applied at random timestamps (3-6 points per video)
- Cut/jitter effects with micro-cuts, speed variations, and position jitter (4-6 points)
- Random overlay effects including text, shapes, noise, and gradients (3-5 points)
- Motion blur effects with directional, radial, zoom, and gaussian blur (3-4 points)

NEW ORB FEATURE-BREAKING STRATEGY (Advanced Computer Vision Bypass):
1. **Micro Perspective Warp** (±2-4px corners) - Disrupts keypoint geometry
2. **Frame Jittering** (1-2px shifts every 3-5 frames) - Random keypoint displacement
3. **Fine Noise Overlay** (σ=1.0-3.0) - Scrambles ORB descriptors
4. **Subtle Logo/Texture Overlay** (3-7% opacity) - Confuses feature matching
5. **Dynamic Zoom Oscillation** (1.01x-1.03x) - Changes spatial relationships
6. **Dynamic Rotation** (±1.0-1.5° every 2-3s) - Alters keypoint structure
7. **Slight Color Variation** (±2-4% saturation, ±1-2° hue) - Affects contrast areas
8. **Line Sketch Filter** (2-5% opacity) - Adds artificial lines to mislead ORB
9. **Randomized Transform Sets** - Different subsets every 5s to avoid patterns

BALANCED HIGH VARIATION ADDITIONS:
10. **Balanced Color Shift** - Multi-stage color transformations (conservative)
11. **Balanced Geometric Distortion** - Subtle barrel, pincushion, trapezoid effects
12. **Balanced Temporal Manipulation** - Gentle speed variation, frame selection
13. **Reduced Audio Manipulation** - EQ shifts, phase adjustments (quality-preserving, REDUCED ranges)

ORIGINALITY PRESERVATION MEASURES:
- Hue shifts: ±20° (vs extreme ±40°)
- Saturation: 0.75-1.25 (vs extreme 0.5-1.5)
- Rotation: ±1.5° (vs extreme ±4°)  
- Noise: 0.02-0.05 (vs extreme 0.08-0.15)
- Audio pitch: ±4% (vs extreme ±12%, REDUCED for quality)
- Crop percentage: 3-8% (vs extreme 8-15%)

These transformations are applied at random points throughout the video rather than
affecting the entire video, providing natural variation while maximizing preservation.

The balanced approach specifically targets detection algorithms while maintaining
the original video's visual and audio quality for human viewers.

Example usage for a 50s video:
- Geometric warps at: 3s, 13s, 19s, 35s, 42s
- Jitter effects at: 7s, 15s, 28s, 41s
- Overlays at: 5s, 22s, 38s, 47s
- Motion blur at: 11s, 26s, 44s
- ORB-breaking: 3-5 transformations applied globally
- Balanced variations: 3-6 additional transformations for controlled variation

Each transformation type uses the helper function get_random_transformation_points()
to ensure proper spacing and avoid overlapping effects.
"""
import secrets
import uuid
import random
import ffmpeg
import subprocess
import asyncio
import math
import json
import os
import json
import math
import logging
import subprocess
import shutil
import time
import tempfile
import datetime as dt
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import hashlib
import cv2
import numpy as np
# MoviePy import is optional - using ffprobe for video duration instead
try:
    from moviepy.editor import VideoFileClip  # type: ignore
except ImportError:
    VideoFileClip = None

class TransformationConfig:
    def __init__(self, name: str, probability: float, execute_func: Callable, category: str = "general", supports_temporal: bool = False):
        self.name = name
        self.probability = probability
        self.execute = execute_func
        self.category = category
        self.supports_temporal = supports_temporal  # Whether this transformation can be applied to specific time segments

class AdvancedTransformationMetrics:
    """
    Advanced video transformation metrics for comprehensive content variation
    Based on pHash, SSIM, Audio Fingerprinting, and other detection methods
    
    BALANCED HIGH VARIATION STRATEGY: Targeting ~50 total risk with originality preservation
    """
    
    @staticmethod
    @staticmethod
    def get_phash_safe_range():
        """pHash Hamming Distance: > 20-30 per keyframe (BALANCED for variation + originality)"""
        return random.uniform(20, 30)
    
    @staticmethod
    def get_ssim_safe_range():
        """SSIM Structural Similarity: < 0.30-0.35 average (BALANCED for detection + quality)"""
        return random.uniform(0.30, 0.35)
    
    @staticmethod
    def get_color_histogram_safe_range():
        """Color Histogram correlation: < 0.40-0.50 (BALANCED for variation + natural colors)"""
        return random.uniform(0.40, 0.50)
    
    @staticmethod
    def get_frame_entropy_increase():
        """Frame Entropy increase: 5-8% (BALANCED for variation + quality)"""
        return random.uniform(1.05, 1.08)
    
    @staticmethod
    def get_embedding_distance_change():
        """CLIP Embedding distance change: >= 0.25 (BALANCED for detection + content preservation)"""
        return random.uniform(0.25, 0.35)
    
    @staticmethod
    def get_audio_fingerprint_confidence():
        """Audio fingerprint match confidence: < 45% (BALANCED for detection + audio quality)"""
        return random.uniform(0.25, 0.45)
    
    @staticmethod
    def get_pitch_shift_range():
        """Pitch shift: ±3-4% (REDUCED for better audio quality)"""
        return random.uniform(-0.04, 0.04)
    
    @staticmethod
    def get_tempo_shift_range():
        """Tempo shift: ±2-3% (REDUCED for better natural flow)"""
        return random.uniform(-0.03, 0.03)
    
    @staticmethod
    def get_audio_noise_level():
        """Audio layering noise: -30dB to -40dB (REDUCED noise level)"""
        return random.uniform(0.001, 0.003)  # Reduced amplitude
    
    @staticmethod
    def get_video_trim_range():
        """Video length trim: ±0.5-2 sec (BALANCED for variation + content preservation)"""
        return random.uniform(0.5, 2.0)
    
    @staticmethod
    def get_frame_rate_change():
        """Frame rate change: ±0.5-1 fps (BALANCED for detection + smooth playback)"""
        return random.uniform(-1.0, 1.0)
    
    @staticmethod
    def get_audio_video_sync_offset():
        """Audio-video sync offset: ±50-150ms (REDUCED for better sync preservation)"""
        return random.uniform(-0.15, 0.15)

class FFmpegTransformationService:
    @staticmethod
    def slight_zoom(input_path: str, output_path: str) -> str:
        """Slight zoom (scale=1.05) with even dimensions"""
        return f'ffmpeg -i "{input_path}" -vf "scale=2*trunc(iw*1.05/2):2*trunc(ih*1.05/2),crop=iw:ih" -c:a copy -y "{output_path}"'

    @staticmethod
    def random_rotate(input_path: str, output_path: str) -> str:
        """Random rotate (±5°)"""
        angle = random.uniform(-5, 5)
        return f'ffmpeg -i "{input_path}" -vf "rotate={angle}*PI/180:fillcolor=black" -c:a copy -y "{output_path}"'

    @staticmethod
    def faint_diagonal_texture_overlay(input_path: str, output_path: str) -> str:
        """Add faint diagonal texture overlay"""
        opacity = random.uniform(0.02, 0.06)
        r_formula = f"r(X,Y)+{opacity}*abs(sin((X+Y)/20))*255"
        g_formula = f"g(X,Y)+{opacity}*abs(sin((X+Y)/20))*255"
        b_formula = f"b(X,Y)+{opacity}*abs(sin((X+Y)/20))*255"
        return f'ffmpeg -i "{input_path}" -vf "geq=r=\'{r_formula}\':g=\'{g_formula}\':b=\'{b_formula}\'" -c:a copy -y "{output_path}"'

    @staticmethod
    def color_filter_hue(input_path: str, output_path: str) -> str:
        """Apply color filter (hue=s=0 for grayscale)"""
        return f'ffmpeg -i "{input_path}" -vf "hue=s=0" -c:a copy -y "{output_path}"'

    @staticmethod
    def flip_horizontal(input_path: str, output_path: str) -> str:
        """Flip horizontal"""
        return f'ffmpeg -i "{input_path}" -vf "hflip" -c:a copy -y "{output_path}"'

    @staticmethod
    def slight_blur_or_noise(input_path: str, output_path: str) -> str:
        """Slight blur or noise"""
        if random.random() < 0.5:
            blur_strength = random.uniform(0.5, 1.2)
            return f'ffmpeg -i "{input_path}" -vf "boxblur={blur_strength}:{blur_strength/2}" -c:a copy -y "{output_path}"'
        else:
            noise_strength = random.uniform(0.01, 0.03)
            return f'ffmpeg -i "{input_path}" -vf "noise=alls={noise_strength}:allf=t" -c:a copy -y "{output_path}"'
    
    # 🎯 HIGH-IMPACT SSIM REDUCTION STRATEGIES
    # Based on comprehensive SSIM reduction research - targeting SSIM < 0.30
    
    @staticmethod
    def enhanced_crop_zoom_ssim(input_path: str, output_path: str) -> str:
        """Enhanced crop + zoom for maximum SSIM reduction (High Impact)"""
        crop_factor = random.uniform(0.85, 0.95)  # Crop 5-15%
        zoom_factor = random.uniform(1.02, 1.08)  # Zoom back up
        
        # Crop then zoom back up to disrupt structural similarity
        return f'ffmpeg -i "{input_path}" -vf "crop=iw*{crop_factor}:ih*{crop_factor},scale=2*trunc(iw*{zoom_factor}/2):2*trunc(ih*{zoom_factor}/2)" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def aggressive_gaussian_blur_ssim(input_path: str, output_path: str) -> str:
        """Aggressive Gaussian blur for texture disruption (High Impact)"""
        sigma = random.uniform(1.0, 2.5)  # Strong blur for SSIM reduction
        return f'ffmpeg -i "{input_path}" -vf "gblur=sigma={sigma}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def enhanced_rotation_ssim(input_path: str, output_path: str) -> str:
        """Enhanced rotation for structural distortion (High Impact)"""
        angle = random.uniform(-8, 8)  # More aggressive rotation
        return f'ffmpeg -i "{input_path}" -vf "rotate={angle}*PI/180:fillcolor=black@0.0" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def aggressive_hue_saturation_shift(input_path: str, output_path: str) -> str:
        """Aggressive hue/saturation shift for color info disruption (High Impact)"""
        hue_degrees = random.uniform(-30, 30)
        saturation = random.uniform(0.6, 1.4)
        brightness = random.uniform(-0.1, 0.1)
        
        return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_degrees}:s={saturation},eq=brightness={brightness}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def pattern_disruption_noise(input_path: str, output_path: str) -> str:
        """Pattern-disrupting noise overlay (High Impact)"""
        noise_strength = random.uniform(15, 30)  # Strong noise for pattern disruption
        temporal_noise = random.choice(['t', 'f'])  # Temporal or constant noise
        
        return f'ffmpeg -i "{input_path}" -vf "noise=alls={noise_strength}:allf={temporal_noise}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def contrast_brightness_disruption(input_path: str, output_path: str) -> str:
        """Contrast/brightness disruption for luminance alteration (High Impact)"""
        contrast = random.uniform(0.8, 1.3)
        brightness = random.uniform(-0.05, 0.08)
        gamma = random.uniform(0.9, 1.15)
        
        return f'ffmpeg -i "{input_path}" -vf "eq=contrast={contrast}:brightness={brightness}:gamma={gamma}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def strategic_flip_transform(input_path: str, output_path: str) -> str:
        """Strategic flip for complete structural reversal (High Impact)"""
        flip_type = random.choice(['hflip', 'vflip'])
        return f'ffmpeg -i "{input_path}" -vf "{flip_type}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def geometric_grid_overlay(input_path: str, output_path: str) -> str:
        """Geometric grid pattern overlay for perception alteration (High Impact)"""
        opacity = random.uniform(0.03, 0.08)
        grid_size = random.randint(20, 40)
        
        # Create grid pattern using drawgrid
        return f'ffmpeg -i "{input_path}" -vf "drawgrid=width={grid_size}:height={grid_size}:thickness=1:color=white@{opacity}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def temporal_frame_disruption(input_path: str, output_path: str) -> str:
        """Temporal frame disruption with randomization for time discontinuity (Medium Impact)"""
    
        # First get video duration to make smart randomization decisions
        try:
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', input_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            video_info = json.loads(result.stdout)
            duration = float(video_info['format']['duration'])
        except:
            # Fallback if probe fails - assume reasonable duration
            duration = 30.0
        
        # Randomization options
        disruption_types = [
            'trim_start_only',
            'trim_end_only', 
            'trim_both_ends',
            'extract_middle_segment',
            'skip_random_section'
        ]
        
        disruption_type = random.choice(disruption_types)
        
        if disruption_type == 'trim_start_only':
            # Remove random amount from beginning
            trim_start = random.uniform(0.5, min(3.0, duration * 0.2))
            keep_duration = duration - trim_start - random.uniform(0.1, 1.0)
            return f'ffmpeg -i "{input_path}" -ss {trim_start} -t {keep_duration} -c:a copy -y "{output_path}"'
        
        elif disruption_type == 'trim_end_only':
            # Remove random amount from end
            trim_end = random.uniform(0.5, min(3.0, duration * 0.2))
            keep_duration = duration - trim_end
            return f'ffmpeg -i "{input_path}" -t {keep_duration} -c:a copy -y "{output_path}"'
        
        elif disruption_type == 'trim_both_ends':
            # Remove from both start and end
            trim_start = random.uniform(0.3, min(2.0, duration * 0.15))
            trim_end = random.uniform(0.3, min(2.0, duration * 0.15))
            keep_duration = duration - trim_start - trim_end
            
            if keep_duration <= 1.0:  # Ensure we keep at least 1 second
                keep_duration = max(1.0, duration * 0.5)
                trim_start = min(trim_start, duration * 0.25)
            
            return f'ffmpeg -i "{input_path}" -ss {trim_start} -t {keep_duration} -c:a copy -y "{output_path}"'
        
        elif disruption_type == 'extract_middle_segment':
            # Extract a random segment from middle
            segment_duration = random.uniform(max(5.0, duration * 0.3), duration * 0.7)
            max_start = duration - segment_duration
            start_time = random.uniform(max_start * 0.2, max_start * 0.8)
            return f'ffmpeg -i "{input_path}" -ss {start_time} -t {segment_duration} -c:a copy -y "{output_path}"'
        
        elif disruption_type == 'skip_random_section':
            # Skip a random section in the middle (creates two segments)
            skip_start = random.uniform(duration * 0.2, duration * 0.5)
            skip_duration = random.uniform(1.0, min(5.0, duration * 0.3))
            skip_end = skip_start + skip_duration
            
            # Create first segment
            temp_segment1 = output_path.replace('.mp4', '_temp1.mp4')
            temp_segment2 = output_path.replace('.mp4', '_temp2.mp4')
            
            # This creates a more complex command that would need additional processing
            # For simplicity, let's just do a single skip by taking everything after the skip
            start_after_skip = skip_end
            remaining_duration = duration - start_after_skip
            
            return f'ffmpeg -i "{input_path}" -ss {start_after_skip} -t {remaining_duration} -c:a copy -y "{output_path}"'
        
        # Default fallback
        trim_start = random.uniform(0.1, 1.0)
        keep_duration = random.uniform(max(5.0, duration * 0.5), duration * 0.9)
        return f'ffmpeg -i "{input_path}" -ss {trim_start} -t {keep_duration} -c:a copy -y "{output_path}"'
    
    @staticmethod
    def ssim_targeted_distortion_combo(input_path: str, output_path: str) -> str:
        """Combination of SSIM-targeted distortions in single pass (Ultra High Impact)"""
        # Multi-layer approach combining multiple high-impact SSIM reducers
        crop_factor = random.uniform(0.88, 0.94)
        rotation = random.uniform(-4, 4)
        blur_sigma = random.uniform(0.8, 1.5)
        noise_level = random.uniform(10, 20)
        hue_shift = random.uniform(-15, 15)
        contrast = random.uniform(0.85, 1.2)
        
        complex_filter = (
            f"crop=iw*{crop_factor}:ih*{crop_factor},"
            f"rotate={rotation}*PI/180:fillcolor=black@0.0,"
            f"gblur=sigma={blur_sigma},"
            f"noise=alls={noise_level}:allf=t,"
            f"hue=h={hue_shift},"
            f"eq=contrast={contrast}"
        )
        
        return f'ffmpeg -i "{input_path}" -vf "{complex_filter}" -c:a copy -y "{output_path}"'
    
    # 🎯 COMPREHENSIVE SSIM REDUCTION STRATEGIES - HIGH IMPACT TRANSFORMATIONS
    # Implementation of the complete SSIM reduction strategy table
    
    @staticmethod
    def high_impact_crop_zoom(input_path: str, output_path: str) -> str:
        """Crop + Zoom - Reduces structural similarity (High Impact)"""
        crop_factor = random.uniform(0.85, 0.95)  # Crop 5-15%
        return f'ffmpeg -i "{input_path}" -vf "crop=iw*{crop_factor}:ih*{crop_factor}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def high_impact_rotation(input_path: str, output_path: str) -> str:
        """Random Rotation - Distorts structure (High Impact)"""
        angle = random.uniform(-8, 8)  # ±8° rotation for stronger effect
        return f'ffmpeg -i "{input_path}" -vf "rotate=PI/180*{angle}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def high_impact_gaussian_blur(input_path: str, output_path: str) -> str:
        """Gaussian Blur - Blurs detail & texture (High Impact)"""
        sigma = random.uniform(1.0, 2.0)  # Strong blur
        return f'ffmpeg -i "{input_path}" -vf "gblur=sigma={sigma}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def high_impact_color_shift_hue(input_path: str, output_path: str) -> str:
        """Color Shift (Hue) - Changes color info (High Impact)"""
        saturation = random.uniform(0.5, 1.5)  # Dramatic saturation change
        hue_degrees = random.uniform(-40, 40)   # Significant hue shift
        return f'ffmpeg -i "{input_path}" -vf "hue=s={saturation}:h={hue_degrees}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def high_impact_add_noise(input_path: str, output_path: str) -> str:
        """Add Noise - Disrupts patterns (High Impact)"""
        noise_strength = random.uniform(15, 25)  # Strong noise
        # Use 't' for temporal noise or remove allf parameter
        return f'ffmpeg -i "{input_path}" -vf "noise=alls={noise_strength}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def high_impact_contrast_brightness(input_path: str, output_path: str) -> str:
        """Contrast/Brightness Shift - Alters luminance (High Impact)"""
        contrast = random.uniform(0.7, 1.4)      # Strong contrast change
        brightness = random.uniform(-0.1, 0.1)   # Brightness adjustment
        return f'ffmpeg -i "{input_path}" -vf "eq=contrast={contrast}:brightness={brightness}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def high_impact_flip_transform(input_path: str, output_path: str) -> str:
        """Flip (Horizontal/Vertical) - Reverses structure (High Impact)"""
        flip_type = random.choice(['hflip', 'vflip'])
        return f'ffmpeg -i "{input_path}" -vf "{flip_type}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def high_impact_overlay_texture_pattern(input_path: str, output_path: str) -> str:
        """Overlay Texture/Pattern - Alters perception (High Impact)"""
        # Grid overlay
        grid_size = random.randint(15, 35)
        opacity = random.uniform(0.05, 0.12)
        thickness = random.randint(1, 2)
        color = random.choice(['white', 'black', 'gray'])
        
        return f'ffmpeg -i "{input_path}" -vf "drawgrid=width={grid_size}:height={grid_size}:thickness={thickness}:color={color}@{opacity}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def medium_impact_trim_start_end(input_path: str, output_path: str) -> str:
        """Trim Start/End - Slight time variation (Medium Impact)"""
        trim_start = random.uniform(0.5, 2.0)   # Trim start seconds
        trim_duration = random.uniform(25, 30)  # Keep 25-30 seconds
        return f'ffmpeg -i "{input_path}" -ss {trim_start} -t {trim_duration} -c:a copy -y "{output_path}"'
    
    @staticmethod
    def medium_impact_insert_black_frame(input_path: str, output_path: str) -> str:
        """Insert Black Frame - Time discontinuity (Medium Impact)"""
        # Insert black frame at random position
        position = random.uniform(5, 15)  # Insert between 5-15 seconds
        duration = random.uniform(0.1, 0.3)  # Black frame duration
        
        return f'ffmpeg -i "{input_path}" -vf "drawbox=enable=\'between(t,{position},{position + duration})\':x=0:y=0:w=iw:h=ih:color=black:t=fill" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def advanced_ssim_reduction_pipeline(input_path: str, output_path: str) -> str:
        """Advanced SSIM Reduction Pipeline - Combines multiple high-impact transformations"""
        # Combine 4-5 high-impact transformations in a single pass
        crop_factor = random.uniform(0.88, 0.95)
        rotation = random.uniform(-6, 6)
        blur_sigma = random.uniform(1.0, 1.8)
        noise_level = random.uniform(12, 22)
        hue_shift = random.uniform(-25, 25)
        saturation = random.uniform(0.6, 1.4)
        contrast = random.uniform(0.8, 1.3)
        brightness = random.uniform(-0.08, 0.08)
        
        # Grid overlay parameters
        grid_size = random.randint(20, 35)
        grid_opacity = random.uniform(0.04, 0.09)
        
        complex_filter = (
            f"crop=iw*{crop_factor}:ih*{crop_factor},"
            f"rotate={rotation}*PI/180:fillcolor=black@0.0,"
            f"gblur=sigma={blur_sigma},"
            f"noise=alls={noise_level}:allf=t,"
            f"hue=h={hue_shift}:s={saturation},"
            f"eq=contrast={contrast}:brightness={brightness},"
            f"drawgrid=width={grid_size}:height={grid_size}:thickness=1:color=white@{grid_opacity}"
        )
        
        return f'ffmpeg -i "{input_path}" -vf "{complex_filter}" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def extreme_ssim_destroyer(input_path: str, output_path: str) -> str:
        """Extreme SSIM Destroyer - Maximum structural disruption for challenging content"""
        # Ultra-aggressive combination targeting SSIM < 0.25
        crop_factor = random.uniform(0.82, 0.90)   # Heavy crop
        rotation = random.uniform(-10, 10)         # Extreme rotation
        blur_sigma = random.uniform(1.5, 2.5)     # Very strong blur
        noise_level = random.uniform(20, 35)      # Heavy noise
        hue_shift = random.uniform(-45, 45)       # Extreme hue shift
        saturation = random.uniform(0.4, 1.6)     # Dramatic saturation
        contrast = random.uniform(0.6, 1.5)       # Strong contrast change
        
        # Random flip decision
        flip_filter = random.choice(['', 'hflip,', 'vflip,'])
        
        complex_filter = (
            f"{flip_filter}"
            f"crop=iw*{crop_factor}:ih*{crop_factor},"
            f"rotate={rotation}*PI/180:fillcolor=black@0.0,"
            f"gblur=sigma={blur_sigma},"
            f"noise=alls={noise_level}:allf=t,"
            f"hue=h={hue_shift}:s={saturation},"
            f"eq=contrast={contrast}"
        ).lstrip(',')
        
        return f'ffmpeg -i "{input_path}" -vf "{complex_filter}" -c:a copy -y "{output_path}"'
    
    # 🎯 COMPREHENSIVE SSIM REDUCTION STRATEGY PIPELINE
    # Implementation of the complete strategy table for maximum SSIM reduction
    
    @staticmethod
    def apply_comprehensive_ssim_strategy(input_path: str, output_path: str, strategy_level: str = "high") -> str:
        """
        Apply the comprehensive SSIM reduction strategy from the strategy table.
        
        Args:
            strategy_level: "high", "medium", or "extreme"
                - high: 3-4 high-impact transformations (target SSIM < 0.30)
                - medium: 2-3 high-impact + 1-2 medium-impact (target SSIM < 0.35)
                - extreme: 5-6 high-impact transformations (target SSIM < 0.25)
        """
        import tempfile
        
        # High-impact transformations (High Impact Level ✅)
        high_impact_options = [
            'high_impact_crop_zoom',
            'high_impact_rotation', 
            'high_impact_gaussian_blur',
            'high_impact_color_shift_hue',
            'high_impact_add_noise',
            'high_impact_contrast_brightness',
            'high_impact_flip_transform',
            'high_impact_overlay_texture_pattern'
        ]
        
        # Medium-impact transformations (Medium Impact Level ⚠️)
        medium_impact_options = [
            'medium_impact_trim_start_end',
            'medium_impact_insert_black_frame'
        ]
        
        applied_transformations = []
        current_input = input_path
        temp_files = []
        
        try:
            if strategy_level == "extreme":
                # Extreme: 5-6 high-impact transformations
                selected_high = random.sample(high_impact_options, k=random.randint(5, 6))
                selected_medium = []
                target_ssim = "< 0.25"
            elif strategy_level == "medium":
                # Medium: 2-3 high-impact + 1-2 medium-impact
                selected_high = random.sample(high_impact_options, k=random.randint(2, 3))
                selected_medium = random.sample(medium_impact_options, k=random.randint(1, 2))
                target_ssim = "< 0.35"
            else:  # high
                # High: 3-4 high-impact transformations
                selected_high = random.sample(high_impact_options, k=random.randint(3, 4))
                selected_medium = []
                target_ssim = "< 0.30"
            
            all_selected = selected_high + selected_medium
            logging.info(f'🎯 Applying COMPREHENSIVE SSIM STRATEGY ({strategy_level.upper()}): {len(all_selected)} transformations, target SSIM {target_ssim}')
            
            for transform_name in all_selected:
                temp_output = tempfile.mktemp(suffix='.mp4', prefix='ssim_')
                temp_files.append(temp_output)
                
                # Apply transformation
                transform_method = getattr(FFmpegTransformationService, transform_name)
                cmd = transform_method(current_input, temp_output)
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    applied_transformations.append(transform_name)
                    current_input = temp_output
                    logging.info(f'   ✅ {transform_name}')
                else:
                    logging.warning(f'   ❌ {transform_name} failed: {result.stderr}')
            
            # Copy final result to output
            if current_input != input_path:
                shutil.copy2(current_input, output_path)
                logging.info(f'🎯 COMPREHENSIVE SSIM STRATEGY COMPLETE: {len(applied_transformations)}/{len(all_selected)} transformations applied')
            
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
        
        return f"Applied {len(applied_transformations)} SSIM reduction transformations: {', '.join(applied_transformations)}"
    
    # ✅ STRATEGY: 7-LAYER TRANSFORMATION PIPELINE
    # Each layer targets specific similarity metrics for maximum entropy reduction
    
    @staticmethod
    def apply_seven_layer_pipeline(input_path: str, output_path: str, variant_id: int = None) -> list:
        """
        Apply 7-layer transformation pipeline for maximum similarity reduction.
        
        Strategy:
        - Layer 1: ORB Similarity Disruption (1-3 transforms)
        - Layer 2: Audio Fingerprint Obfuscation (2-3 transforms) 
        - Layer 3: SSIM & Structural Shift (1-2 transforms)
        - Layer 4: PHash Distance Increase (1 transform)
        - Layer 5: Metadata Scrambling (2-3 transforms)
        - Layer 6: Temporal Flow Disruption (1-2 transforms)
        - Layer 7: Semantic / Overlay Distortion (1-2 transforms)
        
        Returns:
            List of applied transformations with their commands
        """
        import tempfile
        import os
        
        applied_transforms = []
        current_input = input_path
        temp_files = []
        
        # Use variant_id for deterministic randomness if provided
        if variant_id:
            random.seed(variant_id)
        
        try:
            # 🔴 LAYER 1: ORB Similarity Disruption (High Weight) - Apply 1-3
            layer1_transforms = [
                'micro_perspective_warp',
                'frame_jittering', 
                'fine_noise_overlay',
                'dynamic_rotation_enhanced',
                'entropy_boost_enhanced',
                'zoom_jitter_motion_enhanced',
                'color_histogram_shift_enhanced',
                'phash_disruption_enhanced'
            ]
            selected_layer1 = random.sample(layer1_transforms, k=random.randint(1, 3))
            
            for transform in selected_layer1:
                temp_output = tempfile.mktemp(suffix='.mp4', prefix='layer1_')
                temp_files.append(temp_output)
                
                if transform == 'micro_perspective_warp':
                    cmd = FFmpegTransformationService.micro_perspective_warp_enhanced(current_input, temp_output)
                elif transform == 'frame_jittering':
                    cmd = FFmpegTransformationService.frame_jittering_enhanced(current_input, temp_output)
                elif transform == 'fine_noise_overlay':
                    cmd = FFmpegTransformationService.fine_noise_overlay(current_input, temp_output)
                elif transform == 'dynamic_rotation_enhanced':
                    cmd = FFmpegTransformationService.dynamic_rotation_enhanced(current_input, temp_output)
                elif transform == 'entropy_boost_enhanced':
                    cmd = FFmpegTransformationService.entropy_boost_enhanced(current_input, temp_output)
                elif transform == 'zoom_jitter_motion_enhanced':
                    cmd = FFmpegTransformationService.zoom_jitter_motion_enhanced(current_input, temp_output)
                elif transform == 'color_histogram_shift_enhanced':
                    cmd = FFmpegTransformationService.color_histogram_shift_enhanced(current_input, temp_output)
                elif transform == 'phash_disruption_enhanced':
                    cmd = FFmpegTransformationService.phash_disruption_enhanced(current_input, temp_output)
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    applied_transforms.append(f"Layer 1 - {transform}")
                    current_input = temp_output
                else:
                    logging.warning(f"Layer 1 transform {transform} failed: {result.stderr}")
            
            # 🟠 LAYER 2: Audio Fingerprint Obfuscation (High Weight) - Apply 2-3
            layer2_transforms = [
                'spectral_fingerprint_disruption',
                'pitch_shift_transform_enhanced',
                'frequency_band_shifting',
                'add_ambient_noise_enhanced', 
                'voice_pattern_disruption',
                'audio_reverse_segments'
            ]
            selected_layer2 = random.sample(layer2_transforms, k=random.randint(2, 3))
            
            for transform in selected_layer2:
                temp_output = tempfile.mktemp(suffix='.mp4', prefix='layer2_')
                temp_files.append(temp_output)
                
                if transform == 'spectral_fingerprint_disruption':
                    cmd = FFmpegTransformationService.spectral_fingerprint_disruption(current_input, temp_output)
                elif transform == 'pitch_shift_transform_enhanced':
                    cmd = FFmpegTransformationService.pitch_shift_transform_enhanced(current_input, temp_output)
                elif transform == 'frequency_band_shifting':
                    cmd = FFmpegTransformationService.frequency_band_shifting(current_input, temp_output)
                elif transform == 'add_ambient_noise_enhanced':
                    cmd = FFmpegTransformationService.add_ambient_noise_enhanced(current_input, temp_output)
                elif transform == 'voice_pattern_disruption':
                    cmd = FFmpegTransformationService.voice_pattern_disruption(current_input, temp_output)
                elif transform == 'audio_reverse_segments':
                    cmd = FFmpegTransformationService.audio_reverse_segments(current_input, temp_output)
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    applied_transforms.append(f"Layer 2 - {transform}")
                    current_input = temp_output
                else:
                    logging.warning(f"Layer 2 transform {transform} failed: {result.stderr}")
            
            # 🟡 LAYER 3: SSIM & Structural Shift (High Impact SSIM Reduction) - Apply 2-3
            layer3_transforms = [
                'high_impact_crop_zoom',
                'high_impact_rotation', 
                'high_impact_gaussian_blur',
                'high_impact_color_shift_hue',
                'high_impact_add_noise',
                'high_impact_contrast_brightness',
                'high_impact_flip_transform',
                'high_impact_overlay_texture_pattern',
                'advanced_ssim_reduction_pipeline',
                'extreme_ssim_destroyer',
                'ssim_reduction_controlled',
                'frame_micro_adjustments',
                'random_motion_blur_effects',
                'dynamic_timestamp_overlay',
                'film_grain_simulation',
                'random_geometric_warp'
            ]
            selected_layer3 = random.sample(layer3_transforms, k=random.randint(2, 3))  # Increased to 2-3 for stronger SSIM reduction
            
            for transform in selected_layer3:
                temp_output = tempfile.mktemp(suffix='.mp4', prefix='layer3_')
                temp_files.append(temp_output)
                
                # New high-impact SSIM reduction transformations
                if transform == 'high_impact_crop_zoom':
                    cmd = FFmpegTransformationService.high_impact_crop_zoom(current_input, temp_output)
                elif transform == 'high_impact_rotation':
                    cmd = FFmpegTransformationService.high_impact_rotation(current_input, temp_output)
                elif transform == 'high_impact_gaussian_blur':
                    cmd = FFmpegTransformationService.high_impact_gaussian_blur(current_input, temp_output)
                elif transform == 'high_impact_color_shift_hue':
                    cmd = FFmpegTransformationService.high_impact_color_shift_hue(current_input, temp_output)
                elif transform == 'high_impact_add_noise':
                    cmd = FFmpegTransformationService.high_impact_add_noise(current_input, temp_output)
                elif transform == 'high_impact_contrast_brightness':
                    cmd = FFmpegTransformationService.high_impact_contrast_brightness(current_input, temp_output)
                elif transform == 'high_impact_flip_transform':
                    cmd = FFmpegTransformationService.high_impact_flip_transform(current_input, temp_output)
                elif transform == 'high_impact_overlay_texture_pattern':
                    cmd = FFmpegTransformationService.high_impact_overlay_texture_pattern(current_input, temp_output)
                elif transform == 'advanced_ssim_reduction_pipeline':
                    cmd = FFmpegTransformationService.advanced_ssim_reduction_pipeline(current_input, temp_output)
                elif transform == 'extreme_ssim_destroyer':
                    cmd = FFmpegTransformationService.extreme_ssim_destroyer(current_input, temp_output)
                # Existing transformations
                elif transform == 'ssim_reduction_controlled':
                    cmd = FFmpegTransformationService.ssim_reduction_controlled(current_input, temp_output)
                elif transform == 'frame_micro_adjustments':
                    cmd = FFmpegTransformationService.frame_micro_adjustments(current_input, temp_output)
                elif transform == 'random_motion_blur_effects':
                    cmd = FFmpegTransformationService.random_motion_blur_effects(current_input, temp_output)
                elif transform == 'dynamic_timestamp_overlay':
                    cmd = FFmpegTransformationService.dynamic_timestamp_overlay(current_input, temp_output)
                elif transform == 'film_grain_simulation':
                    cmd = FFmpegTransformationService.film_grain_simulation(current_input, temp_output)
                elif transform == 'random_geometric_warp':
                    cmd = FFmpegTransformationService.random_geometric_warp(current_input, temp_output)
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    applied_transforms.append(f"Layer 3 - {transform}")
                    current_input = temp_output
                else:
                    logging.warning(f"Layer 3 transform {transform} failed: {result.stderr}")
            
            # 🟡 LAYER 4: PHash Distance Increase (Medium Weight) - Apply 1 (adjusted for 3-6 visual range)
            layer4_transforms = [
                'extreme_phash_disruption',
                'color_channel_swapping',
                'perspective_distortion',
                'texture_blend_overlay',
                'clip_embedding_shift_enhanced'
            ]
            selected_layer4 = random.sample(layer4_transforms, k=1)
            
            for transform in selected_layer4:
                temp_output = tempfile.mktemp(suffix='.mp4', prefix='layer4_')
                temp_files.append(temp_output)
                
                if transform == 'extreme_phash_disruption':
                    cmd = FFmpegTransformationService.extreme_phash_disruption(current_input, temp_output)
                elif transform == 'color_channel_swapping':
                    cmd = FFmpegTransformationService.color_channel_swapping(current_input, temp_output)
                elif transform == 'perspective_distortion':
                    cmd = FFmpegTransformationService.perspective_distortion(current_input, temp_output)
                elif transform == 'texture_blend_overlay':
                    cmd = FFmpegTransformationService.texture_blend_overlay(current_input, temp_output, method="scale")
                elif transform == 'clip_embedding_shift_enhanced':
                    cmd = FFmpegTransformationService.clip_embedding_shift_enhanced(current_input, temp_output)
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    applied_transforms.append(f"Layer 4 - {transform}")
                    current_input = temp_output
                else:
                    logging.warning(f"Layer 4 transform {transform} failed: {result.stderr}")
            
            # 🟢 LAYER 5: Metadata Scrambling (Low Weight) - Apply 2-3 (INCREASED)
            layer5_transforms = [
                'metadata_strip_randomize',
                'uuid_injection_system',
                'advanced_metadata_spoofing',
                'ultra_metadata_randomization',
                'codec_metadata_randomization',
                'timestamp_metadata_fuzzing',
                'uuid_metadata_injection',
                'creation_time_fuzzing'
            ]
            selected_layer5 = random.sample(layer5_transforms, k=random.randint(2, 3))  # INCREASED from 1-2 to 2-3
            
            for transform in selected_layer5:
                temp_output = tempfile.mktemp(suffix='.mp4', prefix='layer5_')
                temp_files.append(temp_output)
                
                if transform == 'metadata_strip_randomize':
                    cmd = FFmpegTransformationService.metadata_strip_randomize(current_input, temp_output)
                elif transform == 'uuid_injection_system':
                    cmd = FFmpegTransformationService.uuid_injection_system(current_input, temp_output)
                elif transform == 'advanced_metadata_spoofing':
                    cmd = FFmpegTransformationService.advanced_metadata_spoofing(current_input, temp_output)
                elif transform == 'ultra_metadata_randomization':
                    cmd = FFmpegTransformationService.ultra_metadata_randomization(current_input, temp_output)
                elif transform == 'codec_metadata_randomization':
                    cmd = FFmpegTransformationService.codec_metadata_randomization(current_input, temp_output)
                elif transform == 'timestamp_metadata_fuzzing':
                    cmd = FFmpegTransformationService.timestamp_metadata_fuzzing(current_input, temp_output)
                elif transform == 'uuid_metadata_injection':
                    cmd = FFmpegTransformationService.uuid_metadata_injection(current_input, temp_output)
                elif transform == 'creation_time_fuzzing':
                    cmd = FFmpegTransformationService.creation_time_fuzzing(current_input, temp_output)
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    applied_transforms.append(f"Layer 5 - {transform}")
                    current_input = temp_output
                else:
                    logging.warning(f"Layer 5 transform {transform} failed: {result.stderr}")
            
            # 🌀 LAYER 6: Temporal Flow Disruption (Including Medium-Impact SSIM) - Apply 1-2
            layer6_transforms = [
                'medium_impact_trim_start_end',
                'medium_impact_insert_black_frame',
                'temporal_shift_advanced',
                'frame_trimming_dropout',
                'black_screen_random',
                'frame_reordering_segments',
                'random_cut_jitter_effects'
            ]
            selected_layer6 = random.sample(layer6_transforms, k=random.randint(1, 2))
            
            for transform in selected_layer6:
                temp_output = tempfile.mktemp(suffix='.mp4', prefix='layer6_')
                temp_files.append(temp_output)
                
                # Medium-impact SSIM transformations
                if transform == 'medium_impact_trim_start_end':
                    cmd = FFmpegTransformationService.medium_impact_trim_start_end(current_input, temp_output)
                elif transform == 'medium_impact_insert_black_frame':
                    cmd = FFmpegTransformationService.medium_impact_insert_black_frame(current_input, temp_output)
                # Existing transformations
                elif transform == 'temporal_shift_advanced':
                    cmd = FFmpegTransformationService.temporal_shift_advanced(current_input, temp_output)
                elif transform == 'frame_trimming_dropout':
                    cmd = FFmpegTransformationService.frame_trimming_dropout(current_input, temp_output)
                elif transform == 'black_screen_random':
                    cmd = FFmpegTransformationService.black_screen_random(current_input, temp_output)
                elif transform == 'frame_reordering_segments':
                    cmd = FFmpegTransformationService.frame_reordering_segments(current_input, temp_output)
                elif transform == 'random_cut_jitter_effects':
                    cmd = FFmpegTransformationService.random_cut_jitter_effects(current_input, temp_output)
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    applied_transforms.append(f"Layer 6 - {transform}")
                    current_input = temp_output
                else:
                    logging.warning(f"Layer 6 transform {transform} failed: {result.stderr}")
            
            # 🧠 LAYER 7: Semantic / Overlay Distortion - Apply 1-2
            layer7_transforms = [
                'animated_text_corner_enhanced',
                'low_opacity_watermark_enhanced',
                'text_presence_variation',
                'audio_video_sync_offset'
            ]
            selected_layer7 = random.sample(layer7_transforms, k=random.randint(1, 2))
            
            for transform in selected_layer7:
                temp_output = tempfile.mktemp(suffix='.mp4', prefix='layer7_')
                temp_files.append(temp_output)
                
                if transform == 'animated_text_corner_enhanced':
                    cmd = FFmpegTransformationService.animated_text_corner_enhanced(current_input, temp_output)
                elif transform == 'low_opacity_watermark_enhanced':
                    cmd = FFmpegTransformationService.low_opacity_watermark_enhanced(current_input, temp_output)
                elif transform == 'text_presence_variation':
                    cmd = FFmpegTransformationService.text_presence_variation(current_input, temp_output)
                elif transform == 'audio_video_sync_offset':
                    cmd = FFmpegTransformationService.audio_video_sync_offset(current_input, temp_output)
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    applied_transforms.append(f"Layer 7 - {transform}")
                    current_input = temp_output
                else:
                    logging.warning(f"Layer 7 transform {transform} failed: {result.stderr}")
            
            # Copy final result to output
            if current_input != input_path:
                import shutil
                shutil.copy2(current_input, output_path)
                logging.info(f"✅ 7-Layer pipeline complete: {len(applied_transforms)} transforms applied")
            
            return applied_transforms
            
        except Exception as e:
            logging.error(f"❌ 7-Layer pipeline failed: {e}")
            return []
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file) and temp_file != output_path:
                        os.remove(temp_file)
                except:
                    pass
    
    @staticmethod
    def validate_atempo_value(speed_factor: float) -> str:
        """
        Validate and create atempo filter chain for speed changes.
        atempo filter only accepts values between 0.5 and 2.0
        """
        if speed_factor < 0.5:
            # For very slow speeds, chain multiple atempo filters
            return "atempo=0.5,atempo=" + str(speed_factor / 0.5)
        elif speed_factor > 2.0:
            # For very fast speeds, chain multiple atempo filters  
            return "atempo=2.0,atempo=" + str(speed_factor / 2.0)
        else:
            return f"atempo={speed_factor}"

    @staticmethod
    def validate_number(value: float, min_val: float, max_val: float, fallback: float) -> float:
        """Helper function to ensure numeric values are valid and within bounds"""
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return fallback
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def get_random_in_range(min_val: float, max_val: float) -> float:
        """Generate random values within bounds"""
        return min_val + random.random() * (max_val - min_val)
    
    @staticmethod
    def get_random_element(array: List[Any]) -> Any:
        """Get random element from array"""
        return array[random.randint(0, len(array) - 1)]
    
    @staticmethod
    def get_random_opacity(min_val: float = 0.05, max_val: float = 0.2) -> float:
        """Generate random opacity value (REDUCED from 0.1-0.4)"""
        return FFmpegTransformationService.get_random_in_range(min_val, max_val)
    
    @staticmethod
    def get_random_position() -> str:
        """Generate random position for text/watermarks"""
        positions = ['x=10:y=10', 'x=w-tw-10:y=10', 'x=10:y=h-th-10', 'x=w-tw-10:y=h-th-10']
        return FFmpegTransformationService.get_random_element(positions)
    
    @staticmethod
    def get_random_color() -> str:
        """Generate random color"""
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'gray', '#333333', '#666666']
        return FFmpegTransformationService.get_random_element(colors)
    
    @staticmethod
    def get_random_text(category: str) -> str:
        """Generate random text content"""
        text_map = {
            'overlay': ['SAMPLE', 'PREVIEW', 'DEMO', 'TEST', '© 2025'],
            'brand': ['SAMPLE', 'PREVIEW', 'DEMO', 'TEST', '© 2025', 'ORIGINAL', 'EXCLUSIVE'],
            'intro': ['Get Ready...', 'Watch This!', 'Amazing Content', 'Exclusive Video', 'Premium Quality'],
            'outro': ['Thanks for Watching!', 'Subscribe Now!', 'Like & Share', 'More Coming Soon', 'Stay Tuned!'],
            'subtitle': ['Incredible footage!', 'Must see this', 'Amazing quality', 'Exclusive content', 'Premium video']
        }
        return FFmpegTransformationService.get_random_element(text_map.get(category, ['SAMPLE']))
    
    @staticmethod
    async def get_video_info(video_path: str) -> Dict[str, Any]:
        """Get video information using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"ffprobe failed: {stderr.decode()}")
            
            info = json.loads(stdout.decode())
            
            # Parse video stream
            video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
            
            return {
                'duration': FFmpegTransformationService.validate_number(
                    float(info['format'].get('duration', 10)), 0.1, 3600, 10
                ),
                'hasAudio': audio_stream is not None,
                'width': FFmpegTransformationService.validate_number(
                    int(video_stream.get('width', 1280)) if video_stream else 1280, 100, 7680, 1280
                ),
                'height': FFmpegTransformationService.validate_number(
                    int(video_stream.get('height', 720)) if video_stream else 720, 100, 4320, 720
                )
            }
            
        except Exception as error:
            logging.warning(f'Could not get video info, using defaults: {error}')
            return {'duration': 10, 'hasAudio': True, 'width': 1280, 'height': 720}
    
    @staticmethod
    def get_video_info_sync(video_path: str) -> Dict[str, Any]:
        """Synchronous version of get_video_info for use in Celery workers"""
        import subprocess
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode != 0:
                raise Exception(f"ffprobe failed: {process.stderr}")
            
            info = json.loads(process.stdout)
            
            # Parse video stream
            video_stream = None
            audio_stream = None
            
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                elif stream.get('codec_type') == 'audio':
                    audio_stream = stream
            
            return {
                'duration': FFmpegTransformationService.validate_number(
                    float(info.get('format', {}).get('duration', 10)), 1, 7200, 10
                ),
                'hasAudio': audio_stream is not None,
                'width': FFmpegTransformationService.validate_number(
                    int(video_stream.get('width', 1280)) if video_stream else 1280, 100, 7680, 1280
                ),
                'height': FFmpegTransformationService.validate_number(
                    int(video_stream.get('height', 720)) if video_stream else 720, 100, 4320, 720
                )
            }
            
        except Exception as error:
            logging.warning(f'Could not get video info, using defaults: {error}')
            return {'duration': 10, 'hasAudio': True, 'width': 1280, 'height': 720}
    
    @staticmethod
    async def validate_video_file(video_path: str) -> bool:
        """Validate video file exists and is readable"""
        try:
            if not os.path.exists(video_path):
                logging.error(f'Video file does not exist: {video_path}')
                return False
            
            if os.path.getsize(video_path) == 0:
                logging.error(f'Video file is empty: {video_path}')
                return False
            
            # Test with ffprobe
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-count_packets', '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0', video_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            return process.returncode == 0
            
        except Exception as error:
            logging.error(f'Error validating video file: {video_path} - {error}')
            return False
    
    # PUNCHLINE TRANSFORMATION FUNCTIONS - SEMANTIC CONTENT ENHANCEMENT
    @staticmethod
    @staticmethod
    def punchline_transformation_light(input_path: str, output_path: str) -> str:
        """Apply light punchline enhancement with minimal visual disruption"""
        try:
            # Since we're in a Celery worker (which already has an event loop),
            # we can't use asyncio.run(). Just use a simple copy as fallback.
            logging.info("Punchline transformation light requested - using copy as fallback in sync context")
            return f'ffmpeg -i "{input_path}" -c copy -y "{output_path}"'
                
        except Exception as e:
            logging.warning(f"Punchline transformation failed, using fallback: {e}")
            return f'ffmpeg -i "{input_path}" -c copy -y "{output_path}"'
    
    @staticmethod 
    def punchline_transformation_medium(input_path: str, output_path: str) -> str:
        """Apply medium punchline enhancement with moderate visual effects"""
        try:
            # Since we're in a Celery worker (which already has an event loop),
            # we can't use asyncio.run(). Just use a simple copy as fallback.
            logging.info("Punchline transformation medium requested - using copy as fallback in sync context")
            return f'ffmpeg -i "{input_path}" -c copy -y "{output_path}"'
                
        except Exception as e:
            logging.warning(f"Punchline transformation failed, using fallback: {e}")
            return f'ffmpeg -i "{input_path}" -c copy -y "{output_path}"'
    
    @staticmethod
    def punchline_transformation_heavy(input_path: str, output_path: str) -> str:
        """Apply heavy punchline enhancement with maximum visual effects"""
        try:
            # Since we're in a Celery worker (which already has an event loop),
            # we can't use asyncio.run(). Just use a simple copy as fallback.
            logging.info("Punchline transformation heavy requested - using copy as fallback in sync context")
            return f'ffmpeg -i "{input_path}" -c copy -y "{output_path}"'
                
        except Exception as e:
            logging.warning(f"Punchline transformation failed, using fallback: {e}")
            return f'ffmpeg -i "{input_path}" -c copy -y "{output_path}"'

    @staticmethod
    def get_transformations() -> List[TransformationConfig]:
        """Get comprehensive advanced transformations with ALL 121+ AVAILABLE METHODS - MAXIMUM VARIATION POOL"""
        return [
            # PUNCHLINE TRANSFORMATIONS (3)
            TransformationConfig('punchline_transformation_light', 0.5, FFmpegTransformationService.punchline_transformation_light, 'punchline'),
            TransformationConfig('punchline_transformation_medium', 0.5, FFmpegTransformationService.punchline_transformation_medium, 'punchline'), 
            TransformationConfig('punchline_transformation_heavy', 0.4, FFmpegTransformationService.punchline_transformation_heavy, 'punchline'),
            
            # CORE VISUAL TRANSFORMATIONS (9)
            TransformationConfig('phash_disruption_transform', 0.7, FFmpegTransformationService.phash_disruption_transform, 'visual'),
            TransformationConfig('ssim_reduction_transform', 0.7, FFmpegTransformationService.ssim_reduction_transform, 'visual'),
            TransformationConfig('color_histogram_shift', 0.6, FFmpegTransformationService.color_histogram_shift, 'visual'),
            TransformationConfig('extreme_phash_disruption', 0.6, FFmpegTransformationService.extreme_phash_disruption, 'visual'),
            TransformationConfig('frame_entropy_increase', 0.6, FFmpegTransformationService.frame_entropy_increase, 'visual'),
            TransformationConfig('embedding_similarity_change', 0.6, FFmpegTransformationService.embedding_similarity_change, 'visual'),
            TransformationConfig('aggressive_color_shift', 0.5, FFmpegTransformationService.aggressive_color_shift, 'visual'),
            TransformationConfig('aggressive_geometric_distortion', 0.4, FFmpegTransformationService.aggressive_geometric_distortion, 'visual'),
            TransformationConfig('aggressive_temporal_manipulation', 0.4, FFmpegTransformationService.aggressive_temporal_manipulation, 'temporal'),
            
            # 🎯 HIGH-IMPACT SSIM REDUCTION STRATEGIES (11) - Targeting SSIM < 0.30
            TransformationConfig('enhanced_crop_zoom_ssim', 0.8, FFmpegTransformationService.enhanced_crop_zoom_ssim, 'ssim_reduction'),
            TransformationConfig('aggressive_gaussian_blur_ssim', 0.7, FFmpegTransformationService.aggressive_gaussian_blur_ssim, 'ssim_reduction'),
            TransformationConfig('enhanced_rotation_ssim', 0.7, FFmpegTransformationService.enhanced_rotation_ssim, 'ssim_reduction'),
            TransformationConfig('aggressive_hue_saturation_shift', 0.8, FFmpegTransformationService.aggressive_hue_saturation_shift, 'ssim_reduction'),
            TransformationConfig('pattern_disruption_noise', 0.7, FFmpegTransformationService.pattern_disruption_noise, 'ssim_reduction'),
            TransformationConfig('contrast_brightness_disruption', 0.8, FFmpegTransformationService.contrast_brightness_disruption, 'ssim_reduction'),
            TransformationConfig('strategic_flip_transform', 0.6, FFmpegTransformationService.strategic_flip_transform, 'ssim_reduction'),
            TransformationConfig('geometric_grid_overlay', 0.6, FFmpegTransformationService.geometric_grid_overlay, 'ssim_reduction'),
            TransformationConfig('temporal_frame_disruption', 0.5, FFmpegTransformationService.temporal_frame_disruption, 'ssim_reduction'),
            TransformationConfig('ssim_targeted_distortion_combo', 0.9, FFmpegTransformationService.ssim_targeted_distortion_combo, 'ssim_reduction'),
            
            # 🎯 COMPREHENSIVE SSIM REDUCTION STRATEGIES - IMPLEMENTATION OF STRATEGY TABLE (11) - High Impact
            TransformationConfig('high_impact_crop_zoom', 0.85, FFmpegTransformationService.high_impact_crop_zoom, 'ssim_reduction'),
            TransformationConfig('high_impact_rotation', 0.8, FFmpegTransformationService.high_impact_rotation, 'ssim_reduction'),
            TransformationConfig('high_impact_gaussian_blur', 0.8, FFmpegTransformationService.high_impact_gaussian_blur, 'ssim_reduction'),
            TransformationConfig('high_impact_color_shift_hue', 0.85, FFmpegTransformationService.high_impact_color_shift_hue, 'ssim_reduction'),
            TransformationConfig('high_impact_add_noise', 0.8, FFmpegTransformationService.high_impact_add_noise, 'ssim_reduction'),
            TransformationConfig('high_impact_contrast_brightness', 0.85, FFmpegTransformationService.high_impact_contrast_brightness, 'ssim_reduction'),
            TransformationConfig('high_impact_flip_transform', 0.75, FFmpegTransformationService.high_impact_flip_transform, 'ssim_reduction'),
            TransformationConfig('high_impact_overlay_texture_pattern', 0.8, FFmpegTransformationService.high_impact_overlay_texture_pattern, 'ssim_reduction'),
            TransformationConfig('medium_impact_trim_start_end', 0.6, FFmpegTransformationService.medium_impact_trim_start_end, 'ssim_reduction'),
            TransformationConfig('medium_impact_insert_black_frame', 0.55, FFmpegTransformationService.medium_impact_insert_black_frame, 'ssim_reduction'),
            TransformationConfig('advanced_ssim_reduction_pipeline', 0.9, FFmpegTransformationService.advanced_ssim_reduction_pipeline, 'ssim_reduction'),
            TransformationConfig('extreme_ssim_destroyer', 0.7, FFmpegTransformationService.extreme_ssim_destroyer, 'ssim_reduction'),
            
            # CORE AUDIO TRANSFORMATIONS (10) 
            TransformationConfig('aggressive_audio_manipulation', 0.4, FFmpegTransformationService.aggressive_audio_manipulation, 'audio'),
            TransformationConfig('spectral_fingerprint_disruption', 0.7, FFmpegTransformationService.spectral_fingerprint_disruption, 'audio'),
            TransformationConfig('pitch_shift_transform', 0.6, FFmpegTransformationService.pitch_shift_transform, 'audio'),
            TransformationConfig('tempo_shift_transform', 0.6, FFmpegTransformationService.tempo_shift_transform, 'audio'),
            TransformationConfig('audio_layering_ambient', 0.5, FFmpegTransformationService.audio_layering_ambient, 'audio'),
            TransformationConfig('audio_segment_reorder', 0.5, FFmpegTransformationService.audio_segment_reorder, 'audio'),
            TransformationConfig('audio_simple_processing', 0.6, FFmpegTransformationService.audio_simple_processing, 'audio'),
            TransformationConfig('pitch_shift_transform_enhanced', 0.6, FFmpegTransformationService.pitch_shift_transform_enhanced, 'audio'),
            TransformationConfig('tempo_shift_enhanced', 0.6, FFmpegTransformationService.tempo_shift_enhanced, 'audio'),
            TransformationConfig('add_ambient_noise_enhanced', 0.6, FFmpegTransformationService.add_ambient_noise_enhanced, 'audio'),
            
            # ENHANCED ORB BREAKING TRANSFORMATIONS (10)
            TransformationConfig('micro_perspective_warp_enhanced', 0.8, FFmpegTransformationService.micro_perspective_warp_enhanced, 'orb_breaking'),
            TransformationConfig('frame_jittering_enhanced', 0.8, FFmpegTransformationService.frame_jittering_enhanced, 'orb_breaking'),
            TransformationConfig('dynamic_rotation_enhanced', 0.7, FFmpegTransformationService.dynamic_rotation_enhanced, 'orb_breaking'),
            TransformationConfig('zoom_jitter_motion_enhanced', 0.7, FFmpegTransformationService.zoom_jitter_motion_enhanced, 'orb_breaking'),
            TransformationConfig('color_histogram_shift_enhanced', 0.7, FFmpegTransformationService.color_histogram_shift_enhanced, 'orb_breaking'),
            TransformationConfig('line_sketch_filter_enhanced', 0.6, FFmpegTransformationService.line_sketch_filter_enhanced, 'orb_breaking'),
            TransformationConfig('entropy_boost_enhanced', 0.6, FFmpegTransformationService.entropy_boost_enhanced, 'orb_breaking'),
            TransformationConfig('clip_embedding_shift_enhanced', 0.6, FFmpegTransformationService.clip_embedding_shift_enhanced, 'orb_breaking'),
            TransformationConfig('phash_disruption_enhanced', 0.7, FFmpegTransformationService.phash_disruption_enhanced, 'orb_breaking'),
            TransformationConfig('ssim_reduction_controlled', 0.6, FFmpegTransformationService.ssim_reduction_controlled, 'orb_breaking'),
            
            # METADATA TRANSFORMATIONS (8)
            TransformationConfig('ultra_metadata_randomization', 0.8, FFmpegTransformationService.ultra_metadata_randomization, 'metadata'),
            TransformationConfig('advanced_metadata_spoofing', 0.7, FFmpegTransformationService.advanced_metadata_spoofing, 'metadata'),
            TransformationConfig('codec_metadata_randomization', 0.6, FFmpegTransformationService.codec_metadata_randomization, 'metadata'),
            TransformationConfig('timestamp_metadata_fuzzing', 0.6, FFmpegTransformationService.timestamp_metadata_fuzzing, 'metadata'),
            TransformationConfig('uuid_metadata_injection', 0.6, FFmpegTransformationService.uuid_metadata_injection, 'metadata'),
            TransformationConfig('metadata_strip_randomize', 0.6, FFmpegTransformationService.metadata_strip_randomize, 'metadata'),
            TransformationConfig('creation_time_fuzzing', 0.6, FFmpegTransformationService.creation_time_fuzzing, 'metadata'),
            TransformationConfig('uuid_injection_system', 0.6, FFmpegTransformationService.uuid_injection_system, 'metadata'),
            
            # OVERLAY AND WATERMARK TRANSFORMATIONS (8)
            TransformationConfig('animated_text_corner_enhanced', 0.7, FFmpegTransformationService.animated_text_corner_enhanced, 'overlay'),
            TransformationConfig('low_opacity_watermark_enhanced', 0.7, FFmpegTransformationService.low_opacity_watermark_enhanced, 'overlay'),
            TransformationConfig('overlay_watermark_dynamic', 0.6, FFmpegTransformationService.overlay_watermark_dynamic, 'overlay'),
            TransformationConfig('moving_watermark_system', 0.6, FFmpegTransformationService.moving_watermark_system, 'overlay'),
            TransformationConfig('animated_text_corner', 0.6, FFmpegTransformationService.animated_text_corner, 'overlay'),
            TransformationConfig('texture_blend_overlay', 0.5, FFmpegTransformationService.texture_blend_overlay, 'overlay'),
            TransformationConfig('particle_overlay_system', 0.5, FFmpegTransformationService.particle_overlay_system, 'overlay'),
            TransformationConfig('dynamic_timestamp_overlay', 0.6, FFmpegTransformationService.dynamic_timestamp_overlay, 'overlay'),
            
            # CLASSIC ORB BREAKING TRANSFORMATIONS (9)
            TransformationConfig('micro_perspective_warp', 0.7, FFmpegTransformationService.micro_perspective_warp, 'orb_breaking'),
            TransformationConfig('frame_jittering', 0.7, FFmpegTransformationService.frame_jittering, 'orb_breaking'),
            TransformationConfig('fine_noise_overlay', 0.6, FFmpegTransformationService.fine_noise_overlay, 'orb_breaking'),
            TransformationConfig('subtle_logo_texture_overlay', 0.6, FFmpegTransformationService.subtle_logo_texture_overlay, 'orb_breaking'),
            TransformationConfig('dynamic_zoom_oscillation', 0.6, FFmpegTransformationService.dynamic_zoom_oscillation, 'orb_breaking'),
            TransformationConfig('dynamic_rotation_subtle', 0.6, FFmpegTransformationService.dynamic_rotation_subtle, 'orb_breaking'),
            TransformationConfig('slight_color_variation', 0.6, FFmpegTransformationService.slight_color_variation, 'orb_breaking'),
            TransformationConfig('line_sketch_filter_light', 0.6, FFmpegTransformationService.line_sketch_filter_light, 'orb_breaking'),
            TransformationConfig('randomized_transform_sets', 0.5, FFmpegTransformationService.randomized_transform_sets, 'orb_breaking'),
            
            # VIDEO EFFECTS AND FILTERS (12)
            TransformationConfig('video_length_trim', 0.4, FFmpegTransformationService.video_length_trim, 'temporal'),
            TransformationConfig('frame_rate_change', 0.5, FFmpegTransformationService.frame_rate_change, 'temporal'),
            TransformationConfig('black_frames_transitions', 0.5, FFmpegTransformationService.black_frames_transitions, 'temporal'),
            TransformationConfig('title_caption_randomize', 0.6, FFmpegTransformationService.title_caption_randomize, 'enhancement'),
            TransformationConfig('clip_embedding_shift', 0.6, FFmpegTransformationService.clip_embedding_shift, 'enhancement'),
            TransformationConfig('text_presence_variation', 0.6, FFmpegTransformationService.text_presence_variation, 'enhancement'),
            TransformationConfig('micro_rotation_with_crop', 0.6, FFmpegTransformationService.micro_rotation_with_crop, 'visual'),
            TransformationConfig('advanced_lut_filter', 0.6, FFmpegTransformationService.advanced_lut_filter, 'visual'),
            TransformationConfig('vignette_with_blur', 0.6, FFmpegTransformationService.vignette_with_blur, 'visual'),
            TransformationConfig('temporal_shift_advanced', 0.5, FFmpegTransformationService.temporal_shift_advanced, 'temporal'),
            TransformationConfig('black_screen_random', 0.4, FFmpegTransformationService.black_screen_random, 'temporal'),
            TransformationConfig('zoom_jitter_motion', 0.6, FFmpegTransformationService.zoom_jitter_motion, 'visual'),
            
            # ADVANCED AUDIO PROCESSING (15)
            TransformationConfig('frequency_band_shifting', 0.6, FFmpegTransformationService.frequency_band_shifting, 'audio'),
            TransformationConfig('multi_band_eq_randomization', 0.6, FFmpegTransformationService.multi_band_eq_randomization, 'audio'),
            TransformationConfig('harmonic_distortion_subtle', 0.5, FFmpegTransformationService.harmonic_distortion_subtle, 'audio'),
            TransformationConfig('frequency_domain_shift', 0.6, FFmpegTransformationService.frequency_domain_shift, 'audio'),
            TransformationConfig('stereo_phase_inversion', 0.5, FFmpegTransformationService.stereo_phase_inversion, 'audio'),
            TransformationConfig('stereo_width_manipulation', 0.6, FFmpegTransformationService.stereo_width_manipulation, 'audio'),
            TransformationConfig('binaural_processing', 0.4, FFmpegTransformationService.binaural_processing, 'audio'),
            TransformationConfig('audio_reverse_segments', 0.5, FFmpegTransformationService.audio_reverse_segments, 'audio'),
            TransformationConfig('echo_delay_variation', 0.6, FFmpegTransformationService.echo_delay_variation, 'audio'),
            TransformationConfig('audio_chorus_effect', 0.6, FFmpegTransformationService.audio_chorus_effect, 'audio'),
            TransformationConfig('dynamic_range_compression', 0.6, FFmpegTransformationService.dynamic_range_compression, 'audio'),
            TransformationConfig('audio_time_stretching', 0.5, FFmpegTransformationService.audio_time_stretching, 'audio'),
            TransformationConfig('voice_pattern_disruption', 0.6, FFmpegTransformationService.voice_pattern_disruption, 'audio'),
            TransformationConfig('voice_formant_modification', 0.5, FFmpegTransformationService.voice_formant_modification, 'audio'),
            TransformationConfig('audio_panning_balance', 0.6, FFmpegTransformationService.audio_panning_balance, 'audio'),
            
            # SPECIALIZED AUDIO EFFECTS (4)
            TransformationConfig('vocal_range_compression', 0.5, FFmpegTransformationService.vocal_range_compression, 'audio'),
            TransformationConfig('pitch_shift_semitones', 0.6, FFmpegTransformationService.pitch_shift_semitones, 'audio'),
            TransformationConfig('enhanced_ambient_layering', 0.6, FFmpegTransformationService.enhanced_ambient_layering, 'audio'),
            TransformationConfig('audio_video_sync_offset', 0.5, FFmpegTransformationService.audio_video_sync_offset, 'audio'),
            
            # INSTAGRAM-SPECIFIC TRANSFORMATIONS (7)
            TransformationConfig('instagram_speed_micro_changes', 0.6, FFmpegTransformationService.instagram_speed_micro_changes, 'instagram'),
            TransformationConfig('instagram_pitch_shift_segments', 0.6, FFmpegTransformationService.instagram_pitch_shift_segments, 'instagram'),
            TransformationConfig('variable_frame_interpolation', 0.5, FFmpegTransformationService.variable_frame_interpolation, 'instagram'),
            TransformationConfig('instagram_rotation_micro', 0.6, FFmpegTransformationService.instagram_rotation_micro, 'instagram'),
            TransformationConfig('instagram_crop_resize_cycle', 0.6, FFmpegTransformationService.instagram_crop_resize_cycle, 'instagram'),
            TransformationConfig('instagram_brightness_pulse', 0.6, FFmpegTransformationService.instagram_brightness_pulse, 'instagram'),
            TransformationConfig('instagram_audio_ducking', 0.5, FFmpegTransformationService.instagram_audio_ducking, 'instagram'),
            
            # ADVANCED COLOR PROCESSING (4)
            TransformationConfig('color_channel_swapping', 0.6, FFmpegTransformationService.color_channel_swapping, 'visual'),
            TransformationConfig('chromatic_aberration_effect', 0.5, FFmpegTransformationService.chromatic_aberration_effect, 'visual'),
            TransformationConfig('selective_color_isolation', 0.5, FFmpegTransformationService.selective_color_isolation, 'visual'),
            TransformationConfig('color_space_conversion', 0.6, FFmpegTransformationService.color_space_conversion, 'visual'),
            
            # GEOMETRIC DISTORTIONS (4)
            TransformationConfig('perspective_distortion', 0.5, FFmpegTransformationService.perspective_distortion, 'visual'),
            TransformationConfig('barrel_distortion', 0.5, FFmpegTransformationService.barrel_distortion, 'visual'),
            TransformationConfig('optical_flow_stabilization', 0.4, FFmpegTransformationService.optical_flow_stabilization, 'visual'),
            TransformationConfig('random_geometric_warp', 0.6, FFmpegTransformationService.random_geometric_warp, 'visual'),
            
            # FILM AND TEXTURE EFFECTS (2)
            TransformationConfig('film_grain_simulation', 0.6, FFmpegTransformationService.film_grain_simulation, 'enhancement'),
            TransformationConfig('color_pulse_effect', 0.6, FFmpegTransformationService.color_pulse_effect, 'enhancement'),
            
            # METADATA SPOOFING EXTENSIONS (4) 
            TransformationConfig('gps_exif_randomization', 0.6, FFmpegTransformationService.gps_exif_randomization, 'metadata'),
            TransformationConfig('camera_settings_simulation', 0.6, FFmpegTransformationService.camera_settings_simulation, 'metadata'),
            TransformationConfig('software_version_cycling', 0.6, FFmpegTransformationService.software_version_cycling, 'metadata'),
            TransformationConfig('codec_parameter_variation', 0.6, FFmpegTransformationService.codec_parameter_variation, 'metadata'),
            
            # ADVANCED TEMPORAL EFFECTS (8)
            TransformationConfig('frame_trimming_dropout', 0.5, FFmpegTransformationService.frame_trimming_dropout, 'temporal'),
            TransformationConfig('random_frame_inserts', 0.4, FFmpegTransformationService.random_frame_inserts, 'temporal'),
            TransformationConfig('frame_reordering_segments', 0.4, FFmpegTransformationService.frame_reordering_segments, 'temporal'),
            TransformationConfig('complex_speed_patterns', 0.5, FFmpegTransformationService.complex_speed_patterns, 'temporal'),
            TransformationConfig('frame_micro_adjustments', 0.6, FFmpegTransformationService.frame_micro_adjustments, 'temporal'),
            TransformationConfig('random_cut_jitter_effects', 0.6, FFmpegTransformationService.random_cut_jitter_effects, 'temporal'),
            TransformationConfig('random_overlay_effects', 0.6, FFmpegTransformationService.random_overlay_effects, 'temporal'),
            TransformationConfig('random_motion_blur_effects', 0.6, FFmpegTransformationService.random_motion_blur_effects, 'temporal'),
            
            # ENHANCEMENT AND VISUAL EFFECTS (5)
            TransformationConfig('noise_blur_regions', 0.6, FFmpegTransformationService.noise_blur_regions, 'enhancement'),
            TransformationConfig('grayscale_segment', 0.5, FFmpegTransformationService.grayscale_segment, 'enhancement'),
            TransformationConfig('clip_embedding_shuffle', 0.6, FFmpegTransformationService.clip_embedding_shuffle, 'enhancement'),
        ]
    
    @staticmethod
    def phash_disruption_transform(input_path: str, output_path: str) -> str:
            """pHash Hamming Distance > 20-30 per keyframe (BALANCED for variation + originality)"""
            hue_shift = random.uniform(-20, 20)  # BALANCED: was ±35, now ±20
            saturation = random.uniform(0.75, 1.25)  # BALANCED: was 0.6-1.4, now 0.75-1.25
            brightness = random.uniform(-0.12, 0.12)  # BALANCED: was ±0.2, now ±0.12
            contrast = random.uniform(0.8, 1.2)  # BALANCED: was 0.7-1.3, now 0.8-1.2
            gamma = random.uniform(0.85, 1.15)  # BALANCED: was 0.7-1.3, now 0.85-1.15
            
            # BALANCED micro-rotation - subtle but effective
            rotation = random.uniform(-1.5, 1.5)  # BALANCED: was ±3.0, now ±1.5
            
            # BALANCED noise - noticeable for detection but not visual quality
            noise_strength = random.uniform(0.02, 0.05)  # BALANCED: was 0.05-0.10, now 0.02-0.05
            
            return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_shift}:s={saturation},eq=brightness={brightness}:contrast={contrast}:gamma={gamma},rotate={rotation}*PI/180:fillcolor=black,noise=alls={noise_strength}:allf=t" -c:a copy -y "{output_path}"'
    
    @staticmethod
    def ssim_reduction_transform(input_path: str, output_path: str) -> str:
        """SSIM Structural Similarity < 0.30 (BALANCED for variation while preserving content)"""
        crop_percent = random.uniform(0.03, 0.08)  # BALANCED: was 0.08-0.15, now 0.03-0.08
        pan_x = random.uniform(0, crop_percent * 0.6)  # BALANCED pan bounds
        pan_y = random.uniform(0, crop_percent * 0.6)  # BALANCED pan bounds
        noise_strength = random.uniform(0.02, 0.05)  # BALANCED: was 0.05-0.10, now 0.02-0.05
        
        # BALANCED blur and sharpening - effective but not destructive
        blur_strength = random.uniform(0.5, 1.2)  # BALANCED: was 1.0-2.5, now 0.5-1.2
        sharpen_strength = random.uniform(0.3, 0.8)  # BALANCED: was 0.5-1.2, now 0.3-0.8
        
        # BALANCED rotation
        rotation = random.uniform(-1.2, 1.2)  # BALANCED: was ±2.5, now ±1.2
        
        # Ensure safe values
        crop_percent = max(0.02, min(0.08, crop_percent))
        pan_x = max(0, min(crop_percent * 0.6, pan_x))
        pan_y = max(0, min(crop_percent * 0.6, pan_y))
        
        return f'ffmpeg -i "{input_path}" -vf "crop=iw*(1-{crop_percent:.3f}):ih*(1-{crop_percent:.3f}):iw*{pan_x:.3f}:ih*{pan_y:.3f},scale=2*trunc(iw/2):2*trunc(ih/2),noise=alls={noise_strength:.3f}:allf=t,unsharp=5:5:{sharpen_strength:.3f}:5:5:{blur_strength:.3f},rotate={rotation:.3f}*PI/180:fillcolor=black" -c:a copy -y "{output_path}"'
        
    @staticmethod
    def color_histogram_shift(input_path: str, output_path: str) -> str:
        """Color Histogram correlation < 0.40 (BALANCED for detection while preserving natural colors)"""
        hue_shift = random.uniform(-15, 15)  # BALANCED: was ±25, now ±15
        saturation = random.uniform(0.85, 1.15)  # BALANCED: was 0.7-1.3, now 0.85-1.15
        gamma_r = random.uniform(0.9, 1.1)  # BALANCED: was 0.8-1.2, now 0.9-1.1
        gamma_g = random.uniform(0.9, 1.1)
        gamma_b = random.uniform(0.9, 1.1)

        return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_shift}:s={saturation},eq=gamma_r={gamma_r}:gamma_g={gamma_g}:gamma_b={gamma_b}" -c:a copy -y "{output_path}"'

    @staticmethod
    def extreme_phash_disruption(input_path: str, output_path: str) -> str:
        """BALANCED pHash disruption for variation while maintaining watchability"""
        hue1 = random.uniform(-25, 25)  # BALANCED: was ±40, now ±25
        saturation1 = random.uniform(0.7, 1.3)  # BALANCED: was 0.5-1.5, now 0.7-1.3

        # BALANCED brightness/contrast adjustments
        brightness_adj = random.uniform(-0.15, 0.15)  # BALANCED: was ±0.25, now ±0.15
        contrast_adj = random.uniform(0.75, 1.25)  # BALANCED: was 0.6-1.4, now 0.75-1.25

        # BALANCED noise
        noise_strength = random.uniform(0.03, 0.08)  # BALANCED: was 0.08-0.15, now 0.03-0.08

        # BALANCED vignette
        vignette_strength = random.uniform(0.3, 0.6)  # BALANCED: was 0.5-0.9, now 0.3-0.6

        # BALANCED rotation
        rotation = random.uniform(-2.0, 2.0)  # BALANCED: was ±4.0, now ±2.0

        return f'ffmpeg -i "{input_path}" -vf "hue=h={hue1}:s={saturation1},eq=brightness={brightness_adj}:contrast={contrast_adj},noise=alls={noise_strength}:allf=t,vignette=PI/6*{vignette_strength},rotate={rotation}*PI/180:fillcolor=black" -c:a copy -y "{output_path}"'

    @staticmethod
    def frame_entropy_increase(input_path: str, output_path: str) -> str:
        """Frame Entropy increase 8-15% (INCREASED for high variation)"""
        noise_strength = random.uniform(0.05, 0.12)  # INCREASED from 0.01-0.025
        blur_strength = random.uniform(1.0, 2.5)  # INCREASED from 0.3-0.8

        return f'ffmpeg -i "{input_path}" -vf "noise=alls={noise_strength}:allf=t,unsharp=5:5:{blur_strength}:5:5:{blur_strength}" -c:a copy -y "{output_path}"'

    @staticmethod
    def embedding_similarity_change(input_path: str, output_path: str) -> str:
        """Enhanced CLIP Embedding distance change >= 0.35 with dynamic visual disruption"""
    
        # Get video properties for intelligent modifications
        try:
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', input_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            video_info = json.loads(result.stdout)
            
            # Extract video properties
            video_stream = next(s for s in video_info['streams'] if s['codec_type'] == 'video')
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            duration = float(video_info['format']['duration'])
            fps = eval(video_stream['r_frame_rate'])  # Convert fraction to float
        except:
            # Fallback values
            width, height, duration, fps = 1920, 1080, 30.0, 24.0
        
        # Enhanced text options with more variation
        watermark_texts = [
            'SAMPLE', 'PREVIEW', 'DEMO', '© 2025', 'ORIGINAL', 'HD QUALITY', 
            '🎬 EXCLUSIVE', '⭐ PREMIUM', 'DRAFT', 'UNFINISHED', 'BETA VERSION',
            '🔒 LOCKED', '💎 VIP', '🎯 BRANDED', 'PROTOTYPE', 'CONFIDENTIAL',
            'NOT FOR SALE', 'EVALUATION COPY', '📺 BROADCAST', '🌟 FEATURED'
        ]
        
        # Enhanced modification types
        modification_types = [
            'animated_watermark',
            'multi_layer_overlay',
            'corner_branding',
            'dynamic_text_scroll',
            'pulsing_watermark',
            'rotating_overlay',
            'gradient_overlay',
            'multi_position_stamps'
        ]
        
        mod_type = random.choice(modification_types)
        
        if mod_type == 'animated_watermark':
            # Animated moving watermark
            text = random.choice(watermark_texts)
            opacity = random.uniform(0.4, 0.7)
            fontsize = random.randint(48, 96)
            speed = random.uniform(50, 150)  # pixels per second
            
            # Animate across screen
            start_x = random.choice([-200, width + 200])
            end_x = width + 200 if start_x < 0 else -200
            y_pos = random.randint(height//4, 3*height//4)
            
            return f'''ffmpeg -i "{input_path}" -vf "drawtext=text='{text}':fontcolor=white@{opacity}:fontsize={fontsize}:x='if(lt(t,0),{start_x},{start_x}+({end_x}-{start_x})*t/{duration})':y={y_pos}:box=1:boxcolor=black@0.4:enable='between(t,0,{duration})'" -c:a copy -y "{output_path}"'''
        
        elif mod_type == 'multi_layer_overlay':
            # Multiple overlapping watermarks
            text1 = random.choice(watermark_texts)
            text2 = random.choice(watermark_texts)
            opacity1 = random.uniform(0.3, 0.5)
            opacity2 = random.uniform(0.2, 0.4)
            fontsize1 = random.randint(40, 70)
            fontsize2 = random.randint(30, 50)
            
            return f'''ffmpeg -i "{input_path}" -vf "drawtext=text='{text1}':fontcolor=red@{opacity1}:fontsize={fontsize1}:x=50:y=50:box=1:boxcolor=black@0.3,drawtext=text='{text2}':fontcolor=blue@{opacity2}:fontsize={fontsize2}:x=w-tw-50:y=h-th-50:box=1:boxcolor=white@0.2" -c:a copy -y "{output_path}"'''
        
        elif mod_type == 'corner_branding':
            # Enhanced corner watermarks with rotation
            text = random.choice(watermark_texts)
            opacity = random.uniform(0.4, 0.8)
            fontsize = random.randint(36, 80)
            angle = random.uniform(-15, 15)  # Slight rotation
            corner = random.choice(['tl', 'tr', 'bl', 'br'])
            
            positions = {
                'tl': 'x=20:y=40',
                'tr': 'x=w-tw-20:y=40', 
                'bl': 'x=20:y=h-th-20',
                'br': 'x=w-tw-20:y=h-th-20'
            }
            
            return f'''ffmpeg -i "{input_path}" -vf "drawtext=text='{text}':fontcolor=yellow@{opacity}:fontsize={fontsize}:{positions[corner]}:box=1:boxcolor=black@0.5:boxborderw=3" -c:a copy -y "{output_path}"'''
        
        elif mod_type == 'dynamic_text_scroll':
            # Scrolling text across screen
            text = random.choice(watermark_texts)
            opacity = random.uniform(0.5, 0.8)
            fontsize = random.randint(60, 120)
            direction = random.choice(['horizontal', 'vertical'])
            
            if direction == 'horizontal':
                return f'''ffmpeg -i "{input_path}" -vf "drawtext=text='{text}':fontcolor=cyan@{opacity}:fontsize={fontsize}:x='w-mod(50*t,w+tw)':y=h/2:box=1:boxcolor=black@0.4" -c:a copy -y "{output_path}"'''
            else:
                return f'''ffmpeg -i "{input_path}" -vf "drawtext=text='{text}':fontcolor=magenta@{opacity}:fontsize={fontsize}:x=w/2-tw/2:y='h-mod(30*t,h+th)':box=1:boxcolor=black@0.4" -c:a copy -y "{output_path}"'''
        
        elif mod_type == 'pulsing_watermark':
            # Pulsing/breathing effect
            text = random.choice(watermark_texts)
            base_opacity = random.uniform(0.3, 0.6)
            fontsize = random.randint(50, 100)
            pulse_speed = random.uniform(0.5, 2.0)
            
            return f'''ffmpeg -i "{input_path}" -vf "drawtext=text='{text}':fontcolor=white@'{base_opacity}+0.3*sin(2*PI*{pulse_speed}*t)':fontsize={fontsize}:x=(w-tw)/2:y=(h-th)/2:box=1:boxcolor=black@0.4" -c:a copy -y "{output_path}"'''
        
        elif mod_type == 'rotating_overlay':
            # Rotating watermark
            text = random.choice(watermark_texts)
            opacity = random.uniform(0.4, 0.7)
            fontsize = random.randint(40, 80)
            rotation_speed = random.uniform(10, 45)  # degrees per second
            
            return f'''ffmpeg -i "{input_path}" -vf "drawtext=text='{text}':fontcolor=orange@{opacity}:fontsize={fontsize}:x=(w-tw)/2:y=(h-th)/2:box=1:boxcolor=black@0.3" -c:a copy -y "{output_path}"'''
        
        elif mod_type == 'gradient_overlay':
            # Semi-transparent gradient overlay with text
            text = random.choice(watermark_texts)
            opacity = random.uniform(0.6, 0.9)
            fontsize = random.randint(72, 140)
            gradient_opacity = random.uniform(0.1, 0.3)
            
            # Create gradient overlay first, then add text
            return f'''ffmpeg -i "{input_path}" -vf "drawbox=x=0:y=h*3/4:w=w:h=h/4:color=black@{gradient_opacity}:t=fill,drawtext=text='{text}':fontcolor=white@{opacity}:fontsize={fontsize}:x=(w-tw)/2:y=h*7/8-th/2" -c:a copy -y "{output_path}"'''
        
        elif mod_type == 'multi_position_stamps':
            # Multiple small stamps at different positions
            text = random.choice(watermark_texts[:5])  # Shorter texts for stamps
            opacity = random.uniform(0.4, 0.6)
            fontsize = random.randint(24, 48)
            
            # Generate 3-5 random positions
            num_stamps = random.randint(3, 5)
            stamp_filter = ""
            
            for i in range(num_stamps):
                x_pos = random.randint(20, max(20, width - 200))
                y_pos = random.randint(20, max(20, height - 100))
                if i == 0:
                    stamp_filter = f"drawtext=text='{text}':fontcolor=lime@{opacity}:fontsize={fontsize}:x={x_pos}:y={y_pos}:box=1:boxcolor=black@0.2"
                else:
                    stamp_filter += f",drawtext=text='{text}':fontcolor=lime@{opacity}:fontsize={fontsize}:x={x_pos}:y={y_pos}:box=1:boxcolor=black@0.2"
            
            return f'''ffmpeg -i "{input_path}" -vf "{stamp_filter}" -c:a copy -y "{output_path}"'''
        
        # Enhanced default fallback with random effects
        text = random.choice(watermark_texts)
        opacity = random.uniform(0.4, 0.8)
        fontsize = random.randint(48, 96)
        
        # Random advanced positioning
        positioning_styles = [
            'x=(w-tw)/2:y=50',  # Top center
            'x=(w-tw)/2:y=h-th-50',  # Bottom center
            'x=50:y=(h-th)/2',  # Left center  
            'x=w-tw-50:y=(h-th)/2',  # Right center
            f'x={(width//4)}:y={(height//4)}',  # Custom position
            'x=(w-tw)/2:y=(h-th)/2'  # Dead center
        ]
        
        position = random.choice(positioning_styles)
        
        # Random color schemes
        colors = ['white', 'yellow', 'cyan', 'lime', 'orange', 'red', 'magenta']
        color = random.choice(colors)
        
        return f'''ffmpeg -i "{input_path}" -vf "drawtext=text='{text}':fontcolor={color}@{opacity}:fontsize={fontsize}:{position}:box=1:boxcolor=black@0.4:boxborderw=2" -c:a copy -y "{output_path}"'''

        # BALANCED HIGH VARIATION TRANSFORMATIONS (New aggressive effects with originality preservation)
    @staticmethod
    def aggressive_color_shift(input_path: str, output_path: str) -> str:
        """BALANCED aggressive color shifting for variation while preserving natural look"""
        # Multiple color shifts combined but more conservative
        hue_shift1 = random.uniform(-30, 30)  # BALANCED: was ±50, now ±30
        hue_shift2 = random.uniform(-20, 20)  # BALANCED: was ±30, now ±20
        saturation1 = random.uniform(0.7, 1.3)  # BALANCED: was 0.4-1.6, now 0.7-1.3
        saturation2 = random.uniform(0.8, 1.2)  # BALANCED: was 0.6-1.4, now 0.8-1.2

        # Channel mixing for more complex color changes - more subtle
        channel_mix = random.choice([
            "colorchannelmixer=rr=0.9:rg=0.1:rb=0.05:gr=0.05:gg=0.95:gb=0.05:br=0.05:bg=0.05:bb=0.9",
            "colorchannelmixer=rr=0.85:rg=0.15:rb=0.1:gr=0.1:gg=0.9:gb=0.1:br=0.1:bg=0.1:bb=0.85",
            "colorchannelmixer=rr=0.95:rg=0.05:rb=0.0:gr=0.0:gg=1.0:gb=0.05:br=0.05:bg=0.0:bb=0.95"
        ])

        return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_shift1}:s={saturation1},{channel_mix},hue=h={hue_shift2}:s={saturation2}" -c:a copy -y "{output_path}"'

    @staticmethod
    def aggressive_geometric_distortion(input_path: str, output_path: str) -> str:
        """BALANCED geometric distortions for spatial differences while preserving content"""
        distortion_type = random.choice(['barrel', 'pincushion', 'trapezoid', 'skew'])

        if distortion_type == 'barrel':
            # Barrel distortion - more subtle
            k1 = random.uniform(0.05, 0.15)  # BALANCED: was 0.1-0.3, now 0.05-0.15
            return f'ffmpeg -i "{input_path}" -vf "lenscorrection=k1={k1}" -c:a copy -y "{output_path}"'

        elif distortion_type == 'pincushion':
            # Pincushion distortion - more subtle
            k1 = random.uniform(-0.15, -0.05)  # BALANCED: was -0.3 to -0.1, now -0.15 to -0.05
            return f'ffmpeg -i "{input_path}" -vf "lenscorrection=k1={k1}" -c:a copy -y "{output_path}"'

        elif distortion_type == 'trapezoid':
            # Trapezoid perspective - more subtle
            offset = random.randint(5, 15)  # BALANCED: was 10-30, now 5-15
            direction = random.choice([-1, 1])
            shift = offset * direction
            coords = f"0:0:W+{shift}:0:0:H:W-{shift}:H"
            return f'ffmpeg -i "{input_path}" -vf "perspective={coords}" -c:a copy -y "{output_path}"'

        else:  # skew
            # Skew - more subtle
            shear = random.randint(3, 8)  # BALANCED: was 5-15, now 3-8
            coords = f"{shear}:0:W-{shear}:0:0:H:W:H"
            return f'ffmpeg -i "{input_path}" -vf "perspective={coords}" -c:a copy -y "{output_path}"'

    @staticmethod
    def aggressive_temporal_manipulation(input_path: str, output_path: str) -> str:
        """BALANCED temporal effects for timing differences while preserving flow"""
        manipulation_type = random.choice(['speed_variation', 'frame_select', 'micro_cuts'])

        if manipulation_type == 'speed_variation':
            # Variable speed changes - more subtle with proper audio sync
            speed_factor = random.uniform(0.92, 1.08)  # BALANCED: was 0.8-1.2, now 0.92-1.08
            atempo_filter = FFmpegTransformationService.validate_atempo_value(speed_factor)
            return f'ffmpeg -i "{input_path}" -filter_complex "[0:v]setpts={1/speed_factor}*PTS[v];[0:a]{atempo_filter}[a]" -map "[v]" -map "[a]" -y "{output_path}"'

        elif manipulation_type == 'frame_select':
            # Strategic frame selection - more conservative
            select_pattern = random.choice([
                "select='not(mod(n\\,30))'",  # Skip 1 frame every 30
                "select='not(mod(n\\,25))'",  # Skip 1 frame every 25
                "select='gt(scene\\,0.01)'"   # Scene change detection
            ])
            return f'ffmpeg -i "{input_path}" -vf "{select_pattern}" -c:a copy -y "{output_path}"'

        else:  # micro_cuts
            # Micro cuts with frame rate adjustment
            fps_adjustment = random.uniform(0.98, 1.02)  # Very subtle
            return f'ffmpeg -i "{input_path}" -r 30*{fps_adjustment} -c:a copy -y "{output_path}"'

    @staticmethod
    def aggressive_audio_manipulation(input_path: str, output_path: str) -> str:
        """BALANCED audio changes for fingerprint breaking while preserving quality"""
        audio_type = random.choice(['eq_shift', 'phase_shift', 'stereo_adjust'])

        if audio_type == 'eq_shift':
            # Multi-band EQ shifting - more conservative
            low_gain = random.uniform(-1, 1)  # REDUCED: was ±2, now ±1
            mid_gain = random.uniform(-0.8, 0.8)  # REDUCED: was ±1.5, now ±0.8
            high_gain = random.uniform(-1, 1)  # REDUCED: was ±2, now ±1
            return f'ffmpeg -i "{input_path}" -af "equalizer=f=200:width_type=h:width=100:g={low_gain},equalizer=f=2000:width_type=h:width=500:g={mid_gain},equalizer=f=8000:width_type=h:width=2000:g={high_gain}" -c:v copy -y "{output_path}"'

        elif audio_type == 'phase_shift':
            # Stereo phase adjustment
            phase_shift = random.uniform(-0.15, 0.15)  # REDUCED: was ±0.3, now ±0.15
            return f'ffmpeg -i "{input_path}" -af "aphaser=in_gain=0.4:out_gain=0.74:delay=3:decay=0.4:speed=0.5" -c:v copy -y "{output_path}"'

        else:  # stereo_adjust
            # Stereo width adjustment - subtle
            width_factor = random.uniform(0.9, 1.1)  # REDUCED: was 0.8-1.2, now 0.9-1.1
            return f'ffmpeg -i "{input_path}" -af "aecho=0.8:0.88:6:0.4" -c:v copy -y "{output_path}"'

        # AUDIO TRANSFORMATIONS - BALANCED HIGH VARIATION VALUES (Preserving Audio Quality)
    @staticmethod
    def spectral_fingerprint_disruption(input_path: str, output_path: str) -> str:
        """Spectral Fingerprint Match < 45% (REDUCED for better audio quality)"""
        pitch_shift = 1.0 + random.uniform(-0.04, 0.04)  # REDUCED: was ±0.08, now ±0.04
        tempo_shift = 1.0 + random.uniform(-0.03, 0.03)  # REDUCED: was ±0.06, now ±0.03
        bass_gain = random.uniform(-1.5, 1.5)  # REDUCED: was ±3, now ±1.5
        treble_gain = random.uniform(-1.5, 1.5)  # REDUCED: was ±3, now ±1.5

        # REDUCED volume variation
        volume_gain = random.uniform(-1, 1)  # REDUCED: was ±2, now ±1

        rate_factor = pitch_shift
        tempo_compensation = tempo_shift / pitch_shift
        return f'ffmpeg -i "{input_path}" -af "asetrate=44100*{rate_factor},atempo={tempo_compensation},bass=g={bass_gain},treble=g={treble_gain},volume={volume_gain}dB" -c:v copy -y "{output_path}"'

    @staticmethod
    def pitch_shift_transform(input_path: str, output_path: str) -> str:
        """Pitch Shift ±3-4% (REDUCED for audio quality without distortion)"""
        pitch_cents = random.uniform(-100, 100)  # REDUCED: was ±200, now ±100 (±1 semitone)
        pitch_ratio = 2 ** (pitch_cents / 1200)

        rate_factor = pitch_ratio
        tempo_compensation = 1.0 / rate_factor
        return f'ffmpeg -i "{input_path}" -af "asetrate=44100*{rate_factor},atempo={tempo_compensation}" -c:v copy -y "{output_path}"'

    @staticmethod
    def tempo_shift_transform(input_path: str, output_path: str) -> str:
        """Tempo Shift ±2-3% (REDUCED for natural flow without noticeable change)"""
        tempo_factor = 1.0 + random.uniform(-0.03, 0.03)  # REDUCED: was ±0.06, now ±0.03

        return f'ffmpeg -i "{input_path}" -af "atempo={tempo_factor}" -c:v copy -y "{output_path}"'

    @staticmethod
    def audio_layering_ambient(input_path: str, output_path: str) -> str:
        """Audio Layering with ambient noise -20dB to -30dB (BALANCED noise for fingerprint breaking)"""
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', input_path
            ], capture_output=True, text=True)

            duration = 10  # default
            if result.returncode == 0:
                try:
                    duration = float(result.stdout.strip())
                except:
                    pass

            noise_volume = random.uniform(0.001, 0.003)  # REDUCED: was 0.002-0.008, now 0.001-0.003
            noise_types = ['pink', 'brown', 'white']
            noise_type = random.choice(noise_types)

            return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={noise_type}:a={noise_volume}" -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.08[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'

        except Exception:
            gain_db = random.uniform(-0.8, 0.8)  # REDUCED: was ±1.5, now ±0.8
            return f'ffmpeg -i "{input_path}" -af "volume={gain_db}dB" -c:v copy -y "{output_path}"'

    @staticmethod
    def audio_segment_reorder(input_path: str, output_path: str) -> str:
        """Audio Segment Reorder ≤ 10% of total audio (BALANCED for detection breaking)"""
        offset = random.uniform(-0.15, 0.15)  # REDUCED: was ±0.3, now ±0.15

        return f'ffmpeg -i "{input_path}" -itsoffset {offset} -i "{input_path}" -map 0:v -map 1:a -c:v copy -c:a aac -y "{output_path}"'

    @staticmethod
    def audio_simple_processing(input_path: str, output_path: str) -> str:
        """Enhanced Audio Processing - BALANCED for fingerprint disruption while preserving quality"""
        bass_gain = random.uniform(-1.5, 1.5)  # REDUCED: was ±3, now ±1.5
        treble_gain = random.uniform(-1.2, 1.2)  # REDUCED: was ±2.5, now ±1.2
        volume_gain = random.uniform(-1, 1)  # REDUCED: was ±2, now ±1

        mid_gain = random.uniform(-1, 1)  # REDUCED: was ±2, now ±1

        return f'ffmpeg -i "{input_path}" -af "equalizer=f=100:width_type=h:width=50:g={bass_gain},equalizer=f=1000:width_type=h:width=200:g={mid_gain},equalizer=f=8000:width_type=h:width=1000:g={treble_gain},volume={volume_gain}dB" -c:v copy -y "{output_path}"'

        # COMPREHENSIVE ORB BREAKING STRATEGY IMPLEMENTATION
        # 🎯 Strategy Goals:
        # - Reduce ORB similarity < 9000
        # - Maintain SSIM ≥ 0.35, pHash distance ~25–35
        # - Audio similarity ↓, Metadata similarity ↓
        # - Maintain watchability ≥ 90%
        # - Support adaptive randomization across 7 transformation layers

        # 1. CORE ORB DISRUPTORS (always-on)


    @staticmethod
    def micro_perspective_warp_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Micro Perspective Warp: ±2–4px - disrupts keypoint geometry"""

        # Randomly choose between different reliable transformation methods
        method = random.choice(['rotation', 'crop_shift', 'simple_perspective', 'shear'])

        if method == 'rotation':
            # Micro rotation - most reliable
            angle = random.uniform(-1.0, 1.0)  # Slightly larger range for enhanced effect
            return f'ffmpeg -i "{input_path}" -vf "rotate={angle}*PI/180:fillcolor=black" -c:a copy -y "{output_path}"'

        elif method == 'crop_shift':
            # Enhanced crop with larger offsets
            offset_x = random.randint(2, 4)
            offset_y = random.randint(2, 4)
            crop_amount = random.randint(4, 8)  # Larger crop for enhanced effect

            return f'ffmpeg -i "{input_path}" -vf "crop=iw-{crop_amount}:ih-{crop_amount}:{offset_x}:{offset_y}" -c:a copy -y "{output_path}"'

        elif method == 'simple_perspective':
            # Simple working perspective transformation - Fixed to prevent black screen
            offset = random.randint(1, 2)  # Reduced offset to prevent extreme distortion
            direction = random.choice([-1, 1])
            shift = offset * direction

            # Use safe coordinates that keep the image visible
            # Format: x0:y0:x1:y1:x2:y2:x3:y3 (top-left, top-right, bottom-left, bottom-right)
            coords = f"0:0:W+{shift}:0:0:H:W+{shift}:H"
            return f'ffmpeg -i "{input_path}" -vf "perspective={coords}" -c:a copy -y "{output_path}"'

        else:  # shear
            # Enhanced shear effect - Fixed to prevent black screen
            shear_intensity = random.randint(1, 3)  # Reduced from 3-6 to prevent extreme distortion
            direction = random.choice(['horizontal', 'vertical'])

            if direction == 'horizontal':
                # Horizontal shear: keep top corners stable, shift bottom corners
                coords = f"0:0:W:0:{shear_intensity}:H:W-{shear_intensity}:H"
            else:
                # Vertical shear: keep left corners stable, shift right corners  
                coords = f"0:0:W-{shear_intensity}:0:0:H:W:{shear_intensity}"

            return f'ffmpeg -i "{input_path}" -vf "perspective={coords}" -c:a copy -y "{output_path}"'

    @staticmethod
    def frame_jittering_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Frame Jittering: 1–2px per 4–5 frames - invisible but hits ORB hard"""
        shift_x = random.uniform(1, 2)
        shift_y = random.uniform(1, 2) 
        frame_interval = random.randint(4, 5)

        # Apply micro-shifts with frame interval control
        jitter_filter = f"crop=iw-{shift_x*2}:ih-{shift_y*2}:{shift_x}:{shift_y},pad=iw+{shift_x*2}:ih+{shift_y*2}:{shift_x}:{shift_y}"

        return f'ffmpeg -i "{input_path}" -vf "{jitter_filter}" -c:a copy -y "{output_path}"'

    @staticmethod
    def dynamic_rotation_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Dynamic Rotation: ±1.0–1.5° every 2–3s - alters keypoint structure"""
        rotation_amplitude = random.uniform(1.0, 1.5)
        rotation_period = random.uniform(2, 3)

        # Apply rotation with time-based oscillation
        rotation_expr = f"rotate={rotation_amplitude}*sin(t/{rotation_period}*2*PI)*PI/180:fillcolor=black"

        return f'ffmpeg -i "{input_path}" -vf "{rotation_expr}" -c:a copy -y "{output_path}"'

        # 2. VISUAL VARIATION (70% prob)

    @staticmethod
    def zoom_jitter_motion_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Zoom Jitter Motion with multiple random transformation methods"""

        # Randomly choose transformation method
        method = random.choice(['scale_crop', 'pad_crop', 'hardcoded_zoom', 'micro_scale'])

        if method == 'scale_crop':
            # Dynamic scale and crop with calculated values
            zoom_factor = random.uniform(1.01, 1.03)
            jitter_x = random.randint(-4, 4)
            jitter_y = random.randint(-4, 4)

            # Calculate exact dimensions for 720x1280
            new_width = int(720 * zoom_factor)
            new_height = int(1280 * zoom_factor)

            # Ensure positive crop coordinates
            crop_x = max(0, abs(jitter_x))
            crop_y = max(0, abs(jitter_y))

            return f'ffmpeg -i "{input_path}" -vf "scale={new_width}:{new_height},crop=720:1280:{crop_x}:{crop_y}" -c:a copy -y "{output_path}"'

        elif method == 'pad_crop':
            # Pad then crop method for zoom effect
            zoom_intensity = random.randint(15, 35)  # Enhanced padding
            jitter_x = random.randint(-4, 4)
            jitter_y = random.randint(-4, 4)

            center_x = zoom_intensity + jitter_x + 5
            center_y = zoom_intensity + jitter_y + 5

            return f'ffmpeg -i "{input_path}" -vf "pad=iw+{zoom_intensity*2}:ih+{zoom_intensity*2}:{zoom_intensity}:{zoom_intensity}:black,crop=iw-{zoom_intensity*2}:ih-{zoom_intensity*2}:{center_x}:{center_y}" -c:a copy -y "{output_path}"'

        elif method == 'hardcoded_zoom':
            # Pre-calculated zoom transformations with enhanced jitter
            enhanced_zooms = [
                "scale=742:1318,crop=720:1280:11:19",   # 1.031x zoom, jitter
                "scale=744:1324,crop=720:1280:12:22",   # 1.033x zoom, jitter
                "scale=741:1316,crop=720:1280:10:18",   # 1.029x zoom, jitter
                "scale=743:1321,crop=720:1280:11:20",   # 1.032x zoom, jitter
                "scale=745:1326,crop=720:1280:13:23",   # 1.035x zoom, jitter
                "scale=740:1315,crop=720:1280:9:17",    # 1.028x zoom, jitter
                "scale=746:1328,crop=720:1280:14:24",   # 1.036x zoom, jitter
                "scale=739:1314,crop=720:1280:8:16",    # 1.026x zoom, jitter
            ]

            selected_zoom = random.choice(enhanced_zooms)
            return f'ffmpeg -i "{input_path}" -vf "{selected_zoom}" -c:a copy -y "{output_path}"'

        else:  # micro_scale
            # Micro scaling with crop offset
            scale_options = [
                (742, 1318, random.randint(8, 15), random.randint(15, 25)),
                (744, 1324, random.randint(10, 17), random.randint(18, 28)),
                (741, 1316, random.randint(7, 14), random.randint(14, 24)),
                (743, 1321, random.randint(9, 16), random.randint(16, 26)),
            ]

            scale_w, scale_h, crop_x, crop_y = random.choice(scale_options)
            return f'ffmpeg -i "{input_path}" -vf "scale={scale_w}:{scale_h},crop=720:1280:{crop_x}:{crop_y}" -c:a copy -y "{output_path}"'

    @staticmethod
    def color_histogram_shift_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Color Histogram Shift: ±3–5% - affects keypoint contrast areas"""
        hue_shift = random.uniform(-5, 5)
        saturation = random.uniform(0.95, 1.05)  # ±3-5%

        return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_shift}:s={saturation}" -c:a copy -y "{output_path}"'

    @staticmethod
    def line_sketch_filter_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Line Sketch Filter: 2–5% opacity - misleads ORB with artificial lines"""
        edge_opacity = random.uniform(0.02, 0.05)
        edge_threshold = random.uniform(0.1, 0.3)

        # Create subtle edge detection overlay
        edge_filter = f"[0:v]edgedetect=low={edge_threshold}:high={edge_threshold*3}[edges];[0:v][edges]blend=all_mode=overlay:all_opacity={edge_opacity}"

        return f'ffmpeg -i "{input_path}" -filter_complex "{edge_filter}" -c:a copy -y "{output_path}"'

        # 3. STRUCTURED RANDOMIZER (60% prob)
    @staticmethod
    def entropy_boost_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Entropy Boost - increases frame complexity"""
        noise_strength = random.uniform(0.005, 0.015)
        grain_strength = random.uniform(0.3, 0.8)

        return f'ffmpeg -i "{input_path}" -vf "noise=alls={noise_strength}:allf=t,unsharp=5:5:{grain_strength}:5:5:{grain_strength}" -c:a copy -y "{output_path}"'

    @staticmethod
    def clip_embedding_shift_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced CLIP Embedding Shift - changes semantic understanding"""
        emojis = ['🎵', '🎬', '⭐', '🔥', '💎', '🚀', '💯', '🌟']
        emoji = random.choice(emojis)
        text_overlays = [
            f"{emoji} ORIGINAL {emoji}",
            f"{emoji} HD QUALITY {emoji}",
            f"{emoji} EXCLUSIVE {emoji}"
        ]
        text = random.choice(text_overlays)
        opacity = random.uniform(0.03, 0.07)

        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{text}\':fontcolor=white@{opacity}:fontsize=16:x=(w-tw)/2:y=h-th-10" -c:a copy -y "{output_path}"'

    @staticmethod
    def phash_disruption_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced pHash Disruption - targets perceptual hash algorithms"""
        hue_shift = random.uniform(-8, 8)
        saturation = random.uniform(0.92, 1.08)
        brightness = random.uniform(-0.05, 0.05)
        contrast = random.uniform(0.95, 1.05)

        # Add micro-rotation for additional disruption
        rotation = random.uniform(-0.5, 0.5)

        return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_shift}:s={saturation},eq=brightness={brightness}:contrast={contrast},rotate={rotation}*PI/180:fillcolor=black" -c:a copy -y "{output_path}"'

        # 4. STABILITY ENHANCER (50% prob)
    @staticmethod
    def ssim_reduction_controlled(input_path: str, output_path: str) -> str:
        """Controlled SSIM Reduction - maintains watchability while reducing similarity"""
        crop_percent = random.uniform(0.01, 0.03)  # Very subtle
        pan_x = random.uniform(0, crop_percent * 0.3)
        pan_y = random.uniform(0, crop_percent * 0.3)

        return f'ffmpeg -i "{input_path}" -vf "crop=iw*(1-{crop_percent:.3f}):ih*(1-{crop_percent:.3f}):iw*{pan_x:.3f}:ih*{pan_y:.3f},scale=2*trunc(iw/2):2*trunc(ih/2)" -c:a copy -y "{output_path}"'

        # 5. AUDIO TRANSFORM (50% prob)
    @staticmethod
    def pitch_shift_transform_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Pitch Shift - audio fingerprint breaking (REDUCED)"""
        pitch_cents = random.uniform(-40, 40)  # REDUCED: was ±80, now ±40
        pitch_ratio = 2 ** (pitch_cents / 1200)

        rate_factor = pitch_ratio
        tempo_compensation = 1.0 / rate_factor
        return f'ffmpeg -i "{input_path}" -af "asetrate=44100*{rate_factor},atempo={tempo_compensation}" -c:v copy -y "{output_path}"'

    @staticmethod
    def tempo_shift_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Tempo Shift - breaks audio timing patterns (REDUCED)"""
        tempo_factor = 1.0 + random.uniform(-0.015, 0.015)  # REDUCED: was ±0.03, now ±0.015

        return f'ffmpeg -i "{input_path}" -af "atempo={tempo_factor}" -c:v copy -y "{output_path}"'

    @staticmethod
    def add_ambient_noise_enhanced(input_path: str, output_path: str) -> str:

        """Enhanced Ambient Noise - Fixed audio fingerprinting disruption with multiple techniques"""
        try:
            # Get comprehensive audio properties
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', input_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                video_info = json.loads(result.stdout)
                duration = float(video_info['format']['duration'])
                
                # Get audio stream info
                audio_streams = [s for s in video_info['streams'] if s['codec_type'] == 'audio']
                if audio_streams:
                    audio_stream = audio_streams[0]
                    sample_rate = int(audio_stream.get('sample_rate', 44100))
                    channels = int(audio_stream.get('channels', 2))
                else:
                    sample_rate, channels = 44100, 2
            else:
                duration, sample_rate, channels = 30.0, 44100, 2
                
        except Exception:
            duration, sample_rate, channels = 30.0, 44100, 2
        
        # Enhanced audio processing techniques
        processing_types = [
            'layered_ambient_noise',
            'frequency_specific_noise', 
            'dynamic_noise_modulation',
            'spectral_noise_injection',
            'harmonic_distortion',
            'phase_shift_noise',
            'multi_band_processing',
            'temporal_noise_variation',
            'psychoacoustic_masking'
        ]
        
        processing_type = random.choice(processing_types)
        
        if processing_type == 'layered_ambient_noise':
            # Multiple layers of different noise types
            noise_types = ['pink', 'brown', 'white', 'blue', 'violet']
            primary_noise = random.choice(noise_types)
            secondary_noise = random.choice([n for n in noise_types if n != primary_noise])
            
            primary_volume = random.uniform(0.0001, 0.0008)
            secondary_volume = random.uniform(0.0001, 0.0005)
            mix_ratio = random.uniform(0.3, 0.7)
            mix_ratio2 = 1.0 - mix_ratio  # Ensure weights sum to 1
            
            return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={primary_noise}:a={primary_volume}:r={sample_rate}" -f lavfi -i "anoisesrc=d={duration}:c={secondary_noise}:a={secondary_volume}:r={sample_rate}" -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.03[a1];[2:a]volume=0.02[a2];[a1][a2]amix=inputs=2:duration=first:weights={mix_ratio:.3f} {mix_ratio2:.3f}[noise_mix];[a0][noise_mix]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 192k -y "{output_path}"'
        
        elif processing_type == 'frequency_specific_noise':
            # Target specific frequency ranges - SIMPLIFIED
            freq_ranges = [
                (100, 500),    # Low frequencies
                (500, 2000),   # Mid frequencies  
                (2000, 8000),  # High frequencies
                (8000, 16000)  # Very high frequencies
            ]
            
            target_freq = random.choice(freq_ranges)
            noise_type = random.choice(['pink', 'brown', 'white'])
            noise_volume = random.uniform(0.0002, 0.0012)
            center_freq = (target_freq[0] + target_freq[1]) // 2
            width = target_freq[1] - target_freq[0]
            
            return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={noise_type}:a={noise_volume}:r={sample_rate}" -filter_complex "[0:a]volume=1.0[a0];[1:a]bandpass=f={center_freq}:width_type=h:w={width},volume=0.04[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
        
        elif processing_type == 'dynamic_noise_modulation':
            # FIXED: Pre-calculate modulation values to avoid NaN
            noise_type = random.choice(['pink', 'brown', 'white'])
            base_volume = random.uniform(0.0003, 0.001)
            modulation_freq = random.uniform(0.1, 2.0)
            modulation_depth = random.uniform(0.3, 0.8)
            
            # Calculate static volume instead of dynamic expression
            static_volume = 0.05 * (1 + modulation_depth * 0.5)  # Average modulation
            
            return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={noise_type}:a={base_volume}:r={sample_rate}" -filter_complex "[0:a]volume=1.0[a0];[1:a]volume={static_volume:.4f}[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
        
        elif processing_type == 'spectral_noise_injection':
            # SIMPLIFIED: Single frequency injection
            noise_type = random.choice(['pink', 'brown'])
            noise_volume = random.uniform(0.0002, 0.0008)
            
            # Single narrow band injection
            freq1 = random.randint(200, 1000)
            width1 = random.randint(50, 200)
            
            return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={noise_type}:a={noise_volume}:r={sample_rate}" -filter_complex "[0:a]volume=1.0[a0];[1:a]bandpass=f={freq1}:width_type=h:w={width1},volume=0.02[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
        
        elif processing_type == 'harmonic_distortion':
            # SIMPLIFIED: Remove overdrive filter that might cause issues
            noise_type = random.choice(['pink', 'brown'])
            noise_volume = random.uniform(0.0001, 0.0006)
            gain_adjust = random.uniform(0.98, 1.02)
            
            return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={noise_type}:a={noise_volume}:r={sample_rate}" -filter_complex "[0:a]volume={gain_adjust:.3f}[a0];[1:a]volume=0.025[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
        
        elif processing_type == 'phase_shift_noise':
            # SIMPLIFIED: Remove complex phase shifting
            noise_type = random.choice(['white', 'pink'])
            noise_volume = random.uniform(0.0002, 0.0009)
            
            if channels >= 2:
                return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={noise_type}:a={noise_volume}:r={sample_rate}" -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.03[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
            else:
                return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={noise_type}:a={noise_volume}:r={sample_rate}" -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.03[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
        
        elif processing_type == 'multi_band_processing':
            # SIMPLIFIED: Basic EQ instead of complex splitting
            noise_volume = random.uniform(0.0003, 0.001)
            eq_freq = random.choice([500, 1000, 2000, 4000])
            eq_gain = random.uniform(-2, 2)
            
            return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c=brown:a={noise_volume}:r={sample_rate}" -filter_complex "[0:a]equalizer=f={eq_freq}:width_type=h:width=100:g={eq_gain:.2f}[a0];[1:a]volume=0.02[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
        
        elif processing_type == 'temporal_noise_variation':
            # FIXED: Use static volume switching at specific times
            segment_duration = max(2.0, duration / random.randint(3, 8))
            noise_type = random.choice(['pink', 'brown', 'white'])
            volume1 = random.uniform(0.01, 0.03)
            volume2 = random.uniform(0.02, 0.04)
            
            # Switch between two volumes at midpoint
            switch_time = duration / 2
            
            return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={noise_type}:a=0.0005:r={sample_rate}" -filter_complex "[0:a]volume=1.0[a0];[1:a]volume={volume1:.3f}:enable=\'between(t,0,{switch_time:.1f})\',volume={volume2:.3f}:enable=\'gte(t,{switch_time:.1f})\'[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
        
        elif processing_type == 'psychoacoustic_masking':
            # SIMPLIFIED: Basic frequency targeting
            noise_type = random.choice(['pink', 'brown'])
            noise_volume = random.uniform(0.0004, 0.0012)
            masking_freq = random.randint(2000, 5000)
            masking_width = random.randint(200, 800)
            
            return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={noise_type}:a={noise_volume}:r={sample_rate}" -filter_complex "[0:a]volume=1.0[a0];[1:a]bandpass=f={masking_freq}:width_type=h:w={masking_width},volume=0.018[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
        
        # Enhanced fallback with multiple noise sources - SIMPLIFIED
        try:
            primary_noise = random.choice(['pink', 'brown', 'white'])
            secondary_noise = random.choice(['pink', 'brown', 'white'])
            primary_volume = random.uniform(0.0002, 0.0008)
            secondary_volume = random.uniform(0.0001, 0.0005)
            
            return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={primary_noise}:a={primary_volume}:r={sample_rate}" -f lavfi -i "anoisesrc=d={duration}:c={secondary_noise}:a={secondary_volume}:r={sample_rate}" -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.025[a1];[2:a]volume=0.015[a2];[a1][a2]amix=inputs=2:duration=first[noise_mix];[a0][noise_mix]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 192k -y "{output_path}"'
            
        except Exception:
            # Final fallback - VERY SIMPLE
            gain_db = random.uniform(-0.2, 0.2)
            eq_freq = random.choice([500, 1000, 2000])
            eq_gain = random.uniform(-1, 1)
            
            return f'ffmpeg -i "{input_path}" -af "volume={gain_db:.2f}dB,equalizer=f={eq_freq}:width_type=h:width=500:g={eq_gain:.2f}" -c:v copy -c:a aac -y "{output_path}"'
        # 6. METADATA LAYER (100%) - ENHANCED FOR MAXIMUM SIMILARITY REDUCTION
    @staticmethod
    def ultra_metadata_randomization(input_path: str, output_path: str) -> str:
        """Ultra Metadata Randomization - completely strips and randomizes all metadata"""
        import secrets
        import uuid

        # Generate highly randomized metadata with much more variation
        random_uuid = str(uuid.uuid4())
        random_title = f"VID_{secrets.token_hex(8).upper()}"
        random_comment = f"Processed_{secrets.token_urlsafe(12)}"
        random_software = random.choice([
            "Adobe Premiere Pro 2024.2", "Final Cut Pro 10.7.1", "DaVinci Resolve 19.0",
            "OpenShot 3.2.0", "Kdenlive 23.08", "Filmora 13.1", "VSDC 9.2",
            "HandBrake 1.7.3", "VEGAS Pro 21", "Avid Media Composer 2024",
            "Blender 4.0", "OBS Studio 30.0", "FFmpeg 6.1"
        ])
        random_artist = f"User_{secrets.token_hex(4)}"
        random_album = f"Collection_{secrets.randbelow(9999)}"
        random_genre = random.choice(["Video", "Content", "Media", "Digital", "Original"])

        # Randomize creation time with wider range
        days_back = random.randint(1, 365)  # Up to 1 year back
        hours_offset = random.randint(0, 23)
        minutes_offset = random.randint(0, 59)
        random_date = (dt.datetime.now() - dt.timedelta(days=days_back, hours=hours_offset, minutes=minutes_offset)).strftime('%Y-%m-%d %H:%M:%S')

        # Add encoder metadata randomization
        random_encoder = random.choice([
            "libx264", "libx265", "h264_videotoolbox", "hevc_videotoolbox",
            "libvpx-vp9", "libaom-av1", "h264_nvenc", "hevc_nvenc"
        ])

        return f'ffmpeg -i "{input_path}" -map_metadata -1 -metadata creation_time="{random_date}" -metadata title="{random_title}" -metadata comment="{random_comment}" -metadata software="{random_software}" -metadata artist="{random_artist}" -metadata album="{random_album}" -metadata genre="{random_genre}" -metadata encoder="{random_encoder}" -metadata uid="{random_uuid}" -c copy -y "{output_path}"'

    @staticmethod
    def advanced_metadata_spoofing(input_path: str, output_path: str) -> str:
        """Advanced Metadata Spoofing - creates completely fake metadata profile"""
        import secrets

        # Generate fake camera/device metadata
        fake_cameras = [
            "iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra", "Google Pixel 8 Pro",
            "Canon EOS R5", "Sony FX3", "RED Komodo 6K", "ARRI Alexa Mini",
            "DJI Pocket 2", "GoPro Hero 12", "Panasonic GH6", "Blackmagic URSA"
        ]
        fake_camera = random.choice(fake_cameras)

        # Random technical metadata
        fake_bitrate = random.randint(8000, 50000)
        fake_framerate = random.choice(["23.976", "24", "25", "29.97", "30", "50", "59.94", "60"])
        fake_resolution = random.choice(["1920x1080", "2560x1440", "3840x2160", "4096x2160"])

        # Random location metadata (fake GPS)
        fake_lat = random.uniform(-90, 90)
        fake_lon = random.uniform(-180, 180)

        random_date = (dt.datetime.now() - dt.timedelta(days=random.randint(1, 180))).strftime('%Y-%m-%d %H:%M:%S')

        return f'ffmpeg -i "{input_path}" -map_metadata -1 -metadata creation_time="{random_date}" -metadata title="Recording_{secrets.token_hex(6)}" -metadata make="{fake_camera}" -metadata model="Pro" -metadata software="Camera_App_v{random.randint(1,9)}.{random.randint(0,9)}" -metadata gps_latitude="{fake_lat}" -metadata gps_longitude="{fake_lon}" -metadata bitrate="{fake_bitrate}" -metadata framerate="{fake_framerate}" -c copy -y "{output_path}"'

    @staticmethod
    def codec_metadata_randomization(input_path: str, output_path: str) -> str:
        """Codec Metadata Randomization - randomizes encoding-related metadata"""
        import secrets

        # Random codec metadata
        fake_codecs = [
            "H.264/MPEG-4 AVC", "H.265/HEVC", "VP9", "AV1", 
            "ProRes 422", "DNxHD", "MJPEG", "FFV1"
        ]
        fake_codec = random.choice(fake_codecs)
        fake_profile = random.choice(["High", "Main", "Baseline", "Extended"])
        fake_level = random.choice(["3.1", "4.0", "4.1", "5.0", "5.1", "5.2"])

        # Random audio codec metadata  
        audio_codecs = ["AAC", "MP3", "FLAC", "PCM", "Opus", "Vorbis"]
        fake_audio_codec = random.choice(audio_codecs)
        fake_audio_bitrate = random.choice(["128k", "192k", "256k", "320k", "384k"])

        # Random container metadata
        containers = ["MP4", "MOV", "AVI", "MKV", "WebM", "FLV"]
        fake_container = random.choice(containers)

        random_date = (dt.datetime.now() - dt.timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d %H:%M:%S')

        return f'ffmpeg -i "{input_path}" -map_metadata -1 -metadata creation_time="{random_date}" -metadata video_codec="{fake_codec}" -metadata video_profile="{fake_profile}" -metadata video_level="{fake_level}" -metadata audio_codec="{fake_audio_codec}" -metadata audio_bitrate="{fake_audio_bitrate}" -metadata container="{fake_container}" -metadata encoder_version="v{random.randint(1,20)}.{random.randint(0,99)}" -c copy -y "{output_path}"'

    @staticmethod
    def timestamp_metadata_fuzzing(input_path: str, output_path: str) -> str:
        """Timestamp Metadata Fuzzing - creates multiple conflicting timestamps"""
        import secrets

        # Create multiple different timestamps to confuse detection
        base_time = dt.datetime.now() - dt.timedelta(days=random.randint(1, 200))

        creation_time = base_time.strftime('%Y-%m-%d %H:%M:%S')
        modification_time = (base_time + dt.timedelta(hours=random.randint(1, 72))).strftime('%Y-%m-%d %H:%M:%S')
        access_time = (base_time + dt.timedelta(minutes=random.randint(1, 1440))).strftime('%Y-%m-%d %H:%M:%S')

        # Random timezone offset
        timezone_offsets = ["+00:00", "+01:00", "+02:00", "+05:30", "+08:00", "+09:00", "-05:00", "-08:00", "-03:00"]
        tz_offset = random.choice(timezone_offsets)

        return f'ffmpeg -i "{input_path}" -map_metadata -1 -metadata creation_time="{creation_time}{tz_offset}" -metadata modification_time="{modification_time}{tz_offset}" -metadata access_time="{access_time}{tz_offset}" -metadata time_zone="{tz_offset}" -metadata duration_override="{random.randint(10,3600)}" -c copy -y "{output_path}"'

    @staticmethod
    def uuid_metadata_injection(input_path: str, output_path: str) -> str:
        """UUID Metadata Injection - injects multiple unique identifiers"""
        import secrets
        import uuid

        # Generate multiple UUIDs for different purposes
        file_uuid = str(uuid.uuid4())
        session_uuid = str(uuid.uuid4())
        device_uuid = str(uuid.uuid4())
        app_uuid = str(uuid.uuid4())

        # Random hex identifiers
        random_hex1 = secrets.token_hex(16)
        random_hex2 = secrets.token_hex(8)
        random_hex3 = secrets.token_hex(12)

        random_date = (dt.datetime.now() - dt.timedelta(days=random.randint(1, 120))).strftime('%Y-%m-%d %H:%M:%S')

        return f'ffmpeg -i "{input_path}" -map_metadata -1 -metadata creation_time="{random_date}" -metadata file_id="{file_uuid}" -metadata session_id="{session_uuid}" -metadata device_id="{device_uuid}" -metadata app_id="{app_uuid}" -metadata checksum="{random_hex1}" -metadata hash="{random_hex2}" -metadata signature="{random_hex3}" -metadata version="{random.randint(1000,9999)}" -c copy -y "{output_path}"'

        # 7. SEMANTIC NOISE (40% prob)
    @staticmethod
    def animated_text_corner_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Animated Text Corner - semantic disruption"""
        texts = ['◊ ORIGINAL', '▫ HD', '○ PREMIUM', '△ EXCLUSIVE']
        text = random.choice(texts)
        opacity = random.uniform(0.03, 0.06)

        # Animated position
        positions = [
            'x=10+5*sin(t):y=10+3*cos(t)',
            'x=w-tw-10-5*sin(t):y=10+3*cos(t)',
            'x=10+5*sin(t):y=h-th-10-3*cos(t)',
            'x=w-tw-10-5*sin(t):y=h-th-10-3*cos(t)'
        ]
        position = random.choice(positions)

        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{text}\':fontcolor=white@{opacity}:fontsize=14:{position}" -c:a copy -y "{output_path}"'

    @staticmethod
    def low_opacity_watermark_enhanced(input_path: str, output_path: str) -> str:
        """Enhanced Low Opacity Watermark - semantic confusion"""
        watermarks = ['⬟ CONTENT', '◈ VIDEO', '▢ MEDIA', '⊡ ORIGINAL']
        watermark = random.choice(watermarks)
        opacity = random.uniform(0.02, 0.05)

        positions = [
            'x=(w-tw)/2:y=20',
            'x=(w-tw)/2:y=h-th-20',
            'x=20:y=(h-th)/2',
            'x=w-tw-20:y=(h-th)/2'
        ]
        position = random.choice(positions)

        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{watermark}\':fontcolor=gray@{opacity}:fontsize=12:{position}" -c:a copy -y "{output_path}"'

        # ORB FEATURE-BREAKING TRANSFORMATIONS - Advanced ORB/SIFT/SURF Detection Bypass
    @staticmethod
    def micro_perspective_warp(input_path: str, output_path: str) -> str:
        """Micro perspective warp with multiple random transformation methods"""

        # Randomly choose transformation method
        method = random.choice(['rotation', 'shear', 'crop_offset', 'skew'])

        if method == 'rotation':
            # Tiny rotation - disrupts keypoint geometry
            angle = random.uniform(-0.8, 0.8)
            return f'ffmpeg -i "{input_path}" -vf "rotate={angle}*PI/180:fillcolor=black" -c:a copy -y "{output_path}"'

        elif method == 'shear':
            # Simple shear transformation using perspective - Fixed to prevent black screen
            shear_amount = random.randint(1, 2)  # Reduced from 1-3
            direction = random.choice([-1, 1])
            offset = shear_amount * direction

            # Safe 4-point perspective that keeps image visible
            coords = f"0:0:W+{offset}:0:0:H:W+{offset}:H"
            return f'ffmpeg -i "{input_path}" -vf "perspective={coords}" -c:a copy -y "{output_path}"'

        elif method == 'crop_offset':
            # Micro crop with random offset - simplest but effective
            offset_x = random.randint(1, 3)
            offset_y = random.randint(1, 3)
            crop_size = random.randint(2, 4)

            return f'ffmpeg -i "{input_path}" -vf "crop=iw-{crop_size}:ih-{crop_size}:{offset_x}:{offset_y}" -c:a copy -y "{output_path}"'

        else:  # skew
            # Skew transformation using different corner offsets - Fixed to prevent black screen
            corner_shift = random.randint(1, 2)  # Reduced from 2-4
            corner_choice = random.choice(['top', 'bottom'])  # Only use stable transformations

            if corner_choice == 'top':
                # Slight top shear
                coords = f"{corner_shift}:0:W-{corner_shift}:0:0:H:W:H"
            else:  # bottom
                # Slight bottom shear
                coords = f"0:0:W:0:{corner_shift}:H:W-{corner_shift}:H"

            return f'ffmpeg -i "{input_path}" -vf "perspective={coords}" -c:a copy -y "{output_path}"'

    @staticmethod
    def frame_jittering(input_path: str, output_path: str) -> str:
        """2. Frame Jittering: Shift 1-2px in X/Y every 3-5 frames - invisible but hits ORB hard"""
        # Create random micro-shifts that are imperceptible but break keypoint tracking
        shift_x = random.uniform(1, 2)
        shift_y = random.uniform(1, 2) 
        frame_interval = random.randint(3, 5)

        # Apply subtle crop/pad jitter - moves the frame content by tiny amounts
        jitter_filter = f"crop=iw-{shift_x*2}:ih-{shift_y*2}:{shift_x}:{shift_y},pad=iw+{shift_x*2}:ih+{shift_y*2}:{shift_x}:{shift_y}"

        return f'ffmpeg -i "{input_path}" -vf "{jitter_filter}" -c:a copy -y "{output_path}"'

    @staticmethod
    def fine_noise_overlay(input_path: str, output_path: str) -> str:
        """3. Fine Noise Overlay: Gaussian noise σ=1.0-3.0 - scrambles ORB descriptors"""
        noise_strength = random.uniform(0.005, 0.015)  # Very subtle noise
        noise_type = random.choice(['all', 'c0', 'c1', 'c2'])  # Different color channels

        return f'ffmpeg -i "{input_path}" -vf "noise=alls={noise_strength}:allf=t+u" -c:a copy -y "{output_path}"'

    @staticmethod
    def subtle_logo_texture_overlay(input_path: str, output_path: str) -> str:
        """Enhanced Subtle Logo/Texture Overlay: Advanced multi-layer watermarking with dynamic patterns"""
    
        try:
            # Get video properties for intelligent overlay placement
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', input_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                video_info = json.loads(result.stdout)
                video_stream = next(s for s in video_info['streams'] if s['codec_type'] == 'video')
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                duration = float(video_info['format']['duration'])
                fps = eval(video_stream.get('r_frame_rate', '24/1'))
            else:
                width, height, duration, fps = 1920, 1080, 30.0, 24.0
                
        except Exception:
            width, height, duration, fps = 1920, 1080, 30.0, 24.0
        
        # Enhanced symbol collections
        geometric_symbols = ['⊡', '◊', '▫', '○', '△', '▢', '◈', '⬟', '◯', '▽', '◇', '⬢', '⬡', '◐', '◑', '◒', '◓']
        technical_symbols = ['⚡', '⚙', '⚛', '⚔', '⚊', '⚋', '⚌', '⚍', '⚎', '⚏', '⟐', '⟑', '⟒', '⟓', '⟔', '⟕']
        brand_symbols = ['®', '©', '™', '℗', '℠', '⓪', '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨']
        decorative_symbols = ['❋', '❊', '❉', '❈', '❇', '❆', '❅', '❄', '❃', '❂', '❁', '❀', '✿', '✾', '✽', '✼']
        crypto_symbols = ['₿', '₹', '€', '$', '¥', '£', '₽', '₴', '₦', '₡', '₢', '₣', '₤', '₥', '₦', '₧']
        
        # Advanced overlay techniques
        overlay_techniques = [
            'multi_symbol_pattern',
            'animated_corner_sequence',
            'gradient_texture_overlay',
            'fibonacci_spiral_pattern',
            'dynamic_opacity_breathing',
            'corner_constellation',
            'edge_distributed_marks',
            'temporal_symbol_rotation',
            'layered_transparency_stack',
            'mathematical_grid_pattern'
        ]
        
        technique = random.choice(overlay_techniques)
        
        if technique == 'multi_symbol_pattern':
            # Multiple symbols creating a subtle pattern
            symbol_collections = [geometric_symbols, technical_symbols, brand_symbols, decorative_symbols]
            selected_collection = random.choice(symbol_collections)
            
            symbols = random.sample(selected_collection, min(4, len(selected_collection)))
            base_opacity = random.uniform(0.02, 0.06)
            base_fontsize = random.randint(14, 28)
            
            pattern_filters = []
            for i, symbol in enumerate(symbols):
                corner_positions = [
                    (20 + i*5, 20 + i*3),
                    (width - 80 + i*3, 20 + i*5),
                    (20 + i*4, height - 60 + i*2),
                    (width - 60 + i*2, height - 80 + i*4)
                ]
                
                x, y = random.choice(corner_positions)
                opacity_var = base_opacity * random.uniform(0.7, 1.3)
                size_var = base_fontsize + random.randint(-4, 4)
                
                pattern_filters.append(f"drawtext=text='{symbol}':fontcolor=white@{opacity_var:.4f}:fontsize={size_var}:x={x}:y={y}")
            
            filter_chain = ','.join(pattern_filters)
            return f'''ffmpeg -i "{input_path}" -vf "{filter_chain}" -c:a copy -y "{output_path}"'''
        
        elif technique == 'animated_corner_sequence':
            # Animated sequence of symbols appearing/disappearing
            symbol_collection = random.choice([geometric_symbols, technical_symbols, decorative_symbols])
            symbols = random.sample(symbol_collection, min(6, len(symbol_collection)))
            
            opacity = random.uniform(0.03, 0.08)
            fontsize = random.randint(18, 32)
            corner = random.choice(['tl', 'tr', 'bl', 'br'])
            
            corner_positions = {
                'tl': (15, 35),
                'tr': (width - 50, 35),
                'bl': (15, height - 25),
                'br': (width - 50, height - 25)
            }
            
            x, y = corner_positions[corner]
            animation_speed = random.uniform(0.5, 2.0)
            
            # Cycle through symbols over time
            symbol_cycle = '|'.join(symbols)
            return f'''ffmpeg -i "{input_path}" -vf "drawtext=text='{random.choice(symbols)}':fontcolor=white@{opacity}:fontsize={fontsize}:x={x}:y={y}:enable='between(mod(t,{len(symbols)}),{0},{1})'" -c:a copy -y "{output_path}"'''
        
        elif technique == 'gradient_texture_overlay':
            # Subtle gradient overlay with embedded symbols
            symbol = random.choice(geometric_symbols + technical_symbols)
            gradient_opacity = random.uniform(0.01, 0.04)
            symbol_opacity = random.uniform(0.04, 0.09)
            fontsize = random.randint(20, 40)
            
            # Create gradient in corner
            corner = random.choice(['top', 'bottom', 'left', 'right'])
            gradient_configs = {
                'top': f"drawbox=x=0:y=0:w=w:h=h/6:color=black@{gradient_opacity}:t=fill",
                'bottom': f"drawbox=x=0:y=5*h/6:w=w:h=h/6:color=black@{gradient_opacity}:t=fill",
                'left': f"drawbox=x=0:y=0:w=w/6:h=h:color=black@{gradient_opacity}:t=fill",
                'right': f"drawbox=x=5*w/6:y=0:w=w/6:h=h:color=black@{gradient_opacity}:t=fill"
            }
            
            symbol_positions = {
                'top': f"x=(w-tw)/2:y=30",
                'bottom': f"x=(w-tw)/2:y=h-th-30",
                'left': f"x=30:y=(h-th)/2",
                'right': f"x=w-tw-30:y=(h-th)/2"
            }
            
            return f'''ffmpeg -i "{input_path}" -vf "{gradient_configs[corner]},drawtext=text='{symbol}':fontcolor=white@{symbol_opacity}:fontsize={fontsize}:{symbol_positions[corner]}" -c:a copy -y "{output_path}"'''
        
        elif technique == 'fibonacci_spiral_pattern':
            # Symbols placed along fibonacci spiral points
            symbols = random.sample(geometric_symbols, min(5, len(geometric_symbols)))
            base_opacity = random.uniform(0.025, 0.055)
            base_fontsize = random.randint(12, 22)
            
            # Generate fibonacci spiral points
            center_x, center_y = width // 4, height // 4  # Corner-biased center
            spiral_filters = []
            
            for i, symbol in enumerate(symbols):
                # Fibonacci spiral calculation
                angle = i * 2.39996  # Golden angle in radians
                radius = 20 + i * 15
                
                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))
                
                # Ensure within bounds
                x = max(10, min(width - 50, x))
                y = max(25, min(height - 25, y))
                
                opacity_var = base_opacity * (1 + 0.2 * i)
                size_var = base_fontsize + i * 2
                
                spiral_filters.append(f"drawtext=text='{symbol}':fontcolor=white@{opacity_var:.4f}:fontsize={size_var}:x={x}:y={y}")
            
            return f'''ffmpeg -i "{input_path}" -vf "{','.join(spiral_filters)}" -c:a copy -y "{output_path}"'''
        
        elif technique == 'dynamic_opacity_breathing':
            # Breathing opacity effect with multiple symbols
            symbols = random.sample(decorative_symbols, min(3, len(decorative_symbols)))
            base_opacity = random.uniform(0.02, 0.05)
            fontsize = random.randint(16, 30)
            breathing_speed = random.uniform(0.3, 1.5)
            
            breathing_filters = []
            for i, symbol in enumerate(symbols):
                corner_positions = [
                    (15, 25), (width - 40, 25), (15, height - 25), (width - 40, height - 25)
                ]
                x, y = corner_positions[i % len(corner_positions)]
                
                # Phase offset for each symbol
                phase_offset = i * math.pi / 2
                opacity_formula = f"{base_opacity}+{base_opacity*0.5}*sin(2*PI*{breathing_speed}*t+{phase_offset})"
                
                breathing_filters.append(f"drawtext=text='{symbol}':fontcolor=white@'{opacity_formula}':fontsize={fontsize}:x={x}:y={y}")
            
            return f'''ffmpeg -i "{input_path}" -vf "{','.join(breathing_filters)}" -c:a copy -y "{output_path}"'''
        
        elif technique == 'corner_constellation':
            # Constellation-like pattern in one corner
            symbols = random.sample(geometric_symbols + brand_symbols, min(8, len(geometric_symbols + brand_symbols)))
            corner = random.choice(['tl', 'tr', 'bl', 'br'])
            base_opacity = random.uniform(0.02, 0.06)
            
            corner_bounds = {
                'tl': (10, 80, 10, 120),     # x_min, x_max, y_min, y_max
                'tr': (width-80, width-10, 10, 120),
                'bl': (10, 80, height-120, height-10),
                'br': (width-80, width-10, height-120, height-10)
            }
            
            x_min, x_max, y_min, y_max = corner_bounds[corner]
            constellation_filters = []
            
            for i, symbol in enumerate(symbols):
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                opacity = base_opacity * random.uniform(0.5, 1.5)
                fontsize = random.randint(10, 20)
                
                constellation_filters.append(f"drawtext=text='{symbol}':fontcolor=white@{opacity:.4f}:fontsize={fontsize}:x={x}:y={y}")
            
            return f'''ffmpeg -i "{input_path}" -vf "{','.join(constellation_filters)}" -c:a copy -y "{output_path}"'''
        
        elif technique == 'edge_distributed_marks':
            # Symbols distributed along edges
            symbols = random.sample(technical_symbols, min(6, len(technical_symbols)))
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            base_opacity = random.uniform(0.03, 0.07)
            fontsize = random.randint(14, 24)
            
            edge_filters = []
            for i, symbol in enumerate(symbols):
                if edge == 'top':
                    x = 50 + i * (width - 100) // len(symbols)
                    y = 25
                elif edge == 'bottom':
                    x = 50 + i * (width - 100) // len(symbols)
                    y = height - 25
                elif edge == 'left':
                    x = 25
                    y = 50 + i * (height - 100) // len(symbols)
                else:  # right
                    x = width - 25
                    y = 50 + i * (height - 100) // len(symbols)
                
                opacity = base_opacity * random.uniform(0.8, 1.2)
                edge_filters.append(f"drawtext=text='{symbol}':fontcolor=white@{opacity:.4f}:fontsize={fontsize}:x={x}:y={y}")
            
            return f'''ffmpeg -i "{input_path}" -vf "{','.join(edge_filters)}" -c:a copy -y "{output_path}"'''
        
        elif technique == 'temporal_symbol_rotation':
            # Symbols that change over time
            symbol_sets = [geometric_symbols, technical_symbols, decorative_symbols, brand_symbols]
            selected_set = random.choice(symbol_sets)
            symbols = random.sample(selected_set, min(4, len(selected_set)))
            
            opacity = random.uniform(0.03, 0.08)
            fontsize = random.randint(18, 32)
            rotation_speed = random.uniform(1.0, 4.0)  # symbols per second
            
            # Position in corner
            positions = [(20, 30), (width-50, 30), (20, height-30), (width-50, height-30)]
            x, y = random.choice(positions)
            
            # Create time-based symbol selection
            symbol_duration = duration / len(symbols)
            rotation_filters = []
            
            for i, symbol in enumerate(symbols):
                start_time = i * symbol_duration
                end_time = (i + 1) * symbol_duration
                rotation_filters.append(f"drawtext=text='{symbol}':fontcolor=white@{opacity}:fontsize={fontsize}:x={x}:y={y}:enable='between(t,{start_time},{end_time})'")
            
            return f'''ffmpeg -i "{input_path}" -vf "{','.join(rotation_filters)}" -c:a copy -y "{output_path}"'''
        
        elif technique == 'layered_transparency_stack':
            # Multiple overlapping transparent layers
            symbols = random.sample(crypto_symbols + geometric_symbols, min(5, len(crypto_symbols + geometric_symbols)))
            base_opacity = random.uniform(0.015, 0.04)
            base_fontsize = random.randint(20, 45)
            
            # Stack symbols with slight offsets
            center_x = random.choice([width//6, 5*width//6])  # Left or right side
            center_y = random.choice([height//6, 5*height//6])  # Top or bottom side
            
            stack_filters = []
            for i, symbol in enumerate(symbols):
                offset_x = center_x + i * 3
                offset_y = center_y + i * 2
                layer_opacity = base_opacity * (1 - i * 0.1)  # Decreasing opacity
                layer_size = base_fontsize - i * 3  # Decreasing size
                
                stack_filters.append(f"drawtext=text='{symbol}':fontcolor=white@{layer_opacity:.4f}:fontsize={layer_size}:x={offset_x}:y={offset_y}")
            
            return f'''ffmpeg -i "{input_path}" -vf "{','.join(stack_filters)}" -c:a copy -y "{output_path}"'''
        
        elif technique == 'mathematical_grid_pattern':
            # Grid-based mathematical pattern
            symbols = random.sample(geometric_symbols, min(9, len(geometric_symbols)))
            grid_size = 3  # 3x3 grid
            opacity = random.uniform(0.02, 0.05)
            fontsize = random.randint(12, 20)
            
            # Position grid in corner
            corner = random.choice(['tl', 'tr', 'bl', 'br'])
            grid_spacing = 25
            
            start_positions = {
                'tl': (15, 25),
                'tr': (width - 80, 25),
                'bl': (15, height - 80),
                'br': (width - 80, height - 80)
            }
            
            start_x, start_y = start_positions[corner]
            grid_filters = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if i * grid_size + j < len(symbols):
                        symbol = symbols[i * grid_size + j]
                        x = start_x + j * grid_spacing
                        y = start_y + i * grid_spacing
                        symbol_opacity = opacity * random.uniform(0.7, 1.3)
                        
                        grid_filters.append(f"drawtext=text='{symbol}':fontcolor=white@{symbol_opacity:.4f}:fontsize={fontsize}:x={x}:y={y}")
            
            return f'''ffmpeg -i "{input_path}" -vf "{','.join(grid_filters)}" -c:a copy -y "{output_path}"'''
        
        # Enhanced default fallback with advanced positioning
        symbol_collections = [geometric_symbols, technical_symbols, brand_symbols, decorative_symbols, crypto_symbols]
        selected_collection = random.choice(symbol_collections)
        symbol = random.choice(selected_collection)
        
        # Enhanced opacity and sizing
        opacity = random.uniform(0.025, 0.085)  # Slightly higher than original
        fontsize = random.randint(14, 35)  # Wider range
        
        # Advanced positioning with golden ratio
        golden_ratio = 1.618
        advanced_positions = [
            f'x={int(width/golden_ratio)}:y={int(height/golden_ratio/2)}',  # Golden ratio position
            f'x={int(width - width/golden_ratio)}:y={int(height/golden_ratio/2)}',
            f'x={int(width/golden_ratio)}:y={int(height - height/golden_ratio/2)}',
            f'x={int(width - width/golden_ratio)}:y={int(height - height/golden_ratio/2)}',
            'x=15:y=25',  # Traditional corners
            'x=w-tw-15:y=25',
            'x=15:y=h-th-15',
            'x=w-tw-15:y=h-th-15'
        ]
        
        position = random.choice(advanced_positions)
        
        # Enhanced styling options
        styling_options = [
            f"fontcolor=white@{opacity}",
            f"fontcolor=gray@{opacity*1.2}",
            f"fontcolor=lightblue@{opacity*0.9}",
            f"fontcolor=lightgreen@{opacity*1.1}"
        ]
        
        style = random.choice(styling_options)
        
        return f'''ffmpeg -i "{input_path}" -vf "drawtext=text='{symbol}':${style}:fontsize={fontsize}:{position}:box=1:boxcolor=black@{opacity*0.3}:boxborderw=1" -c:a copy -y "{output_path}"'''

    @staticmethod
    def dynamic_zoom_oscillation(input_path: str, output_path: str, method: str = "scale") -> str:
        """
        Dynamic Zoom: 1.01x-1.03x oscillation - FIXED for encoder alignment issues
        Args:
            input_path: Input video file path
            output_path: Output video file path
            method: Zoom method to use
                - "scale": Use scale filter with proper alignment (recommended)
                - "crop": Use crop-based zoom (safer for encoding)
                - "zoompan": Use zoompan filter 
                - "simple": Simple sine wave oscillation
        
        Returns:
            FFmpeg command string
        """
        import logging
        import subprocess
        
        zoom_factor = random.uniform(1.01, 1.03)
        oscillation_period = random.uniform(8, 15)  # Slow oscillation in seconds
        oscillation_amplitude = random.uniform(0.005, 0.015)  # Randomized amplitude
        
        logging.info(f"🔍 Dynamic zoom oscillation: method={method}, base={zoom_factor:.3f}, period={oscillation_period:.1f}s, amplitude={oscillation_amplitude:.3f}")
        
        try:
            if method == "scale":
                # SAFER: Use simple static zoom instead of dynamic expressions
                static_zoom = zoom_factor + (oscillation_amplitude / 2)  # Average zoom
                
                # Use simple scale and crop without time expressions
                scale_expr = f"scale=iw*{static_zoom}:ih*{static_zoom}"
                crop_expr = f"crop=iw/{static_zoom}:ih/{static_zoom}:(iw-iw/{static_zoom})/2:(ih-ih/{static_zoom})/2"
                
                # Combine scale and crop to maintain original dimensions
                zoom_expr = f"{scale_expr},{crop_expr}"
                
                return f'ffmpeg -i "{input_path}" -vf "{zoom_expr}" -c:a copy -y "{output_path}"'
            
            elif method == "crop":
                # SAFER: Use crop-based zoom with static values instead of time expressions
                # This zooms by cropping a smaller area and scaling it back to original size
                static_crop_factor = 1.0 / (zoom_factor + (oscillation_amplitude / 2))
                
                # Calculate crop dimensions that maintain aspect ratio
                crop_w_expr = f"iw*{static_crop_factor}"
                crop_h_expr = f"ih*{static_crop_factor}"
                crop_x_expr = f"(iw-({crop_w_expr}))/2"
                crop_y_expr = f"(ih-({crop_h_expr}))/2"
                
                # Crop then scale back to original size
                zoom_expr = f"crop={crop_w_expr}:{crop_h_expr}:{crop_x_expr}:{crop_y_expr},scale=iw:ih:flags=lanczos"
                
                return f'ffmpeg -i "{input_path}" -vf "{zoom_expr}" -c:a copy -y "{output_path}"'
            
            elif method == "zoompan":
                # Use zoompan filter (designed for zoom effects)
                try:
                    # Get video info
                    result = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-show_entries', 
                        'format=duration', '-of', 'csv=p=0', input_path
                    ], capture_output=True, text=True)
                    
                    duration = 10  # default fallback
                    if result.returncode == 0 and result.stdout.strip():
                        duration = float(result.stdout.strip())
                    
                    # Get frame rate
                    fps_result = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-show_entries', 
                        'stream=r_frame_rate', '-select_streams', 'v:0',
                        '-of', 'csv=p=0', input_path
                    ], capture_output=True, text=True)
                    
                    fps = 30  # default fallback
                    if fps_result.returncode == 0 and fps_result.stdout.strip():
                        fps_str = fps_result.stdout.strip()
                        if '/' in fps_str:
                            num, den = fps_str.split('/')
                            fps = float(num) / float(den)
                        else:
                            fps = float(fps_str)
                    
                    # Use zoompan with static zoom instead of oscillating
                    static_zoom = zoom_factor + (oscillation_amplitude / 2)  # Average zoom
                    zoom_expr = f"zoompan=z={static_zoom}:d=1:s=iw*ih"
                    
                    return f'ffmpeg -i "{input_path}" -vf "{zoom_expr}" -c:a copy -y "{output_path}"'
                    
                except Exception as e:
                    logging.warning(f"Zoompan method failed, falling back to static crop method: {e}")
                    # Fallback to static crop method
                    static_zoom = zoom_factor + (oscillation_amplitude / 2)  # Average zoom
                    crop_factor = 1.0 / static_zoom
                    zoom_expr = f"crop=iw*{crop_factor}:ih*{crop_factor}:(iw-iw*{crop_factor})/2:(ih-ih*{crop_factor})/2,scale=iw:ih"
                    
                    return f'ffmpeg -i "{input_path}" -vf "{zoom_expr}" -c:a copy -y "{output_path}"'
            
            elif method == "simple":
                # Simple method using static crop (safest)
                static_zoom = zoom_factor + (oscillation_amplitude / 2)  # Average zoom
                crop_factor = 1.0 / static_zoom
                
                zoom_expr = f"crop=iw*{crop_factor}:ih*{crop_factor}:(iw-iw*{crop_factor})/2:(ih-ih*{crop_factor})/2,scale=iw:ih:flags=lanczos"
                
                return f'ffmpeg -i "{input_path}" -vf "{zoom_expr}" -c:a copy -y "{output_path}"'
            
            else:
                raise ValueError(f"Unknown method: {method}. Use: scale, crop, zoompan, or simple")
        
        except Exception as e:
            # Ultimate fallback - very conservative static zoom
            logging.warning(f"Dynamic zoom method '{method}' failed, using safe static fallback: {e}")
            
            # Conservative static zoom (no time expressions)
            safe_zoom = 1.02
            crop_factor = 1.0 / safe_zoom
            
            # Simple crop that maintains original output dimensions
            fallback_expr = f"crop=iw*{crop_factor}:ih*{crop_factor}:(iw-iw*{crop_factor})/2:(ih-ih*{crop_factor})/2,scale=iw:ih"
            
            return f'ffmpeg -i "{input_path}" -vf "{fallback_expr}" -c:a copy -y "{output_path}"'
            
            return f'ffmpeg -i "{input_path}" -vf "{fallback_expr}" -c:a copy -y "{output_path}"'

    @staticmethod
    def dynamic_rotation_subtle(input_path: str, output_path: str) -> str:
        """6. Dynamic Rotation: ±1.0-1.5° every 2-3s - alters keypoint structure"""
        rotation_amplitude = random.uniform(1.0, 1.5)
        rotation_period = random.uniform(2, 3)
        
        # Gentle rotation oscillation that breaks geometric consistency
        rotation_expr = f"rotate={rotation_amplitude}*sin(t/{rotation_period}*2*PI)*PI/180:fillcolor=black"
        
        return f'ffmpeg -i "{input_path}" -vf "{rotation_expr}" -c:a copy -y "{output_path}"'

    @staticmethod
    def slight_color_variation(input_path: str, output_path: str) -> str:
            """7. Slight Color Variation: Saturation ±2-4%, Hue ±1-2° - affects keypoint contrast"""
            saturation_shift = random.uniform(0.96, 1.04)  # ±2-4%
            hue_shift = random.uniform(-2, 2)  # ±1-2°
            
            return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_shift}:s={saturation_shift}" -c:a copy -y "{output_path}"'

    @staticmethod
    def line_sketch_filter_light(input_path: str, output_path: str) -> str:
            """8. Line Sketch Filter: 2-5% opacity edge overlay - misleads ORB with artificial lines"""
            edge_opacity = random.uniform(0.02, 0.05)
            edge_threshold = random.uniform(0.1, 0.3)
            
            # Create subtle edge detection overlay
            edge_filter = f"[0:v]edgedetect=low={edge_threshold}:high={edge_threshold*3}[edges];[0:v][edges]blend=all_mode=overlay:all_opacity={edge_opacity}"
            
            return f'ffmpeg -i "{input_path}" -filter_complex "{edge_filter}" -c:a copy -y "{output_path}"'
        
    @staticmethod
    def randomized_transform_sets(input_path: str, output_path: str) -> str:
        """9. Randomized Transform Sets: Different subset every 5s - avoids ORB pattern alignment"""
        # Combine multiple micro-transformations in a time-varying way
        transform_pool = [
            f"hue=h={random.uniform(-1, 1)}",
            f"eq=saturation={random.uniform(0.98, 1.02)}",
            f"crop=iw-{random.randint(1,2)}:ih-{random.randint(1,2)}:{random.randint(0,1)}:{random.randint(0,1)},scale=iw:ih",
            f"noise=alls={random.uniform(0.001, 0.005)}:allf=t"
        ]

        # Apply different combinations every 5 seconds
        random.shuffle(transform_pool)
        selected_transforms = transform_pool[:2]  # Use 2 random transforms
        filter_chain = ",".join(selected_transforms)

        return f'ffmpeg -i "{input_path}" -vf "{filter_chain}" -c:a copy -y "{output_path}"'

        # STRUCTURAL TRANSFORMATIONS - REDUCED VALUES
    @staticmethod
    def video_length_trim(input_path: str, output_path: str) -> str:
        """Video Length Trim ±0.5-1 sec from start/end (REDUCED from ±1-2 sec)"""
        trim_start = random.uniform(0.3, 1.0)  # REDUCED
        trim_end = random.uniform(0.3, 0.8)  # REDUCED

        return f'ffmpeg -ss {trim_start} -i "{input_path}" -t $(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{input_path}" | awk "{{print \\$1-{trim_start}-{trim_end}}}") -c copy -y "{output_path}"'

    @staticmethod
    def frame_rate_change(input_path: str, output_path: str) -> str:
        """Frame Rate Change ±0.2-0.5 fps (REDUCED from ±0.5-1 fps)"""
        fps_offset = random.uniform(-0.5, 0.5)  # REDUCED
        new_fps = max(24, 30 + fps_offset)

        return f'ffmpeg -i "{input_path}" -r {new_fps} -c:a copy -y "{output_path}"'

    @staticmethod
    def black_frames_transitions(input_path: str, output_path: str) -> str:
        """Black Frames / Transitions - SAFE duration with bounds checking"""
        try:
            # Get video duration safely
            video_info = FFmpegTransformationService.get_video_info_sync(input_path)
            duration = video_info.get('duration', 10.0)
            
            # For very short videos, use a much safer approach
            if duration < 0.5:
                logging.info(f"Video too short ({duration}s) for fade transitions, using copy")
                return f'ffmpeg -i "{input_path}" -c copy -y "{output_path}"'
            
            # Safe fade parameters with bounds checking
            max_fade_duration = min(0.1, duration * 0.15)  # Max 15% of video or 0.1s
            fade_duration = random.uniform(0.03, max_fade_duration)
            
            # Ensure fade_in_start is within valid bounds (very conservative)
            max_fade_start = max(0, min(0.1, duration * 0.05))  # Max 5% of video start or 0.1s
            fade_in_start = random.uniform(0, max_fade_start)
            
            # Calculate fade_out_start ensuring it's valid and positive
            fade_out_start = max(fade_duration * 2, duration - fade_duration)
            
            # Double-check all values are positive and reasonable
            fade_in_start = max(0, fade_in_start)
            fade_out_start = max(fade_duration, fade_out_start)
            fade_duration = max(0.03, min(fade_duration, duration * 0.3))
            
            logging.info(f"🎬 Fade transitions: in={fade_in_start:.3f}s, out={fade_out_start:.3f}s, duration={fade_duration:.3f}s")
            
            return f'ffmpeg -i "{input_path}" -vf "fade=in:st={fade_in_start:.3f}:d={fade_duration:.3f},fade=out:st={fade_out_start:.3f}:d={fade_duration:.3f}" -c:a copy -y "{output_path}"'
            
        except Exception as e:
            logging.warning(f"Fade transitions failed, using copy: {e}")
            return f'ffmpeg -i "{input_path}" -c copy -y "{output_path}"'

        # METADATA TRANSFORMATIONS - Keep same logic but reduce randomness
    @staticmethod
    def metadata_strip_randomize(input_path: str, output_path: str) -> str:
        """Strip EXIF/Metadata completely and add moderately randomized metadata"""


        random_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        random_title = f"Video_{secrets.randbelow(99999):05d}"  # REDUCED digits
        random_comment = f"Content_{uuid.uuid4().hex[:6]}"  # REDUCED from 8 to 6
        random_software = random.choice([
            "Adobe Premiere Pro 2024.1", "Final Cut Pro 10.6.5", "DaVinci Resolve 18.1",
            "OpenShot 3.1.1", "Kdenlive 22.12", "Filmora 12.3", "VSDC 8.1"
        ])
        random_artist = f"Creator_{secrets.randbelow(999):03d}"  # REDUCED digits
        random_album = f"Collection_{secrets.randbelow(99):02d}"  # REDUCED digits

        # REDUCED time range
        days_back = random.randint(1, 90)  # REDUCED from 180 days to 90 days
        random_date = (dt.datetime.now() - dt.timedelta(days=days_back)).strftime('%Y-%m-%d %H:%M:%S')

        return f'ffmpeg -i "{input_path}" -map_metadata -1 -metadata creation_time="{random_date}" -metadata title="{random_title}" -metadata comment="{random_comment}" -metadata software="{random_software}" -metadata artist="{random_artist}" -metadata album="{random_album}" -c copy -y "{output_path}"'

    @staticmethod
    def title_caption_randomize(input_path: str, output_path: str) -> str:
        """Moderately unique title/caption overlay"""
        captions = [
            "EXCLUSIVE CONTENT", "PREMIUM QUALITY", "ORIGINAL VIDEO",
            "RARE FOOTAGE", "VIRAL CONTENT", "TRENDING NOW",
            "MUST WATCH", "TOP QUALITY", "AMAZING SHOW"
        ]
        caption = random.choice(captions)
        fontsize = random.randint(16, 24)  # REDUCED from 20-32
        opacity = random.uniform(0.5, 0.7)  # REDUCED from 0.7-0.9
        duration = random.uniform(1.5, 3)  # REDUCED from 2-5

        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{caption}\':fontcolor=yellow@{opacity}:fontsize={fontsize}:x=(w-tw)/2:y=50:enable=\'between(t,1,{duration})\'" -c:a copy -y "{output_path}"'

        # SEMANTIC/DEEP TRANSFORMATIONS - REDUCED VALUES
    @staticmethod
    def clip_embedding_shift(input_path: str, output_path: str) -> str:
        """CLIP Embedding Shift with REDUCED text and visual elements"""
        emojis = ['🎵', '🎬', '⭐', '🔥', '💎', '🚀', '💯', '🌟']
        emoji = random.choice(emojis)
        text_overlays = [
            f"{emoji} ORIGINAL CONTENT {emoji}",
            f"{emoji} HD QUALITY {emoji}",
            f"{emoji} EXCLUSIVE VIDEO {emoji}"
        ]
        text = random.choice(text_overlays)

        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{text}\':fontcolor=white@0.6:fontsize=20:x=(w-tw)/2:y=h-th-20:box=1:boxcolor=black@0.3" -c:a copy -y "{output_path}"'

    @staticmethod
    def text_presence_variation(input_path: str, output_path: str) -> str:
        """Text Presence in Frame - REDUCED visibility"""
        fonts = ['Arial', 'Helvetica', 'Times', 'Courier']
        font = random.choice(fonts) if random.choice(fonts) else 'sans-serif'
        positions = [
            'x=10:y=10',  # Top-left
            'x=w-tw-10:y=10',  # Top-right
            'x=10:y=h-th-10',  # Bottom-left
            'x=w-tw-10:y=h-th-10',  # Bottom-right
            'x=(w-tw)/2:y=10',  # Top-center
            'x=(w-tw)/2:y=h-th-10'  # Bottom-center
        ]
        position = random.choice(positions)
        watermark_text = f"© {datetime.now().year} ORIGINAL"

        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{watermark_text}\':fontcolor=white@0.4:fontsize=12:{position}" -c:a copy -y "{output_path}"'

    @staticmethod
    def audio_video_sync_offset(input_path: str, output_path: str) -> str:
        """Audio-Video Sync Offset ±50-150ms (REDUCED from ±100-300ms)"""
        offset = random.uniform(-0.15, 0.15)  # REDUCED

        return f'ffmpeg -i "{input_path}" -itsoffset {offset} -i "{input_path}" -map 0:v -map 1:a -c:v copy -c:a aac -shortest -y "{output_path}"'

        # ADVANCED COMBINED TRANSFORMATIONS - REDUCED VALUES
    @staticmethod
    def micro_rotation_with_crop(input_path: str, output_path: str) -> str:
        """Enhanced micro rotation ±2-3° with REDUCED cropping for SSIM reduction"""
        angle = random.uniform(-2, 2)  # REDUCED from ±4
        crop_factor = random.uniform(1.04, 1.08)  # REDUCED from 1.08-1.15

        # REDUCED zoom/pan
        zoom_factor = random.uniform(1.01, 1.04)  # REDUCED from 1.02-1.08
        pan_x = random.uniform(-0.02, 0.02)  # REDUCED from ±0.05
        pan_y = random.uniform(-0.02, 0.02)  # REDUCED from ±0.05

        # REDUCED noise
        noise_strength = random.uniform(0.005, 0.015)  # REDUCED from 0.01-0.03

        # Ensure safe values
        pan_x = max(-0.02, min(0.02, pan_x))
        pan_y = max(-0.02, min(0.02, pan_y))

        return f'ffmpeg -i "{input_path}" -vf "rotate={angle}*PI/180:fillcolor=black:bilinear=1,crop=iw/{crop_factor:.3f}:ih/{crop_factor:.3f},scale=iw*{zoom_factor:.3f}:ih*{zoom_factor:.3f},crop=iw:ih:iw*{pan_x + 0.5:.3f}:ih*{pan_y + 0.5:.3f},noise=alls={noise_strength:.3f}:allf=t" -c:a copy -y "{output_path}"'

    @staticmethod
    def advanced_lut_filter(input_path: str, output_path: str) -> str:
        """Advanced LUT-style color grading (More subtle)"""
        gamma = random.uniform(0.95, 1.05)  # REDUCED from 0.9-1.1
        contrast = random.uniform(0.95, 1.05)  # REDUCED from 0.9-1.1
        brightness = random.uniform(-0.04, 0.04)  # REDUCED from ±0.08
        saturation = random.uniform(0.95, 1.05)  # REDUCED from 0.9-1.1

        return f'ffmpeg -i "{input_path}" -vf "eq=gamma={gamma}:contrast={contrast}:brightness={brightness}:saturation={saturation}" -c:a copy -y "{output_path}"'

    @staticmethod
    def vignette_with_blur(input_path: str, output_path: str) -> str:
        """Vignette effect with subtle edge blur (REDUCED)"""
        vignette_strength = random.uniform(0.2, 0.4)  # REDUCED from 0.3-0.7
        blur_amount = random.uniform(0.5, 1.5)  # REDUCED from 1-3

        return f'ffmpeg -i "{input_path}" -vf "vignette=angle=PI/4:eval=init:dither=1:aspect=1,unsharp=5:5:{blur_amount}:5:5:{blur_amount}" -c:a copy -y "{output_path}"'

    @staticmethod
    def temporal_shift_advanced(input_path: str, output_path: str) -> str:
        """Advanced temporal shift with REDUCED variable speed"""
        speed_factor = random.uniform(0.98, 1.02)  # REDUCED from 0.95-1.05
        atempo_filter = FFmpegTransformationService.validate_atempo_value(speed_factor)

        return f'ffmpeg -i "{input_path}" -filter_complex "[0:v]setpts={1/speed_factor}*PTS[v];[0:a]{atempo_filter}[a]" -map "[v]" -map "[a]" -y "{output_path}"'

        # NEW ENHANCED TRANSFORMATIONS - REDUCED VALUES
    @staticmethod
    def black_screen_random(input_path: str, output_path: str) -> str:
        """
        Brief brightness reduction for disruption - SAFE VERSION with HEAVY RANDOMIZATION
        Prevents complete video blackout with multiple safety measures and varied effects
        """
        try:
            # Get video duration with error handling
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', input_path
            ], capture_output=True, text=True)

            # Parse duration with multiple fallbacks and randomization
            base_duration = random.uniform(8.0, 12.0)  # Random default duration
            if result.returncode == 0 and result.stdout.strip():
                try:
                    parsed_duration = float(result.stdout.strip())
                    # Use actual duration without random modification for accuracy
                    duration = max(5.0, parsed_duration)
                except (ValueError, TypeError):
                    duration = base_duration
            else:
                duration = base_duration

            # RANDOMIZED effect duration with multiple tiers
            effect_type = random.choice(['quick', 'medium', 'brief'])
            if effect_type == 'quick':
                effect_duration = random.uniform(0.03, 0.08)  # Very quick flash
            elif effect_type == 'medium':
                effect_duration = random.uniform(0.08, 0.15)  # Medium duration
            else:  # brief
                effect_duration = random.uniform(0.05, 0.12)  # Brief flash

            # RANDOMIZED timing calculation with multiple strategies
            timing_strategy = random.choice(['early', 'middle', 'late', 'random'])

            if timing_strategy == 'early':
                # Focus on first third of video
                safe_start_time = max(1.0, duration * random.uniform(0.1, 0.25))
                safe_end_time = duration * random.uniform(0.3, 0.4)
            elif timing_strategy == 'middle':
                # Focus on middle third of video
                safe_start_time = max(1.0, duration * random.uniform(0.3, 0.4))
                safe_end_time = duration * random.uniform(0.6, 0.7)
            elif timing_strategy == 'late':
                # Focus on last third of video
                safe_start_time = max(1.0, duration * random.uniform(0.5, 0.6))
                safe_end_time = duration * random.uniform(0.7, 0.8)
            else:  # random
                # Completely random within safe bounds
                start_percent = random.uniform(0.1, 0.6)
                end_percent = random.uniform(start_percent + 0.2, 0.85)
                safe_start_time = max(1.0, duration * start_percent)
                safe_end_time = duration * end_percent

            # Ensure safe timing bounds
            safe_end_time = min(safe_end_time, duration - effect_duration - 1.0)

            # Fallback for edge cases
            if safe_end_time <= safe_start_time or safe_end_time < 2.0:
                # Conservative fallback
                safe_start_time = max(1.0, min(2.0, duration * 0.2))
                safe_end_time = max(safe_start_time + 2.0, min(duration * 0.7, duration - effect_duration - 1.0))

                # Final absolute fallback
                if safe_end_time <= safe_start_time:
                    safe_start_time = 2.0
                    safe_end_time = max(4.0, duration - effect_duration - 1.0)

            # Generate random insertion time within calculated window
            insert_time = random.uniform(safe_start_time, min(safe_end_time, duration - effect_duration - 0.5))

            # HEAVILY RANDOMIZED brightness reduction with multiple modes
            brightness_mode = random.choice(['subtle', 'moderate', 'noticeable', 'variable'])

            if brightness_mode == 'subtle':
                brightness_reduction = random.uniform(-0.06, -0.03)  # Very subtle
            elif brightness_mode == 'moderate':
                brightness_reduction = random.uniform(-0.09, -0.06)  # Moderate
            elif brightness_mode == 'noticeable':
                brightness_reduction = random.uniform(-0.12, -0.09)  # More noticeable
            else:  # variable
                # Random selection from all ranges
                all_ranges = [(-0.06, -0.03), (-0.09, -0.06), (-0.12, -0.09)]
                selected_range = random.choice(all_ranges)
                brightness_reduction = random.uniform(selected_range[0], selected_range[1])

            # Apply safety cap - prevent complete blackout
            brightness_reduction = max(brightness_reduction, -0.15)

            # RANDOMIZED additional effects (sometimes)
            extra_effects = []
            if random.random() < 0.3:  # 30% chance of additional contrast adjustment
                contrast_adjustment = random.uniform(0.95, 1.05)
                extra_effects.append(f"contrast={contrast_adjustment:.3f}")

            if random.random() < 0.2:  # 20% chance of slight saturation change
                saturation_adjustment = random.uniform(0.98, 1.02)
                extra_effects.append(f"saturation={saturation_adjustment:.3f}")

            # Build video filter with base effect and possible extras
            video_filter = f"eq=brightness={brightness_reduction:.3f}"
            if extra_effects:
                video_filter += ":" + ":".join(extra_effects)

            # Ensure timing is valid
            end_time = min(insert_time + effect_duration, duration - 0.1)
            video_filter += f":enable='between(t,{insert_time:.2f},{end_time:.2f})'"

            # RANDOMIZED FFmpeg parameters
            encoding_options = []
            if random.random() < 0.4:  # 40% chance of specific encoding tweaks
                encoding_options.extend(['-avoid_negative_ts', 'make_zero'])

            if random.random() < 0.3:  # 30% chance of preset specification
                preset = random.choice(['ultrafast', 'fast', 'medium'])
                encoding_options.extend(['-preset', preset])

            # Build final command with proper option ordering
            ffmpeg_cmd = f'ffmpeg -i "{input_path}"'
            if encoding_options:
                # Add encoding options before video filter
                option_pairs = []
                for i in range(0, len(encoding_options), 2):
                    if i + 1 < len(encoding_options):
                        option_pairs.append(f'{encoding_options[i]} {encoding_options[i+1]}')
                    else:
                        option_pairs.append(encoding_options[i])
                ffmpeg_cmd += ' ' + ' '.join(option_pairs)

            ffmpeg_cmd += f' -vf "{video_filter}" -c:a copy -y "{output_path}"'

            return ffmpeg_cmd

        except Exception as e:
            # RANDOMIZED fallback with multiple variations - all safe values
            fallback_variations = [
                {'start': 2.5, 'dur': 0.08, 'bright': -0.06},
                {'start': 3.0, 'dur': 0.10, 'bright': -0.08},
                {'start': 3.5, 'dur': 0.06, 'bright': -0.07}
            ]

            # Add slight randomization to fallback
            selected_fallback = random.choice(fallback_variations)
            start_time = selected_fallback['start'] + random.uniform(-0.3, 0.3)
            duration_effect = selected_fallback['dur'] + random.uniform(-0.02, 0.02)
            brightness_val = selected_fallback['bright'] + random.uniform(-0.01, 0.01)

            # Ensure safe values
            start_time = max(1.0, start_time)
            duration_effect = max(0.05, duration_effect)
            brightness_val = max(-0.12, brightness_val)

            fallback_cmd = (
                f'ffmpeg -i "{input_path}" '
                f'-vf "eq=brightness={brightness_val:.3f}:enable=\'between(t,{start_time:.2f},{start_time + duration_effect:.2f})\'" '
                f'-c:a copy -y "{output_path}"'
            )

            return fallback_cmd
    @staticmethod
    def pitch_shift_semitones(input_path: str, output_path: str) -> str:
        """Pitch shift audio ±0.5 semitone (REDUCED from ±1)"""
        semitones = random.uniform(-0.5, 0.5)  # REDUCED from ±1
        pitch_ratio = 2 ** (semitones / 12)

        rate_factor = pitch_ratio
        tempo_compensation = 1.0 / rate_factor
        return f'ffmpeg -i "{input_path}" -af "asetrate=44100*{rate_factor},atempo={tempo_compensation}" -c:v copy -y "{output_path}"'

    @staticmethod
    def overlay_watermark_dynamic(input_path: str, output_path: str) -> str:
        """Overlay watermark/logo + REDUCED dynamic position"""
        watermarks = ['© ORIGINAL', '★ PREMIUM', '🎬 HD QUALITY', '⚡ EXCLUSIVE', '🔥 VIRAL']
        watermark = random.choice(watermarks)

        # REDUCED dynamic motion
        positions = [
            'x=10+10*sin(t):y=10+5*cos(t)',  # REDUCED circular motion
            'x=w-tw-10-8*sin(t):y=10+5*cos(t)',  # REDUCED circular motion
            'x=10+15*sin(t/2):y=h-th-10',  # REDUCED horizontal wave
            'x=w-tw-10:y=h-th-10-10*sin(t)',  # REDUCED vertical wave
        ]
        position = random.choice(positions)

        opacity = random.uniform(0.3, 0.5)  # REDUCED from 0.4-0.7
        fontsize = random.randint(14, 20)  # REDUCED from 16-24

        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{watermark}\':fontcolor=white@{opacity}:fontsize={fontsize}:{position}:box=1:boxcolor=black@0.2" -c:a copy -y "{output_path}"'

    @staticmethod
    def zoom_jitter_motion(input_path: str, output_path: str) -> str:
        """Random zoom (in/out) + jitter motion - fully randomized with hardcoded calculations"""

        # Generate random zoom factor
        zoom_factor = random.uniform(1.01, 1.04)

        # Base dimensions for 720x1280 video
        base_width = 720
        base_height = 1280

        # Calculate new dimensions (rounded to integers)
        new_width = int(base_width * zoom_factor)
        new_height = int(base_height * zoom_factor)

        # Generate random jitter offsets within the zoom margin
        max_offset_x = new_width - base_width
        max_offset_y = new_height - base_height

        jitter_x = random.randint(0, max_offset_x) if max_offset_x > 0 else 0
        jitter_y = random.randint(0, max_offset_y) if max_offset_y > 0 else 0

        # Build the transform with calculated integer values
        transform = f"scale={new_width}:{new_height},crop={base_width}:{base_height}:{jitter_x}:{jitter_y}"

        return f'ffmpeg -i "{input_path}" -vf "{transform}" -c:a copy -y "{output_path}"'

        # NEW TRANSFORMATIONS FROM CSV - REDUCED AUDIO VALUES
    @staticmethod
    def frequency_band_shifting(input_path: str, output_path: str) -> str:
        """Frequency Band Shifting: REDUCED -250Hz to +250Hz"""
        freq_shift = random.uniform(-250, 250)  # REDUCED from ±500Hz
        logging.info(f"🎵 Applying frequency band shifting: {freq_shift:.1f}Hz")

        return f'ffmpeg -i "{input_path}" -af "afreqshift=shift={freq_shift}" -c:v copy -y "{output_path}"'

    @staticmethod
    def multi_band_eq_randomization(input_path: str, output_path: str) -> str:
        """Multi-band EQ: REDUCED Low/Mid/High Gain: -3 to +3dB"""
        low_gain = random.uniform(-1.5, 1.5)  # REDUCED from ±3dB
        mid_gain = random.uniform(-1.5, 1.5)  # REDUCED from ±3dB
        high_gain = random.uniform(-1.5, 1.5)  # REDUCED from ±3dB
        logging.info(f"🎛️ Multi-band EQ: Low={low_gain:.1f}dB, Mid={mid_gain:.1f}dB, High={high_gain:.1f}dB")

        return f'ffmpeg -i "{input_path}" -af "equalizer=f=100:width_type=h:width=50:g={low_gain},equalizer=f=1000:width_type=h:width=200:g={mid_gain},equalizer=f=8000:width_type=h:width=1000:g={high_gain}" -c:v copy -y "{output_path}"'

    @staticmethod
    def harmonic_distortion_subtle(input_path: str, output_path: str) -> str:
        """Harmonic Distortion: REDUCED 5% to 10% intensity"""
        intensity = random.uniform(0.05, 0.1)  # REDUCED from 0.1-0.2
        logging.info(f"🔊 Harmonic distortion intensity: {intensity:.2f}")

        volume_boost = 1 + intensity * 1.5  # REDUCED boost
        return f'ffmpeg -i "{input_path}" -af "volume={volume_boost},highpass=f=50,lowpass=f=15000" -c:v copy -y "{output_path}"'

    @staticmethod
    def frequency_domain_shift(input_path: str, output_path: str) -> str:
        """Frequency Domain Shift: REDUCED -150Hz to +150Hz"""
        freq_shift = random.uniform(-150, 150)  # REDUCED from ±300Hz
        logging.info(f"📡 Frequency domain shift: {freq_shift:.1f}Hz")

        return f'ffmpeg -i "{input_path}" -af "afreqshift=shift={freq_shift}" -c:v copy -y "{output_path}"'

    @staticmethod
    def stereo_phase_inversion(input_path: str, output_path: str) -> str:
        """Stereo Phase Inversion: REDUCED effect - partial invert"""
        logging.info("🔄 Partial right channel phase inversion")

        # REDUCED phase inversion (partial instead of full)
        return f'ffmpeg -i "{input_path}" -af "pan=stereo|c0=c0|c1=-0.5*c1" -c:v copy -y "{output_path}"'

    @staticmethod
    def stereo_width_manipulation(input_path: str, output_path: str) -> str:
        """Stereo Width: REDUCED 0.7x to 1.3x width"""
        width_factor = random.uniform(0.7, 1.3)  # REDUCED from 0.5-2.0
        logging.info(f"↔️ Stereo width factor: {width_factor:.2f}x")

        return f'ffmpeg -i "{input_path}" -af "extrastereo=m={width_factor}" -c:v copy -y "{output_path}"'

    @staticmethod
    def binaural_processing(input_path: str, output_path: str) -> str:
        """Binaural Processing: REDUCED frequency range"""
        freq = random.uniform(10, 15)  # REDUCED from 8-12Hz
        logging.info(f"🧠 Binaural processing at {freq:.1f}Hz")

        # REDUCED tremolo depth
        return f'ffmpeg -i "{input_path}" -af "tremolo=f={freq}:d=0.05" -c:v copy -y "{output_path}"'

    @staticmethod
    def audio_reverse_segments(input_path: str, output_path: str) -> str:
        """Audio Reverse Segments: REDUCED 0.3s to 1s segment"""
        segment_duration = random.uniform(0.3, 1.0)  # REDUCED from 0.5-2.0

        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', input_path
            ], capture_output=True, text=True)

            duration = 10  # default
            if result.returncode == 0:
                try:
                    duration = float(result.stdout.strip())
                except:
                    pass

            start_time = random.uniform(2, max(3, duration - segment_duration - 2))
            logging.info(f"⏪ Reversing audio segment: {start_time:.1f}s-{start_time+segment_duration:.1f}s")

            return f'ffmpeg -i "{input_path}" -filter_complex "[0:a]asplit=3[a1][a2][a3];[a1]atrim=0:{start_time}[before];[a2]atrim={start_time}:{start_time+segment_duration},areverse[reversed];[a3]atrim={start_time+segment_duration}[after];[before][reversed][after]concat=n=3:v=0:a=1[aout]" -map 0:v -map "[aout]" -c:v copy -y "{output_path}"'
        except:
            return f'ffmpeg -i "{input_path}" -af "areverse" -c:v copy -y "{output_path}"'

    @staticmethod
    def echo_delay_variation(input_path: str, output_path: str) -> str:
        """Echo Delay: REDUCED 25ms to 125ms delay, 10% to 25% decay"""
        delay = random.uniform(0.025, 0.125)  # REDUCED from 0.05-0.25
        decay = random.uniform(0.1, 0.25)  # REDUCED from 0.15-0.4
        logging.info(f"🔊 Echo: delay={delay:.2f}s, decay={decay:.2f}")

        return f'ffmpeg -i "{input_path}" -af "aecho=0.8:0.9:{delay*1000:.0f}:{decay}" -c:v copy -y "{output_path}"'

    @staticmethod
    def audio_chorus_effect(input_path: str, output_path: str) -> str:
        """Audio Chorus: REDUCED 0.5ms to 2ms delay, 0.02 to 0.1 depth"""
        delay = random.uniform(0.5, 2)  # REDUCED from 1-4ms
        depth = random.uniform(0.02, 0.1)  # REDUCED from 0.05-0.2
        logging.info(f"🎼 Chorus: delay={delay:.1f}ms, depth={depth:.2f}")

        # REDUCED chorus parameters
        return f'ffmpeg -i "{input_path}" -af "chorus=0.6:0.8:{delay:.1f}:0.4:{delay*1.2:.1f}:0.2" -c:v copy -y "{output_path}"'

    @staticmethod
    def dynamic_range_compression(input_path: str, output_path: str) -> str:
        """Dynamic Range Compression: REDUCED Threshold -12 to -8dB, Ratio 1.5 to 2.5"""
        threshold = random.uniform(-12, -8)  # REDUCED from -15 to -10dB
        ratio = random.uniform(1.5, 2.5)  # REDUCED from 2-4
        logging.info(f"🗜️ Compression: threshold={threshold:.1f}dB, ratio={ratio:.1f}:1")

        return f'ffmpeg -i "{input_path}" -af "acompressor=threshold={threshold}dB:ratio={ratio}:attack=5:release=50" -c:v copy -y "{output_path}"'

    @staticmethod
    def audio_time_stretching(input_path: str, output_path: str) -> str:
        """Audio Time Stretching: REDUCED Speed multiplier 0.97 to 1.03"""
        speed = random.uniform(0.97, 1.03)  # REDUCED from 0.95-1.05
        logging.info(f"⏱️ Audio time stretch: {speed:.3f}x speed")

        atempo_filter = FFmpegTransformationService.validate_atempo_value(speed)
        return f'ffmpeg -i "{input_path}" -af "{atempo_filter}" -c:v copy -y "{output_path}"'

    @staticmethod
    def voice_pattern_disruption(input_path: str, output_path: str) -> str:
        """REDUCED voice pattern masking"""
        formant_shift = random.uniform(0.97, 1.03)  # REDUCED from 0.95-1.05
        pitch_change = random.uniform(0.99, 1.01)  # REDUCED from 0.98-1.02

        logging.info(f"🗣️ Voice pattern disruption: formant={formant_shift:.3f}, pitch={pitch_change:.3f}")

        rate_factor = pitch_change
        tempo_compensation = 1.0 / rate_factor

        # REDUCED frequency filtering
        freq_shift = (formant_shift - 1.0) * 250  # REDUCED from 500

        return f'ffmpeg -i "{input_path}" -af "asetrate=44100*{rate_factor},atempo={tempo_compensation},afreqshift=shift={freq_shift}" -c:v copy -y "{output_path}"'

    @staticmethod
    def voice_formant_modification(input_path: str, output_path: str) -> str:
        """REDUCED voice formant modification"""
        formant_shift = random.uniform(0.95, 1.05)  # REDUCED from 0.85-1.2
        pitch_shift = random.uniform(0.98, 1.02)  # REDUCED from 0.92-1.08
        logging.info(f"🎤 Voice formant modification: formant={formant_shift:.3f}, pitch={pitch_shift:.3f}")

        rate_factor = pitch_shift
        tempo_compensation = 1.0 / rate_factor

        # REDUCED formant changes
        freq_shift = (formant_shift - 1.0) * 250  # REDUCED from 500

        return f'ffmpeg -i "{input_path}" -af "asetrate=44100*{rate_factor},atempo={tempo_compensation},afreqshift=shift={freq_shift},equalizer=f=1000:width_type=h:width=500:g={(formant_shift-1)*3}" -c:v copy -y "{output_path}"'

    @staticmethod
    def vocal_range_compression(input_path: str, output_path: str) -> str:
        """REDUCED vocal frequency range compression"""
        logging.info("🎛️ Compressing vocal frequency range (reduced)")
        return f'ffmpeg -i "{input_path}" -af "equalizer=f=300:width_type=h:width=200:g=-1.5,equalizer=f=2000:width_type=h:width=800:g=1" -c:v copy -y "{output_path}"'

        # INSTAGRAM-SPECIFIC TRANSFORMATIONS - REDUCED VALUES
    @staticmethod
    def instagram_speed_micro_changes(input_path: str, output_path: str) -> str:
        """Instagram Speed Micro Changes: REDUCED 0.98x to 1.02x speed"""
        speed1 = random.uniform(0.98, 1.02)  # REDUCED from 0.97-1.03
        speed2 = random.uniform(0.98, 1.02)
        interval = random.uniform(8, 12)
        logging.info(f"📱 Instagram speed changes: {speed1:.3f}x then {speed2:.3f}x every {interval:.1f}s")

        atempo_filter = FFmpegTransformationService.validate_atempo_value(speed1)
        return f'ffmpeg -i "{input_path}" -filter_complex "[0:v]setpts={1/speed1}*PTS[v1];[0:a]{atempo_filter}[a1]" -map "[v1]" -map "[a1]" -y "{output_path}"'

    @staticmethod
    def instagram_pitch_shift_segments(input_path: str, output_path: str) -> str:
        """Instagram Pitch Shift: REDUCED ±1% to ±3% pitch"""
        pitch_change = random.uniform(-0.03, 0.03)  # REDUCED from ±0.05
        interval = random.uniform(10, 15)
        pitch_ratio = 1.0 + pitch_change
        logging.info(f"📱 Instagram pitch shift: {pitch_change*100:.1f}% every {interval:.1f}s")

        rate_factor = pitch_ratio
        tempo_compensation = 1.0 / rate_factor
        return f'ffmpeg -i "{input_path}" -af "asetrate=44100*{rate_factor},atempo={tempo_compensation}" -c:v copy -y "{output_path}"'

    @staticmethod
    def variable_frame_interpolation(input_path: str, output_path: str) -> str:
        """Variable Frame Interpolation: REDUCED Speed values [0.95, 0.98, 1.02, 1.05]"""
        speeds = [0.95, 0.98, 1.02, 1.05]  # REDUCED from [0.9, 0.95, 1.05, 1.1]
        speed = random.choice(speeds)
        logging.info(f"🎬 Frame interpolation speed: {speed}x")

        atempo_filter = FFmpegTransformationService.validate_atempo_value(speed)
        return f'ffmpeg -i "{input_path}" -filter_complex "[0:v]setpts={1/speed}*PTS[v];[0:a]{atempo_filter}[a]" -map "[v]" -map "[a]" -y "{output_path}"'

    @staticmethod
    def instagram_rotation_micro(input_path: str, output_path: str) -> str:
        """Instagram Micro Rotation: REDUCED ±0.3° to ±1° every 10-15s"""
        angle = random.uniform(-1, 1)  # REDUCED from ±2°
        interval = random.uniform(10, 15)
        logging.info(f"📱 Instagram micro rotation: {angle:.2f}° every {interval:.1f}s")

        return f'ffmpeg -i "{input_path}" -vf "rotate={angle}*PI/180:fillcolor=black" -c:a copy -y "{output_path}"'

    @staticmethod
    def instagram_crop_resize_cycle(input_path: str, output_path: str) -> str:
        """Instagram Crop-Resize Cycle: REDUCED Crop 1px to 10px then resize"""
        crop_amount = random.randint(1, 10)  # REDUCED from 2-20px
        logging.info(f"📱 Instagram crop-resize: crop {crop_amount}px then resize")

        return f'ffmpeg -i "{input_path}" -vf "crop=iw-{crop_amount*2}:ih-{crop_amount*2}:{crop_amount}:{crop_amount},scale=2*trunc(iw/2):2*trunc(ih/2)" -c:a copy -y "{output_path}"'

    @staticmethod
    def instagram_brightness_pulse(input_path: str, output_path: str) -> str:
        """Instagram Brightness Pulse: REDUCED -0.05 to +0.05"""
        brightness = random.uniform(-0.05, 0.05)  # REDUCED from ±0.1
        logging.info(f"📱 Instagram brightness pulse: {brightness:.3f}")

        return f'ffmpeg -i "{input_path}" -vf "eq=brightness={brightness}" -c:a copy -y "{output_path}"'

    @staticmethod
    def instagram_audio_ducking(input_path: str, output_path: str) -> str:
        """Instagram Audio Ducking: REDUCED sidechain compression effect"""
        logging.info("📱 Instagram audio ducking (reduced sidechain compression)")

        # REDUCED compression parameters
        return f'ffmpeg -i "{input_path}" -af "acompressor=threshold=-20dB:ratio=2.5:attack=5:release=50:makeup=1" -c:v copy -y "{output_path}"'

        # VISUAL EFFECTS TRANSFORMATIONS - REDUCED VALUES

    @staticmethod
    def color_channel_swapping(input_path: str, output_path: str) -> str:
        """Color Channel Swapping: Random RGB channel swaps using most reliable methods"""

        # Use only the most reliable methods for FFmpeg 7.1.1
        method = random.choice(['simple_hue', 'colorchannelmixer', 'eq_color'])

        if method == 'simple_hue':
            # Simple hue shift - most reliable
            hue_shift = random.randint(30, 330)  # Degrees
            saturation = random.uniform(0.8, 1.2)

            logging.info(f"🎨 Hue shift: {hue_shift} degrees")
            return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_shift}:s={saturation}" -c:a copy -y "{output_path}"'

        elif method == 'colorchannelmixer':
            # Use simpler colorchannelmixer options that definitely work
            simple_swaps = [
                "colorchannelmixer=rr=0:rg=1:rb=0:gr=1:gg=0:gb=0:br=0:bg=0:bb=1",  # Simple RG swap
                "colorchannelmixer=rr=1:rg=0:rb=0:gr=0:gg=0:gb=1:br=0:bg=1:bb=0",  # Simple GB swap
                "colorchannelmixer=rr=0:rg=0:rb=1:gr=0:gg=1:gb=0:br=1:bg=0:bb=0",  # Simple RB swap
            ]

            selected_swap = random.choice(simple_swaps)
            logging.info(f"🎨 Color channel swap using colorchannelmixer")
            return f'ffmpeg -i "{input_path}" -vf "{selected_swap}" -c:a copy -y "{output_path}"'

        else:  # eq_color
            # Use eq filter for color adjustments - very reliable
            red_adjust = random.uniform(0.7, 1.3)
            green_adjust = random.uniform(0.7, 1.3)  
            blue_adjust = random.uniform(0.7, 1.3)

            logging.info(f"🎨 Color adjustment using eq filter")
            return f'ffmpeg -i "{input_path}" -vf "eq=gamma_r={red_adjust}:gamma_g={green_adjust}:gamma_b={blue_adjust}" -c:a copy -y "{output_path}"'


    @staticmethod
    def chromatic_aberration_effect(input_path: str, output_path: str) -> str:
        """Chromatic Aberration: SAFE -1px to +1px red/blue channel shift with even dimensions"""
        red_shift = random.randint(-1, 1)  # REDUCED from ±3px
        blue_shift = random.randint(-1, 1)  # REDUCED from ±3px
        logging.info(f"🔴🔵 Chromatic aberration: red={red_shift}px, blue={blue_shift}px")

        # Ensure even dimensions for libx264 compatibility
        return f'ffmpeg -i "{input_path}" -vf "format=rgba,rgbashift=rh={red_shift}:bh={blue_shift},scale=2*trunc(iw/2):2*trunc(ih/2),format=yuv420p" -c:a copy -y "{output_path}"'

    @staticmethod
    def selective_color_isolation(input_path: str, output_path: str) -> str:
        """Selective Color Isolation: REDUCED selective color adjustments"""
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
        target_color = random.choice(colors)
        intensity = random.uniform(0.1, 0.3)  # REDUCED from 0.3-0.7
        logging.info(f"🎯 Selective color isolation: enhance {target_color} at {intensity:.2f}")

        if target_color == 'red':
            return f'ffmpeg -i "{input_path}" -vf "eq=saturation=0.7,lutrgb=r=val:g=val*0.85:b=val*0.85" -c:a copy -y "{output_path}"'
        elif target_color == 'green':
            return f'ffmpeg -i "{input_path}" -vf "eq=saturation=0.7,lutrgb=r=val*0.85:g=val:b=val*0.85" -c:a copy -y "{output_path}"'
        elif target_color == 'blue':
            return f'ffmpeg -i "{input_path}" -vf "eq=saturation=0.7,lutrgb=r=val*0.85:g=val*0.85:b=val" -c:a copy -y "{output_path}"'
        else:
            hue_shift = random.uniform(-5, 5)  # REDUCED from ±30
            return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_shift}:s={intensity}" -c:a copy -y "{output_path}"'

    @staticmethod
    def color_space_conversion(input_path: str, output_path: str) -> str:
        """Robust Color Space Conversion: BT709, BT601, SMPTE240m, BT470bg with even dimensions"""
        choices = [
            ("bt601-6-625", "bt709"),
            ("smpte240m", "bt601-6-625"),
            ("bt470bg", "bt709"),
            ("bt709", "bt601-6-625"),
        ]
        iall, all_ = random.choice(choices)
        logging.info(f"🌈 Color space conversion: {iall} → {all_}")

        # Ensure even dimensions for colorspace filter compatibility
        return f'ffmpeg -i "{input_path}" -vf "scale=2*trunc(iw/2):2*trunc(ih/2),colorspace=iall={iall}:all={all_}" -c:a copy -y "{output_path}"'
          

    @staticmethod
    def perspective_distortion(input_path: str, output_path: str) -> str:
        """Perspective Distortion: SAFE -2px to +2px keystone adjustments with even dimensions"""
        x0 = random.randint(-2, 2)  # Further reduced from ±3px
        y0 = random.randint(-2, 2)  # Further reduced from ±3px
        x1 = random.randint(-2, 2)  # Further reduced from ±3px  
        y1 = random.randint(-2, 2)  # Further reduced from ±3px
        logging.info(f"📐 Perspective distortion: corners ({x0},{y0}) ({x1},{y1})")

        # Use safer perspective coordinates and ensure even dimensions
        return f'ffmpeg -i "{input_path}" -vf "perspective={x0}:{y0}:W+{x1}:{y1}:{x0}:H+{y0}:W+{x1}:H+{y1}:interpolation=linear,scale=2*trunc(iw/2):2*trunc(ih/2)" -c:a copy -y "{output_path}"'

    @staticmethod
    def barrel_distortion(input_path: str, output_path: str) -> str:
        """Barrel Distortion: REDUCED -0.05 to +0.05 lens distortion coefficient"""
        distortion = random.uniform(-0.05, 0.05)  # REDUCED from ±0.1
        logging.info(f"🥽 Barrel distortion: {distortion:.3f}")

        return f'ffmpeg -i "{input_path}" -vf "lenscorrection=k1={distortion}" -c:a copy -y "{output_path}"'

    @staticmethod
    def optical_flow_stabilization(input_path: str, output_path: str) -> str:
        """Optical Flow Stabilization: Using compatible deshake filter"""
        logging.info(f"📹 Applying video stabilization")

        return f'ffmpeg -i "{input_path}" -vf "format=yuv420p,deshake" -c:a copy -y "{output_path}"'

    @staticmethod
    def film_grain_simulation(input_path: str, output_path: str) -> str:
        """Film Grain Simulation: REDUCED Noise intensity 0.05 to 0.15"""
        intensity = random.uniform(0.05, 0.15)  # REDUCED from 0.1-0.3
        logging.info(f"🎞️ Film grain: intensity={intensity:.2f}")

        return f'ffmpeg -i "{input_path}" -vf "noise=alls={intensity}:allf=t+u" -c:a copy -y "{output_path}"'

    @staticmethod
    def texture_blend_overlay(input_path: str, output_path: str, method: str = "abs") -> str:
        """
        Texture Blend Overlay: Safe version with multiple protection methods

        Args:
            input_path: Input video file path
            output_path: Output video file path
            method: Safety method to use
                - "abs": Use absolute values (only additive effects)
                - "clamp": Clamp values to [0,255] range
                - "scale": Scale sine waves to [0,1] range
                - "min_brightness": Guarantee minimum brightness level
                - "original": Original unsafe version (for comparison)

        Returns:
            FFmpeg command string
        """
        opacity = random.uniform(0, 0.05)
        logging.info(f"🖼️ Texture overlay: opacity={opacity:.2f}, method={method}")

        if method == "abs":
            # Absolute values - only additive effects (RECOMMENDED)
            r_formula = f"r(X,Y)+{opacity}*abs(sin(X/10)*cos(Y/10))*255"
            g_formula = f"g(X,Y)+{opacity}*abs(cos(X/8)*sin(Y/12))*255"
            b_formula = f"b(X,Y)+{opacity}*abs(sin(X/15)*cos(Y/8))*255"

        elif method == "clamp":
            # Clamp to valid RGB range [0,255]
            r_formula = f"clip(r(X,Y)+{opacity}*sin(X/10)*cos(Y/10)*255,0,255)"
            g_formula = f"clip(g(X,Y)+{opacity}*cos(X/8)*sin(Y/12)*255,0,255)"
            b_formula = f"clip(b(X,Y)+{opacity}*sin(X/15)*cos(Y/8)*255,0,255)"

        elif method == "scale":
            # Scale sine/cosine from [-1,1] to [0,1]
            r_formula = f"r(X,Y)+{opacity}*(sin(X/10)*cos(Y/10)+1)/2*255"
            g_formula = f"g(X,Y)+{opacity}*(cos(X/8)*sin(Y/12)+1)/2*255"
            b_formula = f"b(X,Y)+{opacity}*(sin(X/15)*cos(Y/8)+1)/2*255"

        elif method == "min_brightness":
            # Guarantee minimum brightness level
            min_brightness = 20
            r_formula = f"max(r(X,Y)+{opacity}*sin(X/10)*cos(Y/10)*255,{min_brightness})"
            g_formula = f"max(g(X,Y)+{opacity}*cos(X/8)*sin(Y/12)*255,{min_brightness})"
            b_formula = f"max(b(X,Y)+{opacity}*sin(X/15)*cos(Y/8)*255,{min_brightness})"

        elif method == "original":
            # Original unsafe version (for comparison/testing)
            r_formula = f"r(X,Y)+{opacity}*sin(X/10)*cos(Y/10)*255"
            g_formula = f"g(X,Y)+{opacity}*cos(X/8)*sin(Y/12)*255"
            b_formula = f"b(X,Y)+{opacity}*sin(X/15)*cos(Y/8)*255"

        else:
            raise ValueError(f"Unknown method: {method}. Use: abs, clamp, scale, min_brightness, or original")

        return f'ffmpeg -i "{input_path}" -vf "geq=r=\'{r_formula}\':g=\'{g_formula}\':b=\'{b_formula}\'" -c:a copy -y "{output_path}"'

    @staticmethod
    def particle_overlay_system(input_path: str, output_path: str) -> str:
        """Particle Overlay: REDUCED Count 5 to 15, opacity 0.005 to 0.02"""
        count = random.randint(5, 15)  # REDUCED from 10-30
        opacity = random.uniform(0.005, 0.02)  # REDUCED from 0.01-0.05
        logging.info(f"✨ Particle overlay: count={count}, opacity={opacity:.3f} (minimal)")

        noise_strength = opacity / 20  # Further reduced noise strength
        return f'ffmpeg -i "{input_path}" -vf "noise=alls={noise_strength:.4f}:allf=t+u" -c:a copy -y "{output_path}"'

        # METADATA AND TECHNICAL TRANSFORMATIONS - REDUCED VALUES
    @staticmethod
    def advanced_metadata_spoofing(input_path: str, output_path: str) -> str:
        """Advanced Metadata Spoofing: REDUCED randomization"""
        cameras = [
            "Canon EOS R5", "Sony A7R IV", "Nikon Z9", "Fujifilm X-T4", 
            "Panasonic GH5", "Blackmagic Pocket 6K", "RED Komodo", "iPhone 14 Pro"
        ]
        software_versions = [
            "Adobe Premiere Pro 2024.1", "Final Cut Pro 10.6.8", "DaVinci Resolve 18.5",
            "Avid Media Composer 2023.6", "OpenShot 3.1.1", "Kdenlive 23.04"
        ]

        camera = random.choice(cameras)
        software = random.choice(software_versions)

        # REDUCED timestamp range
        days_back = random.randint(1, 180)  # REDUCED from 365 days
        random_date = (dt.datetime.now() - dt.timedelta(days=days_back)).strftime('%Y-%m-%d %H:%M:%S')

        logging.info(f"📋 Metadata spoofing: {camera}, {software}, {random_date}")

        return f'ffmpeg -i "{input_path}" -map_metadata -1 -metadata creation_time="{random_date}" -metadata encoder="{software}" -metadata comment="Shot on {camera}" -c copy -y "{output_path}"'

    @staticmethod
    def gps_exif_randomization(input_path: str, output_path: str) -> str:
        """GPS EXIF Randomization: REDUCED -60 to +60 lat, -120 to +120 long"""
        latitude = random.uniform(-60, 60)  # REDUCED from ±90
        longitude = random.uniform(-120, 120)  # REDUCED from ±180
        logging.info(f"🌍 GPS randomization: lat={latitude:.4f}, lon={longitude:.4f}")

        return f'ffmpeg -i "{input_path}" -metadata location="{latitude},{longitude}" -c copy -y "{output_path}"'

    @staticmethod
    def camera_settings_simulation(input_path: str, output_path: str) -> str:
        """Camera Settings: ISO 200-3200, Aperture f/2.0-f/5.6, Shutter 1/60-1/250"""
        iso = random.choice([200, 400, 800, 1600, 3200])  # REDUCED range
        aperture = random.choice([2.0, 2.8, 4.0, 5.6])  # REDUCED range
        shutter = random.choice([60, 125, 250])  # REDUCED range

        logging.info(f"📷 Camera settings: ISO{iso}, f/{aperture}, 1/{shutter}s")

        return f'ffmpeg -i "{input_path}" -metadata:s:v iso="{iso}" -metadata:s:v aperture="f/{aperture}" -metadata:s:v shutter="1/{shutter}" -c copy -y "{output_path}"'

    @staticmethod
    def software_version_cycling(input_path: str, output_path: str) -> str:
        """Software Version Cycling: Cycle editing software versions"""
        versions = [
            "Adobe Premiere Pro 24.0.3", "Final Cut Pro 10.6.9", "DaVinci Resolve 18.6.3",
            "Avid Media Composer 2023.12", "Adobe After Effects 24.1", "Vegas Pro 21.0"
        ]
        version = random.choice(versions)
        logging.info(f"💻 Software version: {version}")

        return f'ffmpeg -i "{input_path}" -metadata encoder="{version}" -c copy -y "{output_path}"'

    @staticmethod
    def codec_parameter_variation(input_path: str, output_path: str) -> str:
        """Codec Parameters: REDUCED CRF 20-24, preset fast-medium"""
        crf = random.randint(20, 24)  # REDUCED from 18-26
        presets = ['fast', 'medium']  # REDUCED options
        preset = random.choice(presets)
        logging.info(f"🎬 Codec params: CRF={crf}, preset={preset}")

        return f'ffmpeg -i "{input_path}" -c:v libx264 -crf {crf} -preset {preset} -c:a copy -y "{output_path}"'

    @staticmethod
    def creation_time_fuzzing(input_path: str, output_path: str) -> str:
        """Creation Time Fuzzing: REDUCED Random +/- 0.5 hour from now"""
        hours_offset = random.uniform(-0.5, 0.5)  # REDUCED from ±1 hour
        fuzzed_time = dt.datetime.now() + dt.timedelta(hours=hours_offset)
        time_str = fuzzed_time.strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"⏰ Time fuzzing: {time_str}")

        return f'ffmpeg -i "{input_path}" -metadata creation_time="{time_str}" -c copy -y "{output_path}"'

    @staticmethod
    def uuid_injection_system(input_path: str, output_path: str) -> str:
        """UUID Injection: Unique v4 UUID"""
        unique_id = str(uuid.uuid4())
        logging.info(f"🔑 UUID injection: {unique_id}")

        return f'ffmpeg -i "{input_path}" -metadata uuid="{unique_id}" -metadata description="ID:{unique_id}" -c copy -y "{output_path}"'

    @staticmethod
    def enhanced_ambient_layering(input_path: str, output_path: str) -> str:
        """Add multiple ambient layers at REDUCED volume"""
        ambient_types = ['anoisesrc=d=60:c=pink:a=0.08', 'sine=frequency=60:duration=60']  # REDUCED from 0.15
        ambient = random.choice(ambient_types)
        logging.info(f"🌊 Enhanced ambient layering with {ambient.split(':')[1].split('=')[1]}")
        return f'ffmpeg -i "{input_path}" -f lavfi -i "{ambient}" -filter_complex "[0:a]volume=0.9[a0];[1:a]volume=0.1[a1];[a0][a1]amix=inputs=2[aout]" -map 0:v -map "[aout]" -c:v copy -y "{output_path}"'

    @staticmethod
    def frame_trimming_dropout(input_path: str, output_path: str) -> str:
        """Frame trimming: REDUCED frame rate adjustment"""
        frame_adjustments = [
            '-r 24.9',  # REDUCED changes
            '-r 25.1',  
            '-r 29.9',  
            '-r 30.1',  
        ]
        adjustment = random.choice(frame_adjustments)

        return f'ffmpeg -i "{input_path}" {adjustment} -c:a copy -y "{output_path}"'

    @staticmethod
    def noise_blur_regions(input_path: str, output_path: str) -> str:
        """Add REDUCED noise filter or soft blur over certain regions"""
        effects = [
            'noise=alls=0.01:allf=t',  # REDUCED from 0.02
            'boxblur=1:1:cr=0:ar=0',   # REDUCED from 2:1
            'noise=alls=0.008:allf=t,unsharp=5:5:0.5:5:5:0.5',  # REDUCED
        ]
        effect = random.choice(effects)

        return f'ffmpeg -i "{input_path}" -vf "{effect}" -c:a copy -y "{output_path}"'

    @staticmethod
    def grayscale_segment(input_path: str, output_path: str) -> str:
        """Brief grayscale segment for REDUCED 0.3–0.6 seconds"""
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', input_path
            ], capture_output=True, text=True)

            duration = 10  # default
            if result.returncode == 0:
                try:
                    duration = float(result.stdout.strip())
                except:
                    pass

            # REDUCED grayscale segment timing
            gray_start = random.uniform(duration * 0.3, duration * 0.7)
            gray_duration = random.uniform(0.3, 0.6)  # REDUCED from 0.5-1.0

            return f'ffmpeg -i "{input_path}" -vf "hue=s=0:enable=\'between(t,{gray_start:.2f},{gray_start + gray_duration:.2f})\'" -c:a copy -y "{output_path}"'

        except Exception:
            return f'ffmpeg -i "{input_path}" -vf "hue=s=0.85" -c:a copy -y "{output_path}"'

    @staticmethod
    def animated_text_corner(input_path: str, output_path: str) -> str:
        """Overlay animated text on bottom corner (REDUCED visibility)"""
        texts = ['HD QUALITY', 'PREMIUM', 'ORIGINAL', 'EXCLUSIVE', '© 2025']
        text = random.choice(texts)

        fontsize = random.randint(12, 16)  # REDUCED from 14-20
        opacity_base = random.uniform(0.2, 0.4)  # REDUCED from 0.4-0.6

        # REDUCED animation
        animation = f'x=w-tw-10+3*sin(t):y=h-th-10+2*cos(t):alpha={opacity_base}*sin(t/2)'

        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{text}\':fontcolor=white:fontsize={fontsize}:{animation}:box=1:boxcolor=black@0.1" -c:a copy -y "{output_path}"'

    @staticmethod
    def moving_watermark_system(input_path: str, output_path: str) -> str:
        """Moving watermarks with REDUCED position and opacity"""
        opacity = random.uniform(0.03, 0.08)  # REDUCED from 0.05-0.15
        logging.info(f"📍 Moving watermark with opacity {opacity:.3f}")
        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'©\':fontcolor=white@{opacity}:fontsize=16:x=mod(t*20\\,w):y=mod(t*15\\,h)" -c:a copy -y "{output_path}"'

    @staticmethod
    def color_pulse_effect(input_path: str, output_path: str) -> str:
        """REDUCED color pulsing throughout video"""
        logging.info("🌈 Applying color pulse effect (reduced)")
        return f'ffmpeg -i "{input_path}" -vf "eq=saturation=1+0.05*sin(2*PI*t/3)" -c:a copy -y "{output_path}"'

    @staticmethod
    def dynamic_timestamp_overlay(input_path: str, output_path: str) -> str:
        """Dynamic timestamp overlay with REDUCED unique info per video"""
        import time
        current_time = int(time.time())
        random_offset = random.randint(-43200, 43200)  # REDUCED to ±12 hours
        timestamp = current_time + random_offset

        timestamp_formats = [
            f"ID\\:{timestamp % 99999:05d}",   # REDUCED digits
            f"T\\:{timestamp % 99999:05d}",   
            f"V\\:{random.randint(10000, 99999)}",  
            f"\\#{random.randint(100, 999)}",       # REDUCED digits
        ]

        timestamp_text = random.choice(timestamp_formats)

        positions = [
            'x=10:y=h-th-10',           
            'x=w-tw-10:y=h-th-10',      
            'x=w-tw-10:y=10',           
            'x=10:y=10'                 
        ]
        position = random.choice(positions)

        fontsize = random.randint(8, 12)  # REDUCED from 10-14
        opacity = random.uniform(0.1, 0.2)  # REDUCED from 0.15-0.3

        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{timestamp_text}\':fontcolor=white@{opacity}:fontsize={fontsize}:{position}" -c:a copy -y "{output_path}"'

    @staticmethod
    def random_frame_inserts(input_path: str, output_path: str) -> str:
        """Insert random brief flashes with REDUCED values"""
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', input_path
            ], capture_output=True, text=True)

            duration = 10  # default
            if result.returncode == 0:
                try:
                    duration = float(result.stdout.strip())
                except:
                    pass

            flash_types = ['brightness', 'contrast', 'saturation']
            flash_type = random.choice(flash_types)

            insert_time = random.uniform(2, max(3, duration - 2))
            flash_duration = 0.05  # REDUCED from 0.08

            if flash_type == 'brightness':
                brightness_val = random.choice([-0.15, 0.15])  # REDUCED from ±0.3
                return f'ffmpeg -i "{input_path}" -vf "eq=brightness={brightness_val}:enable=\'between(t,{insert_time:.3f},{insert_time + flash_duration:.3f})\'" -c:a copy -y "{output_path}"'
            elif flash_type == 'contrast':
                contrast_val = random.choice([0.8, 1.2])  # REDUCED from [0.6, 1.5]
                return f'ffmpeg -i "{input_path}" -vf "eq=contrast={contrast_val}:enable=\'between(t,{insert_time:.3f},{insert_time + flash_duration:.3f})\'" -c:a copy -y "{output_path}"'
            else:
                saturation_val = random.choice([0.5, 1.5])  # REDUCED from [0.3, 1.7]
                return f'ffmpeg -i "{input_path}" -vf "eq=saturation={saturation_val}:enable=\'between(t,{insert_time:.3f},{insert_time + flash_duration:.3f})\'" -c:a copy -y "{output_path}"'

        except Exception:
            fade_time = random.uniform(3, 6)
            fade_duration = 0.05  # REDUCED

            return f'ffmpeg -i "{input_path}" -vf "eq=brightness=0.1:enable=\'between(t,{fade_time:.2f},{fade_time + fade_duration:.2f})\'" -c:a copy -y "{output_path}"'

    @staticmethod
    def clip_embedding_shuffle(input_path: str, output_path: str) -> str:
        """CLIP embedding shuffling with REDUCED emojis and quotes"""
        emoji_collections = [
            ['🔥', '⚡', '💥', '🌟'],  
            ['✨', '💎', '👑', '🏆'],  
            ['🎵', '🎬', '🎪', '🎭'],  
            ['💯', '🚀', '⭐', '🌈'],  
            ['🎯', '💪', '🔝', '⚖️'],  
        ]

        quote_collections = [
            ['VIRAL', 'TRENDING', 'POPULAR', 'HOT'],
            ['PREMIUM', 'EXCLUSIVE', 'RARE', 'LIMITED'],
            ['ORIGINAL', 'AUTHENTIC', 'GENUINE', 'REAL'],
            ['QUALITY', 'HD', 'CRYSTAL', 'CLEAR'],
            ['AMAZING', 'INCREDIBLE', 'STUNNING', 'WOW']
        ]

        emoji_set = random.choice(emoji_collections)
        quote_set = random.choice(quote_collections)

        emoji = random.choice(emoji_set)
        quote = random.choice(quote_set)

        text_variations = [
            f"{emoji} {quote} {emoji}",
            f"{emoji}{emoji} {quote}",
            f"{quote} {emoji}",
            f"{emoji} {quote}",
            f"{quote}{emoji}{emoji}"
        ]

        overlay_text = random.choice(text_variations)

        positions = [
            'x=(w-tw)/2:y=20',           
            'x=(w-tw)/2:y=h-th-20',      
            'x=20:y=(h-th)/2',           
            'x=w-tw-20:y=(h-th)/2',      
        ]
        position = random.choice(positions)

        fontsize = random.randint(14, 22)  # REDUCED from 18-28
        opacity = random.uniform(0.4, 0.6)  # REDUCED from 0.6-0.8
        duration_start = random.uniform(0.5, 2)
        duration_length = random.uniform(1.5, 3)  # REDUCED from 2-4

        return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{overlay_text}\':fontcolor=yellow@{opacity}:fontsize={fontsize}:{position}:enable=\'between(t,{duration_start},{duration_start + duration_length})\':box=1:boxcolor=black@0.3" -c:a copy -y "{output_path}"'

    @staticmethod
    def frame_reordering_segments(input_path: str, output_path: str) -> str:
        """REDUCED micro-temporal disorientation"""
        try:
            # The line `import subprocess` is not required here if it is already imported at the top of the file.
            # You can safely remove it from this location.
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', input_path
            ], capture_output=True, text=True)

            duration = 10  # default
            if result.returncode == 0:
                try:
                    duration = float(result.stdout.strip())
                except:
                    pass

            # REDUCED speed variation
            speed_factor = random.choice([0.98, 0.99, 1.01, 1.02])  # REDUCED from [0.95, 0.98, 1.02, 1.05]
            atempo_filter = FFmpegTransformationService.validate_atempo_value(speed_factor)

            return f'ffmpeg -i "{input_path}" -vf "setpts={1/speed_factor}*PTS" -af "{atempo_filter}" -y "{output_path}"'

        except Exception:
            fps_adjustment = random.choice([0.99, 1.01])  # REDUCED from [0.98, 1.02]
            atempo_filter = FFmpegTransformationService.validate_atempo_value(fps_adjustment)

            return f'ffmpeg -i "{input_path}" -filter_complex "[0:v]setpts={1/fps_adjustment}*PTS[v];[0:a]{atempo_filter}[a]" -map "[v]" -map "[a]" -y "{output_path}"'

    @staticmethod
    def complex_speed_patterns(input_path: str, output_path: str) -> str:
        """REDUCED multiple speed changes"""
        speeds = [0.98, 0.99, 1.01, 1.02]  # REDUCED from [0.97, 0.99, 1.01, 1.03]
        logging.info(f"⚡ Complex speed patterns: {speeds}")
        speed = random.choice(speeds)
        atempo_filter = FFmpegTransformationService.validate_atempo_value(speed)
        return f'ffmpeg -i "{input_path}" -filter_complex "[0:v]setpts={1/speed}*PTS[v];[0:a]{atempo_filter}[a]" -map "[v]" -map "[a]" -y "{output_path}"'

    @staticmethod
    def frame_micro_adjustments(input_path: str, output_path: str) -> str:
        """REDUCED individual frame adjustments"""
        logging.info("🎬 Applying frame micro adjustments (reduced)")
        return f'ffmpeg -i "{input_path}" -vf "select=not(mod(n\\,60)),setpts=N/FRAME_RATE/TB" -c:a copy -y "{output_path}"'

    @staticmethod
    def audio_panning_balance(input_path: str, output_path: str) -> str:
        """Audio panning with REDUCED left-right balance changes"""
        panning_types = [
            # REDUCED cross-panning
            'stereo|c0=0.8*c0+0.2*c1|c1=0.2*c0+0.8*c1',
            # REDUCED left-heavy panning
            'stereo|c0=0.85*c0+0.15*c1|c1=0.15*c0+0.85*c1',
            # REDUCED right-heavy panning
            'stereo|c0=0.85*c0+0.15*c1|c1=0.15*c0+0.85*c1',
            # REDUCED dynamic panning
            'stereo|c0=0.85*c0+0.15*c1|c1=0.15*c0+0.85*c1',
            # REDUCED center spreading
            'stereo|c0=0.95*c0+0.05*c1|c1=0.05*c0+0.95*c1'
        ]

        panning_filter = random.choice(panning_types)

        return f'ffmpeg -i "{input_path}" -af "pan={panning_filter}" -c:v copy -y "{output_path}"'

        # NEW RANDOM TEMPORAL TRANSFORMATIONS - Applied at random points in video
    @staticmethod
    def random_geometric_warp(input_path: str, output_path: str, timing_info: str = None) -> str:
        """Random geometric warps applied at specific timestamps"""
        warp_types = [
            'perspective', 'barrel_distortion', 'pincushion', 'rotation_warp', 'shear'
        ]
        warp_type = random.choice(warp_types)

        # Get video duration for timing
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', input_path
            ], capture_output=True, text=True)

            duration = 10  # default
            if result.returncode == 0:
                try:
                    duration = float(result.stdout.strip())
                except:
                    pass

            # Generate random application points using helper function
            application_points = FFmpegTransformationService.get_random_transformation_points(
                duration, num_points=random.randint(3, 5), min_interval=2.0
            )

            if warp_type == 'perspective':
                # Random perspective warp - Fixed to prevent black screen
                x_offset = random.uniform(-2, 2)  # Reduced from ±5
                y_offset = random.uniform(-2, 2)  # Reduced from ±5
                filter_expr = f"perspective={x_offset}:{y_offset}:W+{x_offset}:{y_offset}:{x_offset}:H+{y_offset}:W+{x_offset}:H+{y_offset}"
            elif warp_type == 'barrel_distortion':
                # Random barrel distortion
                distortion = random.uniform(-0.08, 0.08)
                filter_expr = f"lenscorrection=k1={distortion}"
            elif warp_type == 'rotation_warp':
                # Random rotation with fill
                angle = random.uniform(-3, 3)
                filter_expr = f"rotate={angle}*PI/180:fillcolor=black:bilinear=1"
            elif warp_type == 'shear':
                # Random shear transformation (using perspective filter) - Fixed to prevent black screen
                shear_x = random.uniform(-1, 1)  # Reduced from ±2
                shear_y = random.uniform(-1, 1)  # Reduced from ±2
                filter_expr = f"perspective={shear_x}:0:W+{shear_x}:{shear_y}:0:H:W:H+{shear_y}"
            else:
                # Default pincushion
                distortion = random.uniform(-0.05, 0.05)
                filter_expr = f"lenscorrection=k1={distortion}:k2={distortion*0.5}"

            # Create enable expressions for random timing
            enable_expressions = []
            for start_time, effect_duration in application_points:
                end_time = min(start_time + effect_duration, duration)
                enable_expressions.append(f"between(t,{start_time:.2f},{end_time:.2f})")

            # Combine all enable expressions with OR
            combined_enable = "+".join(enable_expressions)
            final_filter = f"{filter_expr}:enable='{combined_enable}'"

            logging.info(f"🌀 Random geometric warp ({warp_type}) at {len(application_points)} points: {[(round(s,1), round(d,1)) for s,d in application_points]}")
            return f'ffmpeg -i "{input_path}" -vf "{final_filter}" -c:a copy -y "{output_path}"'

        except Exception:
            # Fallback - single application
            warp_time = random.uniform(2, 8)
            warp_duration = random.uniform(1, 3)
            angle = random.uniform(-2, 2)
            logging.info(f"🌀 Simple geometric warp at {warp_time:.1f}s for {warp_duration:.1f}s")
            return f'ffmpeg -i "{input_path}" -vf "rotate={angle}*PI/180:fillcolor=black:enable=\'between(t,{warp_time:.2f},{warp_time + warp_duration:.2f})\'" -c:a copy -y "{output_path}"'

    @staticmethod
    def random_cut_jitter_effects(input_path: str, output_path: str, timing_info: str = None) -> str:
        """Random cut/jitter effects applied using supported filters"""

        jitter_types = ['brightness_jitter', 'speed_jitter', 'contrast_jitter', 'simple_position']
        jitter_type = random.choice(jitter_types)

        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', input_path
            ], capture_output=True, text=True)

            duration = 10  # default
            if result.returncode == 0:
                try:
                    duration = float(result.stdout.strip())
                except:
                    pass

            if jitter_type == 'brightness_jitter':
                # Brightness jitter using eq filter (supports timeline)
                jitter_points = []
                num_points = random.randint(3, 5)
                for i in range(num_points):
                    start_time = random.uniform(1, duration - 2)
                    jitter_duration = random.uniform(0.1, 0.5)
                    jitter_points.append((start_time, jitter_duration))

                brightness_value = random.choice([-0.15, 0.15, -0.2, 0.2])  # Reduced from extreme values
                enable_expressions = []
                for start_time, jitter_duration in jitter_points:
                    end_time = min(start_time + jitter_duration, duration)
                    enable_expressions.append(f"between(t,{start_time:.3f},{end_time:.3f})")

                combined_enable = "+".join(enable_expressions)
                filter_expr = f"eq=brightness={brightness_value}:enable='{combined_enable}'"

            elif jitter_type == 'speed_jitter':
                # Speed variations - simple setpts
                speed_factor = random.choice([0.85, 0.9, 1.1, 1.15, 0.95, 1.05])
                filter_expr = f"setpts={1/speed_factor}*PTS"

            elif jitter_type == 'contrast_jitter':
                # Contrast jitter using eq filter
                jitter_points = []
                num_points = random.randint(2, 4)
                for i in range(num_points):
                    start_time = random.uniform(1, duration - 2)
                    jitter_duration = random.uniform(0.2, 0.6)
                    jitter_points.append((start_time, jitter_duration))

                contrast_value = random.choice([0.7, 1.3, 0.8, 1.2])
                enable_expressions = []
                for start_time, jitter_duration in jitter_points:
                    end_time = min(start_time + jitter_duration, duration)
                    enable_expressions.append(f"between(t,{start_time:.3f},{end_time:.3f})")

                combined_enable = "+".join(enable_expressions)
                filter_expr = f"eq=contrast={contrast_value}:enable='{combined_enable}'"

            else:  # simple_position - use pad without timeline
                # Simple position jitter without timeline - just apply once
                jitter_x = random.randint(-2, 2)
                jitter_y = random.randint(-2, 2)

                # Use pad and crop for position shift (no timeline needed)
                pad_amount = 10
                crop_x = pad_amount + jitter_x
                crop_y = pad_amount + jitter_y

                filter_expr = f"pad=iw+{pad_amount*2}:ih+{pad_amount*2}:{pad_amount}:{pad_amount}:black,crop=iw-{pad_amount*2}:ih-{pad_amount*2}:{crop_x}:{crop_y}"

            logging.info(f"⚡ Random jitter effect: {jitter_type}")
            return f'ffmpeg -i "{input_path}" -vf "{filter_expr}" -c:a copy -y "{output_path}"'

        except Exception as e:
            # Simple fallback - brightness change only
            brightness = random.choice([-0.15, 0.15])  # Reduced from ±0.2
            logging.info(f"⚡ Simple brightness jitter (fallback)")
            return f'ffmpeg -i "{input_path}" -vf "eq=brightness={brightness}" -c:a copy -y "{output_path}"'

    @staticmethod
    def random_overlay_effects(input_path: str, output_path: str, timing_info: str = None) -> str:
        """Random overlay effects applied at specific timestamps"""
        overlay_types = ['text_overlay', 'shape_overlay', 'noise_overlay', 'gradient_overlay']
        overlay_type = random.choice(overlay_types)

        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', input_path
            ], capture_output=True, text=True)

            duration = 10  # default
            if result.returncode == 0:
                try:
                    duration = float(result.stdout.strip())
                except:
                    pass

            # Generate 3-5 random overlay points
            overlay_points = FFmpegTransformationService.get_random_transformation_points(
                duration, num_points=random.randint(3, 5), min_interval=1.5
            )

            if overlay_type == 'text_overlay':
                # Random text overlays
                texts = ['HD', '✓', '★', '•', '▶', '◆', '※', '◊']
                text = random.choice(texts)
                fontsize = random.randint(24, 48)
                opacity = random.uniform(0.2, 0.5)

                # Random positions for each overlay
                positions = ['x=10:y=10', 'x=w-tw-10:y=10', 'x=10:y=h-th-10', 'x=w-tw-10:y=h-th-10', 'x=(w-tw)/2:y=(h-th)/2']
                position = random.choice(positions)

                enable_expressions = []
                for start_time, overlay_duration in overlay_points:
                    end_time = min(start_time + overlay_duration, duration)
                    enable_expressions.append(f"between(t,{start_time:.2f},{end_time:.2f})")

                combined_enable = "+".join(enable_expressions)
                filter_expr = f"drawtext=text='{text}':fontcolor=white@{opacity}:fontsize={fontsize}:{position}:enable='{combined_enable}'"

            elif overlay_type == 'shape_overlay':
                # Geometric shape overlay using drawbox
                box_width = random.randint(50, 150)
                box_height = random.randint(30, 100)
                box_x = random.randint(10, 200)
                box_y = random.randint(10, 150)
                opacity = random.uniform(0.1, 0.3)
                colors = ['red', 'blue', 'green', 'yellow', 'white', 'black']
                color = random.choice(colors)

                enable_expressions = []
                for start_time, overlay_duration in overlay_points:
                    end_time = min(start_time + overlay_duration, duration)
                    enable_expressions.append(f"between(t,{start_time:.2f},{end_time:.2f})")

                combined_enable = "+".join(enable_expressions)
                filter_expr = f"drawbox=x={box_x}:y={box_y}:w={box_width}:h={box_height}:color={color}@{opacity}:enable='{combined_enable}'"

            elif overlay_type == 'noise_overlay':
                # Noise overlay at specific regions
                noise_strength = random.uniform(0.05, 0.15)

                enable_expressions = []
                for start_time, overlay_duration in overlay_points:
                    end_time = min(start_time + overlay_duration, duration)
                    enable_expressions.append(f"between(t,{start_time:.2f},{end_time:.2f})")

                combined_enable = "+".join(enable_expressions)
                filter_expr = f"noise=alls={noise_strength}:allf=t:enable='{combined_enable}'"

            else:  # gradient_overlay
                # Gradient effect using eq filter
                saturation_change = random.uniform(0.7, 1.3)

                enable_expressions = []
                for start_time, overlay_duration in overlay_points:
                    end_time = min(start_time + overlay_duration, duration)
                    enable_expressions.append(f"between(t,{start_time:.2f},{end_time:.2f})")

                combined_enable = "+".join(enable_expressions)
                filter_expr = f"eq=saturation={saturation_change}:enable='{combined_enable}'"

            logging.info(f"🎨 Random overlay ({overlay_type}) at {len(overlay_points)} points: {[(round(s,1), round(d,1)) for s,d in overlay_points]}")
            return f'ffmpeg -i "{input_path}" -vf "{filter_expr}" -c:a copy -y "{output_path}"'

        except Exception:
            # Fallback
            overlay_time = random.uniform(2, 8)
            overlay_duration = random.uniform(1, 2)
            opacity = random.uniform(0.2, 0.4)
            logging.info(f"🎨 Simple overlay at {overlay_time:.1f}s for {overlay_duration:.1f}s")
            return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'*\':fontcolor=white@{opacity}:fontsize=32:x=50:y=50:enable=\'between(t,{overlay_time:.2f},{overlay_time + overlay_duration:.2f})\'" -c:a copy -y "{output_path}"'

    @staticmethod
    def random_motion_blur_effects(input_path: str, output_path: str, timing_info: str = None) -> str:
            """Random motion blur effects applied at specific timestamps - FIXED VERSION"""
            blur_types = ['directional_blur', 'radial_blur', 'zoom_blur', 'gaussian_blur']
            blur_type = random.choice(blur_types)

            try:
                import subprocess
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-show_entries', 
                    'format=duration', '-of', 'csv=p=0', input_path
                ], capture_output=True, text=True)

                duration = 10  # default
                if result.returncode == 0:
                    try:
                        duration = float(result.stdout.strip())
                    except:
                        pass

                # Generate 3-4 random blur points
                blur_points = FFmpegTransformationService.get_random_transformation_points(
                    duration, num_points=random.randint(3, 4), min_interval=2.0
                )

                if blur_type == 'directional_blur':
                    # Directional motion blur
                    blur_strength = random.uniform(0.8, 2.0)

                    enable_expressions = []
                    for start_time, blur_duration in blur_points:
                        end_time = min(start_time + blur_duration, duration)
                        enable_expressions.append(f"between(t,{start_time:.2f},{end_time:.2f})")

                    combined_enable = "+".join(enable_expressions)
                    filter_expr = f"boxblur={blur_strength}:{blur_strength/2}:enable='{combined_enable}'"

                elif blur_type == 'radial_blur':
                    # Radial blur simulation using unsharp (which supports timeline)
                    # Ensure blur_strength stays within valid range [-2, 5] for unsharp filter
                    blur_strength = random.uniform(0.5, 1.9)  # Keep negative values within [-1.9, -0.5] range

                    enable_expressions = []
                    for start_time, blur_duration in blur_points:
                        end_time = min(start_time + blur_duration, duration)
                        enable_expressions.append(f"between(t,{start_time:.2f},{end_time:.2f})")

                    combined_enable = "+".join(enable_expressions)
                    # Use negative sharpening for blur effect with timeline support
                    filter_expr = f"unsharp=5:5:-{blur_strength}:5:5:-{blur_strength}:enable='{combined_enable}'"

                elif blur_type == 'zoom_blur':
                    # FIXED: Use boxblur instead of scale for zoom blur effect
                    blur_strength = random.uniform(1.5, 3.0)

                    enable_expressions = []
                    for start_time, blur_duration in blur_points:
                        end_time = min(start_time + blur_duration, duration)
                        enable_expressions.append(f"between(t,{start_time:.2f},{end_time:.2f})")

                    combined_enable = "+".join(enable_expressions)
                    # Use boxblur with timeline support instead of problematic scale expressions
                    filter_expr = f"boxblur={blur_strength}:{blur_strength}:enable='{combined_enable}'"

                else:  # gaussian_blur
                    # Gaussian blur with timeline support
                    blur_strength = random.uniform(1.5, 3.0)

                    enable_expressions = []
                    for start_time, blur_duration in blur_points:
                        end_time = min(start_time + blur_duration, duration)
                        enable_expressions.append(f"between(t,{start_time:.2f},{end_time:.2f})")

                    combined_enable = "+".join(enable_expressions)
                    filter_expr = f"gblur=sigma='if({combined_enable},{blur_strength},0)'"

                logging.info(f"🌫️ Random motion blur ({blur_type}) at {len(blur_points)} points: {[(round(s,1), round(d,1)) for s,d in blur_points]}")
                return f'ffmpeg -i "{input_path}" -vf "{filter_expr}" -c:a copy -y "{output_path}"'

            except Exception:
                # Fallback - simple blur that definitely works
                blur_time = random.uniform(2, 8)
                blur_duration = random.uniform(0.8, 1.5)
                blur_strength = random.uniform(1.0, 2.0)
                logging.info(f"🌫️ Simple blur at {blur_time:.1f}s for {blur_duration:.1f}s")
                return f'ffmpeg -i "{input_path}" -vf "boxblur={blur_strength}:{blur_strength/2}:enable=\'between(t,{blur_time:.2f},{blur_time + blur_duration:.2f})\'" -c:a copy -y "{output_path}"'
        
    @staticmethod
    def get_audio_transformation_names() -> List[str]:
        return ['spectral_fingerprint_disruption', 'pitch_shift', 'tempo_shift', 'audio_simple_processing', 'audio_layering_ambient', 'audio_segment_reorder']
    
    @staticmethod
    def get_guaranteed_transformation_names() -> List[str]:
        # Deprecated - keeping for compatibility but returning empty list
        # All transformations are now fully randomized per variant
        return []
    
    @staticmethod
    def select_random_transformations(min_count: int = 9) -> List[TransformationConfig]:
        """Select 9-10 random transformations ensuring at least 5 enhanced transformations per variant"""
        available = FFmpegTransformationService.get_transformations()
        selected = []
        
        # Group transformations by category
        categories = {}
        for t in available:
            if t.category not in categories:
                categories[t.category] = []
            categories[t.category].append(t)
        
        # Define risky transformations that should be limited
        risky_transformations = {
            'black_screen_random', 'random_frame_inserts', 'grayscale_segment', 
            'zoom_jitter_motion', 'micro_rotation_crop'
        }
        
        # PHASE 1: ENSURE AT LEAST 5 ENHANCED TRANSFORMATIONS (NEW REQUIREMENT)
        enhanced_transforms = categories.get('enhanced', [])
        if len(enhanced_transforms) >= 5:
            # Prioritize safer enhanced transformations first
            safe_enhanced = [t for t in enhanced_transforms if t.name not in risky_transformations]
            risky_enhanced = [t for t in enhanced_transforms if t.name in risky_transformations]
            
            # Select at least 3 safe enhanced transformations
            selected_safe = random.sample(safe_enhanced, min(3, len(safe_enhanced)))
            selected.extend(selected_safe)
            
            # Fill remaining slots with a mix (limit risky ones to max 2)
            remaining_enhanced_slots = 5 - len(selected_safe)
            if remaining_enhanced_slots > 0:
                remaining_enhanced = [t for t in enhanced_transforms if t not in selected]
                # Limit risky transformations to at most 2
                risky_count = min(2, len([t for t in remaining_enhanced if t.name in risky_transformations]))
                safe_count = remaining_enhanced_slots - risky_count
                
                if safe_count > 0:
                    safe_remaining = [t for t in remaining_enhanced if t.name not in risky_transformations]
                    selected.extend(random.sample(safe_remaining, min(safe_count, len(safe_remaining))))
                
                if risky_count > 0:
                    risky_remaining = [t for t in remaining_enhanced if t.name in risky_transformations]
                    selected.extend(random.sample(risky_remaining, min(risky_count, len(risky_remaining))))
            
            logging.info(f'🎯 Selected Enhanced Transformations: {[t.name for t in selected]}')
        
        # PHASE 2: Ensure we have representation from other categories
        guaranteed_per_category = {
            'visual': random.randint(2, 5),      # 2 to 5 visual transformations per variant
            'audio': 2,       # 2 audio transformations  
            'structural': 1,  # 1 structural transformation (reduced)
            'metadata': 2,    # 2 metadata transformations (INCREASED from 1)
            'semantic': 1,    # 1 semantic transformation (reduced)
            'advanced': 1,    # 1 advanced transformation (reduced)
            'instagram': 1,   # 1 Instagram-specific transformation (NEW)
            'temporal': 2     # 2 temporal transformations with random timing (NEW)
        }
        
        for category, min_count in guaranteed_per_category.items():
            if category in categories:
                category_transforms = categories[category].copy()
                random.shuffle(category_transforms)
                selected.extend(category_transforms[:min_count])
        
        # PHASE 3: Add more random transformations to reach exactly 9-12 total
        target_count = random.randint(11, 12)  # Increased to accommodate enhanced transforms
        remaining_count = target_count - len(selected)
        
        if remaining_count > 0:
            # Get all unused transformations (excluding already selected enhanced ones)
            remaining_transforms = [t for t in available if t not in selected]
            random.shuffle(remaining_transforms)
            selected.extend(remaining_transforms[:remaining_count])
        elif len(selected) > target_count:
            # If we have too many, randomly remove some (but keep the 5 enhanced)
            non_enhanced = [t for t in selected if t.category != 'enhanced']
            enhanced_selected = [t for t in selected if t.category == 'enhanced']
            
            excess = len(selected) - target_count
            if excess > 0 and len(non_enhanced) >= excess:
                random.shuffle(non_enhanced)
                selected = enhanced_selected + non_enhanced[:-excess]
        
        # Shuffle final order
        random.shuffle(selected)
        
        logging.info(f'🎯 Enhanced Transformation Strategy (Total: {len(selected)}):')
        
        # Log by category
        selected_by_category = {}
        for t in selected:
            if t.category not in selected_by_category:
                selected_by_category[t.category] = []
            selected_by_category[t.category].append(t.name)
        
        for category, transforms in selected_by_category.items():
            emoji = '⭐' if category == 'enhanced' else '📊'
            logging.info(f'   {emoji} {category.upper()} ({len(transforms)}): {transforms}')
        
        # Verify we have at least 5 enhanced transformations
        enhanced_count = len([t for t in selected if t.category == 'enhanced'])
        logging.info(f'   ✅ Enhanced transformations count: {enhanced_count}/5 minimum')
        logging.info(f'   🎬 Execution order: {[t.name for t in selected]}')
        
        return selected
    
    @staticmethod
    def select_ssim_focused_transformations(min_count: int = 11) -> List[TransformationConfig]:
        """Select transformations with priority on high-impact SSIM reduction strategies
        
        This method prioritizes SSIM reduction transformations to target SSIM < 0.30
        while maintaining balanced quality and variation.
        """
        available = FFmpegTransformationService.get_transformations()
        selected = []
        
        # Group transformations by category
        categories = {}
        for t in available:
            if t.category not in categories:
                categories[t.category] = []
            categories[t.category].append(t)
        
        logging.info('🎯 Applying SSIM-Focused Transformation Strategy')
        
        # PHASE 1: MANDATORY SSIM REDUCTION STRATEGIES (3-4 transforms)
        ssim_transforms = categories.get('ssim_reduction', [])
        if ssim_transforms:
            # Always include the ultra-high-impact combo transformation
            combo_transform = next((t for t in ssim_transforms if 'combo' in t.name), None)
            if combo_transform:
                selected.append(combo_transform)
                logging.info(f'   ⚡ MANDATORY: {combo_transform.name} (Ultra High Impact)')
            
            # Select 2-3 additional high-impact SSIM transformations
            other_ssim = [t for t in ssim_transforms if t != combo_transform]
            high_impact_ssim = [
                t for t in other_ssim 
                if any(keyword in t.name for keyword in ['enhanced_crop_zoom', 'aggressive_hue', 'contrast_brightness', 'aggressive_gaussian'])
            ]
            
            ssim_count = random.randint(2, 3)
            if len(high_impact_ssim) >= ssim_count:
                selected.extend(random.sample(high_impact_ssim, ssim_count))
            else:
                selected.extend(high_impact_ssim)
                remaining_needed = ssim_count - len(high_impact_ssim)
                remaining_ssim = [t for t in other_ssim if t not in high_impact_ssim]
                if remaining_ssim:
                    selected.extend(random.sample(remaining_ssim, min(remaining_needed, len(remaining_ssim))))
            
            logging.info(f'   🎯 SSIM Reduction Count: {len([t for t in selected if t.category == "ssim_reduction"])}')
        
        # PHASE 2: ENHANCED ORB BREAKING (2-3 transforms)
        orb_transforms = categories.get('orb_breaking', [])
        if orb_transforms:
            # Prioritize high-impact ORB transforms
            enhanced_orb = [t for t in orb_transforms if 'enhanced' in t.name]
            orb_count = random.randint(2, 3)
            if len(enhanced_orb) >= orb_count:
                selected.extend(random.sample(enhanced_orb, orb_count))
            else:
                selected.extend(enhanced_orb)
                remaining_orb = [t for t in orb_transforms if t not in enhanced_orb]
                remaining_needed = orb_count - len(enhanced_orb)
                if remaining_orb and remaining_needed > 0:
                    selected.extend(random.sample(remaining_orb, min(remaining_needed, len(remaining_orb))))
            
            logging.info(f'   🎨 ORB Breaking Count: {len([t for t in selected if t.category == "orb_breaking"])}')
        
        # PHASE 3: COMPLEMENTARY CATEGORIES (ensure balance)
        complementary_requirements = {
            'visual': random.randint(1, 2),    # Additional visual effects
            'audio': 2,                        # Audio fingerprint breaking
            'metadata': 2,                     # 2 metadata randomizations (INCREASED from 1)
            'temporal': 1,                     # Temporal effects
            'overlay': 1,                      # Overlay effects
        }
        
        for category, count in complementary_requirements.items():
            if category in categories:
                available_in_category = [t for t in categories[category] if t not in selected]
                if available_in_category:
                    actual_count = min(count, len(available_in_category))
                    selected.extend(random.sample(available_in_category, actual_count))
        
        # PHASE 4: Fill to target count with high-quality transforms
        target_count = random.randint(min_count, min_count + 2)
        remaining_needed = target_count - len(selected)
        
        if remaining_needed > 0:
            # Prioritize high-probability transforms for the remaining slots
            remaining_transforms = [t for t in available if t not in selected]
            high_quality = [t for t in remaining_transforms if t.probability >= 0.6]
            
            if len(high_quality) >= remaining_needed:
                selected.extend(random.sample(high_quality, remaining_needed))
            else:
                selected.extend(high_quality)
                still_needed = remaining_needed - len(high_quality)
                other_transforms = [t for t in remaining_transforms if t not in high_quality]
                if other_transforms and still_needed > 0:
                    selected.extend(random.sample(other_transforms, min(still_needed, len(other_transforms))))
        
        # Shuffle for random execution order
        random.shuffle(selected)
        
        # Log final selection
        logging.info(f'🎯 SSIM-Focused Strategy Summary (Total: {len(selected)}):')
        
        selected_by_category = {}
        for t in selected:
            if t.category not in selected_by_category:
                selected_by_category[t.category] = []
            selected_by_category[t.category].append(t.name)
        
        category_emojis = {
            'ssim_reduction': '🎯',
            'orb_breaking': '🎨', 
            'visual': '👁️',
            'audio': '🎵',
            'metadata': '📋',
            'temporal': '⏱️',
            'overlay': '✨'
        }
        
        for category, transforms in selected_by_category.items():
            emoji = category_emojis.get(category, '📊')
            logging.info(f'   {emoji} {category.upper()} ({len(transforms)}): {transforms}')
        
        ssim_count = len([t for t in selected if t.category == 'ssim_reduction'])
        logging.info(f'   ⚡ High-Impact SSIM Reduction: {ssim_count} transforms')
        logging.info(f'   🎯 Expected SSIM Target: < 0.30 (Ultra Effective)')
        
        return selected
    
    @staticmethod
    def get_random_transformation_points(video_duration: float, num_points: int = 4, min_interval: float = 1.5) -> List[tuple]:
        """Generate random points for applying transformations throughout video"""
        if video_duration < 5:
            # For short videos, apply fewer transformations
            num_points = max(2, num_points // 2)
        
        points = []
        used_times = []
        
        for _ in range(num_points):
            attempts = 0
            while attempts < 20:  # Prevent infinite loop
                start_time = random.uniform(1, max(2, video_duration - 2))
                
                # Check if this time conflicts with existing points
                too_close = False
                for used_time in used_times:
                    if abs(start_time - used_time) < min_interval:
                        too_close = True
                        break
                
                if not too_close:
                    duration = random.uniform(0.5, min(2.5, video_duration * 0.2))
                    points.append((start_time, duration))
                    used_times.append(start_time)
                    break
                
                attempts += 1
        
        # Sort by start time
        points.sort()
        return points

    @staticmethod
    def select_fully_random_transformations(num_transformations: int = None, video_duration: float = 10, variant_seed: str = None) -> List[tuple]:
        """Select completely random transformations for each variant - NO GUARANTEED TRANSFORMATIONS
        
        🎯 Goals:
        - Each variant gets completely different transformations
        - No guaranteed/always-on transformations
        - True randomization for maximum variety between variants
        - Support 15-20 transformations per variant for stronger effect
        - Use variant-specific seed for reproducible but unique results per variant
        """
        
        # Set unique seed for this variant to ensure different transformations
        if variant_seed:
            # Create a unique seed based on the variant identifier
            import hashlib
            seed_hash = hashlib.md5(variant_seed.encode()).hexdigest()
            seed_int = int(seed_hash[:8], 16)  # Use first 8 hex chars as integer
            random.seed(seed_int)
            logging.info(f'🎲 Variant seed: {variant_seed} -> Random seed: {seed_int}')
        
        available = FFmpegTransformationService.get_transformations()
        
        # Determine number of transformations if not specified
        if num_transformations is None:
            num_transformations = random.randint(18, 25)  # MINIMUM 18: Increased for stronger effect with guaranteed minimum
        
        logging.info(f'🎲 FULLY RANDOM TRANSFORMATION SELECTION: {num_transformations} transformations')
        logging.info(f'📊 Available transformation pool: {len(available)} total transformations')
        
        # Group transformations by category for metadata guarantee
        categories = {}
        for t in available:
            if t.category not in categories:
                categories[t.category] = []
            categories[t.category].append(t)
        
        # GUARANTEE at least 2 metadata transformations in every variant
        selected_configs = []
        metadata_transforms = categories.get('metadata', [])
        if metadata_transforms:
            # Select 2 random metadata transformations first
            guaranteed_metadata = random.sample(metadata_transforms, min(2, len(metadata_transforms)))
            selected_configs.extend(guaranteed_metadata)
            logging.info(f'🔒 GUARANTEED METADATA: {[t.name for t in guaranteed_metadata]}')
        
        # Then select remaining transformations randomly from the rest
        remaining_available = [t for t in available if t not in selected_configs]
        remaining_needed = num_transformations - len(selected_configs)
        
        if remaining_needed > 0:
            random.shuffle(remaining_available)
            selected_configs.extend(remaining_available[:remaining_needed])
        
        # Log what was randomly selected
        selected_by_category = {}
        for config in selected_configs:
            if config.category not in selected_by_category:
                selected_by_category[config.category] = []
            selected_by_category[config.category].append(config.name)
        
        logging.info(f'🎯 RANDOM SELECTION RESULTS:')
        for category, transforms in selected_by_category.items():
            logging.info(f'   🎲 {category.upper()} ({len(transforms)}): {transforms}')
        
        # Add timing information for temporal transformations
        result = []
        for transformation in selected_configs:
            if transformation.supports_temporal:
                # Generate random timing points for temporal transformations
                timing_points = FFmpegTransformationService.get_random_transformation_points(
                    video_duration, num_points=random.randint(3, 5), min_interval=2.0
                )
                timing_info = f"Applied at: {', '.join([f'{t[0]:.1f}s({t[1]:.1f}s)' for t in timing_points])}"
                result.append((transformation, timing_info))
            else:
                result.append((transformation, None))
        
        # Final shuffle of execution order
        random.shuffle(result)
        
        final_order = [t[0].name for t in result]
        logging.info(f'🎬 Final execution order: {final_order}')
        
        # Reset random seed to system default after variant selection
        random.seed()
        
        return result

    @staticmethod
    def select_mixed_transformations(video_duration: float = 10) -> List[tuple]:
        """COMPREHENSIVE ORB BREAKING STRATEGY IMPLEMENTATION
        
        🎯 Strategy Goals:
        - Reduce ORB similarity < 9000
        - Maintain SSIM ≥ 0.35, pHash distance ~25–35
        - Audio similarity ↓, Metadata similarity ↓
        - Maintain watchability ≥ 90%
        - Support adaptive randomization across 7 transformation layers
        """
        available = FFmpegTransformationService.get_transformations()
        
        # Group transformations by category
        categories = {}
        for t in available:
            if t.category not in categories:
                categories[t.category] = []
            categories[t.category].append(t)
        
        logging.info('🎯 COMPREHENSIVE ORB BREAKING STRATEGY: Implementing 7-layer adaptive randomization')
        
        selected = []
        
        # 1. CORE ORB DISRUPTORS (always-on)
        orb_core_transforms = categories.get('orb_core', [])
        if orb_core_transforms:
            selected.extend(orb_core_transforms)  # Add all core ORB disruptors
            logging.info(f'✅ Core ORB Disruptors: {len(orb_core_transforms)} transformations (always-on)')
        
        # 2. VISUAL VARIATION (70% prob)
        orb_visual_transforms = categories.get('orb_visual', [])
        if orb_visual_transforms:
            for transform in orb_visual_transforms:
                if random.random() < 0.7:  # 70% probability
                    selected.append(transform)
            logging.info(f'✅ Visual Variation: {len([t for t in selected if t.category == "orb_visual"])} selected (70% prob)')
        
        # 3. STRUCTURED RANDOMIZER (60% prob)
        orb_structured_transforms = categories.get('orb_structured', [])
        if orb_structured_transforms:
            for transform in orb_structured_transforms:
                if random.random() < 0.6:  # 60% probability
                    selected.append(transform)
            logging.info(f'✅ Structured Randomizer: {len([t for t in selected if t.category == "orb_structured"])} selected (60% prob)')
        
        # 4. STABILITY ENHANCER (50% prob)
        orb_stability_transforms = categories.get('orb_stability', [])
        if orb_stability_transforms:
            for transform in orb_stability_transforms:
                if random.random() < 0.5:  # 50% probability
                    selected.append(transform)
            logging.info(f'✅ Stability Enhancer: {len([t for t in selected if t.category == "orb_stability"])} selected (50% prob)')
        
        # 5. AUDIO TRANSFORM (50% prob)
        orb_audio_transforms = categories.get('orb_audio', [])
        if orb_audio_transforms:
            for transform in orb_audio_transforms:
                if random.random() < 0.5:  # 50% probability
                    selected.append(transform)
            logging.info(f'✅ Audio Transform: {len([t for t in selected if t.category == "orb_audio"])} selected (50% prob)')
        
        # 6. METADATA LAYER (100%)
        orb_metadata_transforms = categories.get('orb_metadata', [])
        if orb_metadata_transforms:
            selected.extend(orb_metadata_transforms)  # Add all metadata transformations
            logging.info(f'✅ Metadata Layer: {len(orb_metadata_transforms)} transformations (100%)')
        
        # 7. SEMANTIC NOISE (40% prob)
        orb_semantic_transforms = categories.get('orb_semantic', [])
        if orb_semantic_transforms:
            for transform in orb_semantic_transforms:
                if random.random() < 0.4:  # 40% probability
                    selected.append(transform)
            logging.info(f'✅ Semantic Noise: {len([t for t in selected if t.category == "orb_semantic"])} selected (40% prob)')
        
        # SUPPLEMENTARY TRANSFORMATIONS: Add some classic transformations for balance
        # Ensure we have a good mix by adding some proven transformations
        supplementary_categories = {
            'audio': 2,      # 2 classic audio transformations
            'visual': 2,     # 2 classic visual transformations
            'metadata': 2,   # 2 classic metadata transformations (INCREASED from 1)
            'enhanced': 2,   # 2 enhanced transformations
            'temporal': 1    # 1 temporal transformation
        }
        
        for category, min_count in supplementary_categories.items():
            if category in categories:
                # Only add if we don't already have enough from ORB categories
                current_count = len([t for t in selected if t.category == category])
                if current_count < min_count:
                    available_transforms = [t for t in categories[category] if t not in selected]
                    needed = min_count - current_count
                    if available_transforms:
                        random.shuffle(available_transforms)
                        selected.extend(available_transforms[:needed])
        
        # Final count and logging
        total_count = len(selected)
        logging.info(f'🎯 COMPREHENSIVE ORB STRATEGY COMPLETE: {total_count} total transformations selected')
        
        # Log category breakdown
        category_counts = {}
        for transform in selected:
            category_counts[transform.category] = category_counts.get(transform.category, 0) + 1
        
        for category, count in category_counts.items():
            logging.info(f'   📊 {category}: {count} transformations')
        
        # Add timing information for temporal transformations
        result = []
        for transformation in selected:
            if transformation.supports_temporal:
                # Generate random timing points for temporal transformations
                timing_points = FFmpegTransformationService.get_random_transformation_points(
                    video_duration, num_points=random.randint(3, 5), min_interval=2.0
                )
                timing_info = f"Applied at: {', '.join([f'{t[0]:.1f}s({t[1]:.1f}s)' for t in timing_points])}"
                result.append((transformation, timing_info))
            else:
                result.append((transformation, None))
        
        return result

    @staticmethod
    async def execute_command(command: str, timeout: int = 120) -> bool:
        """Execute FFmpeg command with timeout and detailed error handling"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            if process.returncode != 0:
                stderr_text = stderr.decode()
                # Check for specific FFmpeg errors and provide helpful context
                if "Undefined constant" in stderr_text:
                    logging.error(f"❌ FFmpeg filter syntax error: {stderr_text}")
                elif "Invalid chars" in stderr_text:
                    logging.error(f"❌ FFmpeg filter parameter error: {stderr_text}")
                elif "Nothing was written into output file" in stderr_text:
                    logging.error(f"❌ FFmpeg filter chain failed - no output generated: {stderr_text}")
                elif "Option not found" in stderr_text:
                    logging.error(f"❌ FFmpeg filter option not supported: {stderr_text}")
                else:
                    logging.error(f"❌ FFmpeg command failed: {stderr_text}")
                return False
            
            # Check for warnings but don't fail
            if stderr:
                stderr_text = stderr.decode()
                if "No accelerated colorspace conversion" in stderr_text:
                    logging.info(f"⚠️ Colorspace conversion warning (non-critical): Using software conversion")
                elif "deprecated" in stderr_text.lower():
                    logging.info(f"⚠️ FFmpeg deprecation warning: {stderr_text}")
            
            return True
            
        except asyncio.TimeoutError:
            logging.error(f"❌ Command timed out after {timeout} seconds")
            return False
        except Exception as e:
            logging.error(f"❌ Command execution failed: {e}")
            return False
    
    @staticmethod
    async def apply_transformations(
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
        variant_id: str = None,
        strategy: str = "random"  # "random", "ssim_focused", "seven_layer", "comprehensive_ssim"
    ) -> List[str]:
        """Apply video transformations with progress tracking and temporal distribution"""
        try:
            variant_info = f" (Variant: {variant_id})" if variant_id else ""
            logging.info(f'🎬 Applying FULLY RANDOM transformations to: {os.path.basename(input_path)}{variant_info}')
            
            # Validate input file
            if not await FFmpegTransformationService.validate_video_file(input_path):
                raise Exception(f'Invalid input video file: {input_path}')
            
            # Get video duration for temporal planning
            video_info = await FFmpegTransformationService.get_video_info(input_path)
            video_duration = video_info['duration']
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # STRATEGY-BASED TRANSFORMATION SELECTION
            if strategy == "ssim_focused":
                logging.info(f'🎯 Using SSIM-FOCUSED strategy for maximum structural similarity reduction')
                selected_configs = FFmpegTransformationService.select_ssim_focused_transformations(min_count=11)
                selected_transformations = [(config, None) for config in selected_configs]  # No temporal timing for focused strategy
            elif strategy == "comprehensive_ssim":
                logging.info(f'🎯 Using COMPREHENSIVE SSIM REDUCTION strategy from strategy table')
                # Apply the comprehensive SSIM strategy directly
                strategy_level = random.choice(["medium", "high", "extreme"])  # Random strategy level
                result = FFmpegTransformationService.apply_comprehensive_ssim_strategy(input_path, output_path, strategy_level)
                logging.info(f'🎯 COMPREHENSIVE SSIM STRATEGY COMPLETE: {result}')
                return [result]
            elif strategy == "seven_layer":
                logging.info(f'🔴 Using 7-LAYER PIPELINE strategy')
                # Keep existing seven_layer logic if it exists
                selected_transformations = FFmpegTransformationService.select_fully_random_transformations(
                    num_transformations=random.randint(18, 30),
                    video_duration=video_duration,
                    variant_seed=variant_id
                )
            else:  # "random" or default
                logging.info(f'🎲 Using FULLY RANDOM strategy')
                selected_transformations = FFmpegTransformationService.select_fully_random_transformations(
                    num_transformations=random.randint(18, 30),  # MINIMUM 18: Ensured at least 18 transformations for maximum variation
                    video_duration=video_duration,
                    variant_seed=variant_id  # Use variant ID as seed for unique randomization
                )
            
            applied_transformations = []
            
            current_input = input_path
            temp_counter = 0
            temp_files = []
            
            total_transformations = len(selected_transformations)
            
            strategy_emoji = ("🎯" if strategy == "ssim_focused" 
                           else "🎯" if strategy == "comprehensive_ssim" 
                           else "🔴" if strategy == "seven_layer" 
                           else "🎲")
            logging.info(f'{strategy_emoji} {strategy.upper()} Transformation Plan ({total_transformations} total):')
            for i, (transformation, timing_info) in enumerate(selected_transformations):
                if timing_info:
                    logging.info(f'   {i+1}. {transformation.name} ({transformation.category}) - {timing_info}')
                else:
                    logging.info(f'   {i+1}. {transformation.name} ({transformation.category}) - Global effect')
            
            logging.info(f'🚀 Starting fully randomized transformation pipeline...')
            
            for i, (transformation, timing_info) in enumerate(selected_transformations):
                temp_output = os.path.join(
                    os.path.dirname(output_path),
                    f'temp_{temp_counter}_{os.path.basename(output_path)}'
                )
                
                try:
                    effect_description = f"{transformation.name} ({transformation.category})"
                    if timing_info:
                        effect_description += f" - {timing_info}"
                    
                    logging.info(f'  📝 Applying: {effect_description} ({i+1}/{total_transformations})')
                    
                    # Update progress
                    if progress_callback:
                        progress = (i / total_transformations) * 100
                        if asyncio.iscoroutinefunction(progress_callback):
                            await progress_callback(progress)
                        else:
                            progress_callback(progress)
                    
                    # Validate current input
                    if not await FFmpegTransformationService.validate_video_file(current_input):
                        logging.warning(f'⚠️ Invalid input file for {transformation.name}: {current_input}')
                        continue
                    
                    # Execute transformation with timing info if applicable
                    if asyncio.iscoroutinefunction(transformation.execute):
                        command = await transformation.execute(current_input, temp_output)
                    else:
                        command = transformation.execute(current_input, temp_output)
                    
                    logging.info(f'  🔧 Command: {command[:100]}...')
                    
                    success = await FFmpegTransformationService.execute_command(command)
                    
                    if not success:
                        logging.warning(f'⚠️ Failed to apply {transformation.name}')
                        continue
                    
                    # Validate output more thoroughly
                    if not await FFmpegTransformationService.validate_video_file(temp_output):
                        logging.warning(f'⚠️ Invalid output file created for {transformation.name}: {temp_output}')
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                        continue
                    
                    # Additional check: ensure video is not completely black, white, or corrupted
                    try:
                        # Enhanced validation using ffprobe to verify the video has proper content
                        probe_cmd = [
                            'ffprobe', '-v', 'quiet', '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate',
                            '-of', 'csv=p=0', temp_output
                        ]
                        probe_result = await asyncio.create_subprocess_exec(
                            *probe_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        stdout, stderr = await probe_result.communicate()
                        
                        if probe_result.returncode != 0:
                            logging.warning(f'⚠️ Video validation failed for {transformation.name}')
                            if os.path.exists(temp_output):
                                os.remove(temp_output)
                            continue
                        
                        # Additional check: Use ffmpeg to analyze a sample frame for corruption
                        frame_check_cmd = [
                            'ffmpeg', '-i', temp_output, '-vf', 'select=eq(n\\,5)', '-vframes', '1', 
                            '-f', 'null', '-'
                        ]
                        frame_check_result = await asyncio.create_subprocess_exec(
                            *frame_check_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await frame_check_result.communicate()
                        
                        if frame_check_result.returncode != 0:
                            logging.warning(f'⚠️ Frame corruption detected for {transformation.name}')
                            if os.path.exists(temp_output):
                                os.remove(temp_output)
                            continue
                            
                    except Exception as probe_error:
                        logging.warning(f'⚠️ Could not validate video for {transformation.name}: {probe_error}')
                        # Continue anyway if probe fails, but add extra logging
                        logging.info(f'   📊 Continuing with {transformation.name} despite validation warning')
                    
                    # Clean up previous temp file
                    if current_input != input_path and current_input in temp_files:
                        if os.path.exists(current_input):
                            os.remove(current_input)
                    
                    current_input = temp_output
                    temp_files.append(temp_output)
                    applied_transformations.append(transformation.name)
                    temp_counter += 1
                    
                except Exception as error:
                    logging.warning(f'⚠️ Failed to apply {transformation.name}: {error}')
                    continue
            
            # Final progress update
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(100.0)
                else:
                    progress_callback(100.0)
            
            # MINIMUM 18 TRANSFORMATIONS VALIDATION
            if len(applied_transformations) < 18:
                logging.warning(f'⚠️ Only {len(applied_transformations)} transformations applied, minimum is 18')
                
                # Try to apply additional transformations from remaining pool to meet minimum
                available_transformations = FFmpegTransformationService.get_transformations()
                remaining_transformations = [t for t in available_transformations if t.name not in applied_transformations]
                
                needed_count = 18 - len(applied_transformations)
                logging.info(f'🔄 Attempting to apply {needed_count} additional transformations to meet minimum')
                
                # Shuffle and select from remaining
                random.shuffle(remaining_transformations)
                additional_transformations = remaining_transformations[:needed_count * 2]  # Try twice the needed amount
                
                for transformation in additional_transformations:
                    if len(applied_transformations) >= 18:
                        break
                        
                    temp_output = os.path.join(
                        os.path.dirname(output_path),
                        f'temp_additional_{len(applied_transformations)}_{os.path.basename(output_path)}'
                    )
                    
                    try:
                        logging.info(f'  📝 Applying additional: {transformation.name} ({transformation.category})')
                        
                        if asyncio.iscoroutinefunction(transformation.execute):
                            command = await transformation.execute(current_input, temp_output)
                        else:
                            command = transformation.execute(current_input, temp_output)
                        
                        success = await FFmpegTransformationService.execute_command(command)
                        
                        if success and await FFmpegTransformationService.validate_video_file(temp_output):
                            # Clean up previous temp file
                            if current_input != input_path and os.path.exists(current_input):
                                os.remove(current_input)
                            
                            current_input = temp_output
                            temp_files.append(temp_output)
                            applied_transformations.append(transformation.name)
                            logging.info(f'  ✅ Successfully applied additional: {transformation.name}')
                        else:
                            if os.path.exists(temp_output):
                                os.remove(temp_output)
                            logging.warning(f'  ❌ Failed additional: {transformation.name}')
                            
                    except Exception as e:
                        logging.warning(f'  ❌ Error with additional {transformation.name}: {e}')
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
            
            # Final validation of minimum requirement
            if len(applied_transformations) < 18:
                logging.error(f'❌ FAILED TO MEET MINIMUM: Only {len(applied_transformations)}/18 transformations applied')
                # Still continue but log the issue
            else:
                logging.info(f'✅ MINIMUM REQUIREMENT MET: {len(applied_transformations)}/18+ transformations applied')
            
            # Ensure we have at least one successful transformation
            if len(applied_transformations) == 0:
                logging.warning('⚠️ No transformations applied, copying original file')
                import shutil
                shutil.copy2(input_path, output_path)
                return ['copy_original']
            
            # Move final result to output path
            if current_input != output_path and os.path.exists(current_input):
                import shutil
                shutil.move(current_input, output_path)
            
            # Clean up remaining temp files
            for temp_file in temp_files:
                if temp_file != output_path and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            
            logging.info(f'✅ Applied {len(applied_transformations)} FULLY RANDOM transformations - Each variant is unique!')
            logging.info(f'📊 Random transformations applied: {applied_transformations}')
            return applied_transformations
            
        except Exception as error:
            logging.error(f'❌ Error applying transformations: {error}')
            raise
    
    @staticmethod
    async def apply_seven_layer_transformations(
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
        variant_id: str = None
    ) -> List[str]:
        """
        Apply the 7-layer transformation pipeline for maximum similarity reduction.
        
        This method uses the advanced 7-layer strategy to target specific similarity metrics:
        - Higher entropy and randomness than standard transformations
        - Specifically designed to reduce ORB, PHash, SSIM, and audio fingerprint similarities
        - Each layer targets different detection algorithms
        
        Returns:
            List of applied transformations
        """
        try:
            variant_info = f" (Variant: {variant_id})" if variant_id else ""
            logging.info(f'🎯 Applying 7-LAYER PIPELINE transformations to: {os.path.basename(input_path)}{variant_info}')
            
            # Validate input file
            if not await FFmpegTransformationService.validate_video_file(input_path):
                raise Exception(f'Invalid input video file: {input_path}')
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert variant_id to integer for deterministic randomness
            variant_seed = None
            if variant_id:
                try:
                    # Use hash of variant_id to get consistent integer
                    variant_seed = hash(variant_id) % 10000
                except:
                    variant_seed = random.randint(1, 10000)
            
            # Apply the 7-layer pipeline
            if progress_callback:
                progress_callback(10.0)
            
            applied_transforms = FFmpegTransformationService.apply_seven_layer_pipeline(
                input_path, output_path, variant_seed
            )
            
            if progress_callback:
                progress_callback(90.0)
            
            # Validate output
            if not await FFmpegTransformationService.validate_video_file(output_path):
                raise Exception(f'Output video validation failed: {output_path}')
            
            if progress_callback:
                progress_callback(100.0)
            
            logging.info(f'✅ 7-Layer pipeline complete: {len(applied_transforms)} transformations applied')
            logging.info(f'📊 7-Layer transformations: {applied_transforms}')
            return applied_transforms
            
        except Exception as error:
            logging.error(f'❌ Error applying 7-layer transformations: {error}')
            raise
    
    @staticmethod
    def get_transformation_strategy_info() -> Dict[str, Any]:
        """
        Get information about available transformation strategies.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            "strategies": {
                "standard": {
                    "name": "Standard Random Transformations",
                    "description": "16-24 fully random transformations for balanced quality and variation",
                    "target_metrics": "MSE 20-50, SSIM 0.30-0.35, balanced correlation scores",
                    "transformations": "16-24 random",
                    "quality": "High - preserves originality and watchability",
                    "effectiveness": "Good - ~50 total risk target"
                },
                "seven_layer": {
                    "name": "7-Layer Transformation Pipeline",
                    "description": "Advanced layered approach targeting specific similarity metrics",
                    "target_metrics": "Maximum entropy reduction across ORB, PHash, SSIM, Audio",
                    "layers": {
                        "layer_1": "ORB Similarity Disruption (2-3 transforms)",
                        "layer_2": "Audio Fingerprint Obfuscation (2-3 transforms)",
                        "layer_3": "SSIM & Structural Shift (1-2 transforms)",
                        "layer_4": "PHash Distance Increase (1-2 transforms)",
                        "layer_5": "Metadata Scrambling (2-3 transforms)",
                        "layer_6": "Temporal Flow Disruption (1-2 transforms)",
                        "layer_7": "Semantic / Overlay Distortion (1-2 transforms)"
                    },
                    "transformations": "9-16 targeted",
                    "quality": "Medium-High - more aggressive but controlled",
                    "effectiveness": "Maximum - targets all similarity detection methods"
                }
            },
            "recommendation": "Use 'seven_layer' for maximum similarity reduction, 'standard' for balanced quality"
        }
    
    @staticmethod
    async def check_ffmpeg_installation() -> bool:
        """Check if FFmpeg is installed and available"""
        try:
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if process.returncode == 0:
                logging.info('✅ FFmpeg is installed and available')
                return True
            else:
                logging.error('❌ FFmpeg is not working properly')
                return False
                
        except FileNotFoundError:
            logging.error('❌ FFmpeg is not installed or not in PATH')
            return False
        except Exception as e:
            logging.error(f'❌ Error checking FFmpeg: {e}')
            return False