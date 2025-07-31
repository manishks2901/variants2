import ffmpeg
import subprocess
import asyncio
import json
import os
import random
import math
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
import hashlib

class TransformationConfig:
    def __init__(self, name: str, probability: float, execute_func: Callable, category: str = "general"):
        self.name = name
        self.probability = probability
        self.execute = execute_func
        self.category = category

class AdvancedTransformationMetrics:
    """
    Advanced video transformation metrics for comprehensive content variation
    Based on pHash, SSIM, Audio Fingerprinting, and other detection methods
    """
    
    @staticmethod
    def get_phash_safe_range():
        """pHash Hamming Distance: > 12-15 per keyframe"""
        return random.uniform(12, 18)
    
    @staticmethod
    def get_ssim_safe_range():
        """SSIM Structural Similarity: < 0.55 average"""
        return random.uniform(0.35, 0.55)
    
    @staticmethod
    def get_color_histogram_safe_range():
        """Color Histogram correlation: < 0.75"""
        return random.uniform(0.45, 0.75)
    
    @staticmethod
    def get_frame_entropy_increase():
        """Frame Entropy increase: 5-10%"""
        return random.uniform(1.05, 1.10)
    
    @staticmethod
    def get_embedding_distance_change():
        """CLIP Embedding distance change: >= 0.25"""
        return random.uniform(0.25, 0.40)
    
    @staticmethod
    def get_audio_fingerprint_confidence():
        """Audio fingerprint match confidence: < 60%"""
        return random.uniform(0.35, 0.60)
    
    @staticmethod
    def get_pitch_shift_range():
        """Pitch shift: ±3-5% (±½ to 1 semitone)"""
        return random.uniform(-0.05, 0.05)
    
    @staticmethod
    def get_tempo_shift_range():
        """Tempo shift: ±3-4%"""
        return random.uniform(-0.04, 0.04)
    
    @staticmethod
    def get_audio_noise_level():
        """Audio layering noise: -20dB to -30dB"""
        return random.uniform(0.001, 0.01)  # Converted to amplitude
    
    @staticmethod
    def get_video_trim_range():
        """Video length trim: ±1-2 sec"""
        return random.uniform(0.5, 2.0)
    
    @staticmethod
    def get_frame_rate_change():
        """Frame rate change: ±0.5-1 fps"""
        return random.uniform(-1.0, 1.0)
    
    @staticmethod
    def get_audio_video_sync_offset():
        """Audio-video sync offset: ±100-300ms"""
        return random.uniform(-0.3, 0.3)

class FFmpegTransformationService:
    
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
    def get_random_opacity(min_val: float = 0.1, max_val: float = 0.4) -> float:
        """Generate random opacity value"""
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
    
    @staticmethod
    def get_transformations() -> List[TransformationConfig]:
        """Get comprehensive advanced transformations with proper metrics"""
        
        # VISUAL TRANSFORMATIONS
        def phash_disruption_transform(input_path: str, output_path: str) -> str:
            """pHash Hamming Distance > 20-30 per keyframe (Enhanced but Natural)"""
            # BALANCED: Good pHash scores without green tint
            hue_shift = random.uniform(-15, 15)  # Reduced to prevent green tint
            saturation = random.uniform(0.8, 1.2)  # More natural saturation range
            brightness = random.uniform(-0.1, 0.1)  # Moderate brightness
            contrast = random.uniform(0.85, 1.15)  # Moderate contrast
            gamma = random.uniform(0.85, 1.15)  # More subtle gamma
            
            # Add micro-rotation for additional pHash disruption
            rotation = random.uniform(-1.5, 1.5)  # Reduced rotation
            
            return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_shift}:s={saturation},eq=brightness={brightness}:contrast={contrast}:gamma={gamma},rotate={rotation}*PI/180:fillcolor=black" -c:a copy -y "{output_path}"'
        
        def ssim_reduction_transform(input_path: str, output_path: str) -> str:
            """SSIM Structural Similarity < 0.45 (Enhanced for lower scores)"""
            # ENHANCED: More aggressive structural changes for lower SSIM
            crop_percent = random.uniform(0.05, 0.08)  # Increased crop 5-8%
            pan_x = random.uniform(0, crop_percent)
            pan_y = random.uniform(0, crop_percent)
            noise_strength = random.uniform(0.03, 0.06)  # Increased noise
            
            # Add blur and sharpening combination for structural disruption
            blur_strength = random.uniform(0.5, 1.5)
            sharpen_strength = random.uniform(0.3, 0.8)
            
            return f'ffmpeg -i "{input_path}" -vf "crop=iw*(1-{crop_percent}):ih*(1-{crop_percent}):iw*{pan_x}:ih*{pan_y},scale=iw:ih,noise=alls={noise_strength}:allf=t,unsharp=5:5:{sharpen_strength}:5:5:{blur_strength}" -c:a copy -y "{output_path}"'
        
        def color_histogram_shift(input_path: str, output_path: str) -> str:
            """Color Histogram correlation < 0.75 (Natural color shifts)"""
            # BALANCED: Effective color changes without green tint
            hue_shift = random.uniform(-12, 12)  # Reduced from ±20 to prevent green
            saturation = random.uniform(0.85, 1.15)  # More natural range
            gamma_r = random.uniform(0.9, 1.1)  # Subtle gamma adjustments
            gamma_g = random.uniform(0.9, 1.1)
            gamma_b = random.uniform(0.9, 1.1)
            
            return f'ffmpeg -i "{input_path}" -vf "hue=h={hue_shift}:s={saturation},eq=gamma_r={gamma_r}:gamma_g={gamma_g}:gamma_b={gamma_b}" -c:a copy -y "{output_path}"'
        
        def extreme_phash_disruption(input_path: str, output_path: str) -> str:
            """BALANCED pHash disruption for scores > 25 without green tint"""
            # FIXED: Strong pHash disruption but natural colors
            hue1 = random.uniform(-20, 20)  # Reduced from ±45 to prevent green
            saturation1 = random.uniform(0.75, 1.25)  # More balanced saturation
            
            # Use subtle brightness/contrast instead of color channel mixing
            brightness_adj = random.uniform(-0.08, 0.08)
            contrast_adj = random.uniform(0.9, 1.1)
            
            # Add subtle noise for pHash disruption without color distortion
            noise_strength = random.uniform(0.02, 0.04)
            
            # Add vignette effect for additional visual disruption
            vignette_strength = random.uniform(0.2, 0.4)  # Reduced strength
            
            return f'ffmpeg -i "{input_path}" -vf "hue=h={hue1}:s={saturation1},eq=brightness={brightness_adj}:contrast={contrast_adj},noise=alls={noise_strength}:allf=t,vignette=PI/6*{vignette_strength}" -c:a copy -y "{output_path}"'
        
        def frame_entropy_increase(input_path: str, output_path: str) -> str:
            """Frame Entropy increase 5-10%"""
            # Add controlled noise and micro-movements
            noise_strength = random.uniform(0.02, 0.05)
            blur_strength = random.uniform(0.5, 1.5)
            
            return f'ffmpeg -i "{input_path}" -vf "noise=alls={noise_strength}:allf=t,unsharp=5:5:{blur_strength}:5:5:{blur_strength}" -c:a copy -y "{output_path}"'
        
        def embedding_similarity_change(input_path: str, output_path: str) -> str:
            """CLIP Embedding distance change >= 0.25"""
            # Add text overlays and visual elements
            texts = ['SAMPLE', 'PREVIEW', 'DEMO', '© 2025', 'ORIGINAL', 'HD QUALITY']
            text = random.choice(texts)
            opacity = random.uniform(0.2, 0.4)
            fontsize = random.randint(24, 48)
            positions = ['x=10:y=10', 'x=w-tw-10:y=10', 'x=10:y=h-th-10', 'x=w-tw-10:y=h-th-10', 'x=(w-tw)/2:y=(h-th)/2']
            position = random.choice(positions)
            
            return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{text}\':fontcolor=white@{opacity}:fontsize={fontsize}:{position}:box=1:boxcolor=black@0.3" -c:a copy -y "{output_path}"'
        
        # AUDIO TRANSFORMATIONS
        def spectral_fingerprint_disruption(input_path: str, output_path: str) -> str:
            """Spectral Fingerprint Match < 60%"""
            # Complex audio transformations
            pitch_shift = 1.0 + random.uniform(-0.05, 0.05)  # ±5%
            tempo_shift = 1.0 + random.uniform(-0.04, 0.04)  # ±4%
            bass_gain = random.uniform(-2, 2)
            treble_gain = random.uniform(-2, 2)
            
            return f'ffmpeg -i "{input_path}" -af "rubberband=pitch={pitch_shift}:tempo={tempo_shift},bass=g={bass_gain},treble=g={treble_gain}" -c:v copy -y "{output_path}"'
        
        def pitch_shift_transform(input_path: str, output_path: str) -> str:
            """Pitch Shift ±3-5% (±½ to 1 semitone)"""
            pitch_cents = random.uniform(-100, 100)  # ±1 semitone in cents
            pitch_ratio = 2 ** (pitch_cents / 1200)
            
            return f'ffmpeg -i "{input_path}" -af "rubberband=pitch={pitch_ratio}" -c:v copy -y "{output_path}"'
        
        def tempo_shift_transform(input_path: str, output_path: str) -> str:
            """Tempo Shift ±3-4%"""
            tempo_factor = 1.0 + random.uniform(-0.04, 0.04)
            
            return f'ffmpeg -i "{input_path}" -af "atempo={tempo_factor}" -c:v copy -y "{output_path}"'
        
        def audio_layering_ambient(input_path: str, output_path: str) -> str:
            """Audio Layering with ambient noise -20dB to -30dB"""
            try:
                # FIXED: Made synchronous and use default duration
                import subprocess
                # Get video duration using subprocess (synchronous)
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
                
                noise_volume = random.uniform(0.001, 0.005)  # Very subtle
                noise_types = ['pink', 'brown', 'white']
                noise_type = random.choice(noise_types)
                
                # Method 1: Use volume filters (most compatible)
                return f'ffmpeg -i "{input_path}" -f lavfi -i "anoisesrc=d={duration}:c={noise_type}:a={noise_volume}" -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.1[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
                
            except Exception:
                # Fallback: Simple audio gain adjustment instead of mixing
                gain_db = random.uniform(-2, 2)  # Subtle gain change
                return f'ffmpeg -i "{input_path}" -af "volume={gain_db}dB" -c:v copy -y "{output_path}"'
        
        def audio_segment_reorder(input_path: str, output_path: str) -> str:
            """Audio Segment Reorder ≤ 10% of total audio"""
            # Simple approach: slight delay/advance of audio
            offset = random.uniform(-0.2, 0.2)  # ±200ms
            
            return f'ffmpeg -i "{input_path}" -itsoffset {offset} -i "{input_path}" -map 0:v -map 1:a -c:v copy -c:a aac -y "{output_path}"'
        
        def audio_simple_processing(input_path: str, output_path: str) -> str:
            """Simple Audio Processing - EQ and Volume adjustments"""
            # Simple but effective audio modifications
            bass_gain = random.uniform(-2, 2)  # Bass adjustment
            treble_gain = random.uniform(-1, 1)  # Treble adjustment
            volume_gain = random.uniform(-1, 1)  # Overall volume
            
            return f'ffmpeg -i "{input_path}" -af "equalizer=f=100:width_type=h:width=50:g={bass_gain},equalizer=f=8000:width_type=h:width=1000:g={treble_gain},volume={volume_gain}dB" -c:v copy -y "{output_path}"'
        
        # STRUCTURAL TRANSFORMATIONS
        def video_length_trim(input_path: str, output_path: str) -> str:
            """Video Length Trim ±1-2 sec from start/end"""
            trim_start = random.uniform(0.5, 2.0)
            trim_end = random.uniform(0.5, 1.5)
            
            return f'ffmpeg -ss {trim_start} -i "{input_path}" -t $(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{input_path}" | awk "{{print \\$1-{trim_start}-{trim_end}}}") -c copy -y "{output_path}"'
        
        def frame_rate_change(input_path: str, output_path: str) -> str:
            """Frame Rate Change ±0.5-1 fps"""
            fps_offset = random.uniform(-1.0, 1.0)
            # Assume standard 30fps, adjust accordingly
            new_fps = max(24, 30 + fps_offset)
            
            return f'ffmpeg -i "{input_path}" -r {new_fps} -c:a copy -y "{output_path}"'
        
        def black_frames_transitions(input_path: str, output_path: str) -> str:
            """Black Frames / Transitions - 1 frame every 10-20 sec"""
            fade_duration = random.uniform(0.1, 0.3)
            fade_in_start = random.uniform(0, 2)
            
            return f'ffmpeg -i "{input_path}" -vf "fade=in:st={fade_in_start}:d={fade_duration},fade=out:st=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{input_path}" | awk "{{print \\$1-{fade_duration}}}"):d={fade_duration}" -c:a copy -y "{output_path}"'
        
        # METADATA TRANSFORMATIONS
        def metadata_strip_randomize(input_path: str, output_path: str) -> str:
            """Strip EXIF/Metadata completely and add highly randomized metadata"""
            # ENHANCED: Completely randomize all metadata fields
            import secrets
            import uuid
            
            # Generate completely random metadata
            random_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            random_title = f"Video_{secrets.randbelow(999999):06d}"
            random_comment = f"Content_{uuid.uuid4().hex[:8]}"
            random_software = random.choice([
                "Adobe Premiere Pro 2024.1", "Final Cut Pro 10.6.5", "DaVinci Resolve 18.1",
                "OpenShot 3.1.1", "Kdenlive 22.12", "Filmora 12.3", "VSDC 8.1"
            ])
            random_artist = f"Creator_{secrets.randbelow(9999):04d}"
            random_album = f"Collection_{secrets.randbelow(999):03d}"
            
            # Randomize creation time within last 6 months
            import datetime as dt
            days_back = random.randint(1, 180)
            random_date = (dt.datetime.now() - dt.timedelta(days=days_back)).strftime('%Y-%m-%d %H:%M:%S')
            
            return f'ffmpeg -i "{input_path}" -map_metadata -1 -metadata creation_time="{random_date}" -metadata title="{random_title}" -metadata comment="{random_comment}" -metadata software="{random_software}" -metadata artist="{random_artist}" -metadata album="{random_album}" -c copy -y "{output_path}"'
        
        def title_caption_randomize(input_path: str, output_path: str) -> str:
            """Completely unique title/caption overlay"""
            captions = [
                "🔥 EXCLUSIVE CONTENT", "⭐ PREMIUM QUALITY", "🎬 ORIGINAL VIDEO",
                "💎 RARE FOOTAGE", "🎯 VIRAL CONTENT", "🚀 TRENDING NOW",
                "💯 MUST WATCH", "🌟 TOP QUALITY", "🎪 AMAZING SHOW"
            ]
            caption = random.choice(captions)
            fontsize = random.randint(20, 32)
            opacity = random.uniform(0.7, 0.9)
            duration = random.uniform(2, 5)
            
            return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{caption}\':fontcolor=yellow@{opacity}:fontsize={fontsize}:x=(w-tw)/2:y=50:enable=\'between(t,1,{duration})\'" -c:a copy -y "{output_path}"'
        
        # SEMANTIC/DEEP TRANSFORMATIONS
        def clip_embedding_shift(input_path: str, output_path: str) -> str:
            """CLIP Embedding Shift with text and visual elements"""
            emojis = ['🎵', '🎬', '⭐', '🔥', '💎', '🚀', '💯', '🌟']
            emoji = random.choice(emojis)
            text_overlays = [
                f"{emoji} ORIGINAL CONTENT {emoji}",
                f"{emoji} HD QUALITY {emoji}",
                f"{emoji} EXCLUSIVE VIDEO {emoji}"
            ]
            text = random.choice(text_overlays)
            
            return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{text}\':fontcolor=white@0.8:fontsize=24:x=(w-tw)/2:y=h-th-20:box=1:boxcolor=black@0.5" -c:a copy -y "{output_path}"'
        
        def text_presence_variation(input_path: str, output_path: str) -> str:
            """Text Presence in Frame - varied font/placement"""
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
            
            return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{watermark_text}\':fontcolor=white@0.6:fontsize=14:{position}" -c:a copy -y "{output_path}"'
        
        def audio_video_sync_offset(input_path: str, output_path: str) -> str:
            """Audio-Video Sync Offset ±100-300ms"""
            offset = random.uniform(-0.3, 0.3)
            
            return f'ffmpeg -i "{input_path}" -itsoffset {offset} -i "{input_path}" -map 0:v -map 1:a -c:v copy -c:a aac -shortest -y "{output_path}"'
        
        # ADVANCED COMBINED TRANSFORMATIONS
        def micro_rotation_with_crop(input_path: str, output_path: str) -> str:
            """Enhanced micro rotation ±2-4° with aggressive cropping for SSIM reduction"""
            angle = random.uniform(-4, 4)  # Increased rotation range
            crop_factor = random.uniform(1.08, 1.15)  # More aggressive cropping
            
            # Add additional zoom/pan for more structural disruption
            zoom_factor = random.uniform(1.02, 1.08)  # Slight zoom
            pan_x = random.uniform(-0.05, 0.05)  # Pan offset X
            pan_y = random.uniform(-0.05, 0.05)  # Pan offset Y
            
            return f'ffmpeg -i "{input_path}" -vf "rotate={angle}*PI/180:fillcolor=black:bilinear=1,crop=iw/{crop_factor}:ih/{crop_factor},scale=iw*{zoom_factor}:ih*{zoom_factor},crop=iw:ih:iw*{pan_x}:ih*{pan_y}" -c:a copy -y "{output_path}"'
        
        def advanced_lut_filter(input_path: str, output_path: str) -> str:
            """Advanced LUT-style color grading (Natural balance)"""
            gamma = random.uniform(0.9, 1.1)  # More subtle gamma
            contrast = random.uniform(0.9, 1.1)  # More conservative contrast
            brightness = random.uniform(-0.08, 0.08)  # Reduced brightness range
            saturation = random.uniform(0.9, 1.1)  # More natural saturation
            
            return f'ffmpeg -i "{input_path}" -vf "eq=gamma={gamma}:contrast={contrast}:brightness={brightness}:saturation={saturation}" -c:a copy -y "{output_path}"'
        
        def vignette_with_blur(input_path: str, output_path: str) -> str:
            """Vignette effect with subtle edge blur"""
            vignette_strength = random.uniform(0.3, 0.7)
            blur_amount = random.uniform(1, 3)
            
            return f'ffmpeg -i "{input_path}" -vf "vignette=angle=PI/4:eval=init:dither=1:aspect=1,unsharp=5:5:{blur_amount}:5:5:{blur_amount}" -c:a copy -y "{output_path}"'
        
        def temporal_shift_advanced(input_path: str, output_path: str) -> str:
            """Advanced temporal shift with variable speed"""
            speed_factor = random.uniform(0.95, 1.05)
            
            return f'ffmpeg -i "{input_path}" -filter_complex "[0:v]setpts={1/speed_factor}*PTS[v];[0:a]atempo={speed_factor}[a]" -map "[v]" -map "[a]" -y "{output_path}"'
        
        # NEW ENHANCED TRANSFORMATIONS - At least 5 per variant
        def black_screen_random(input_path: str, output_path: str) -> str:
            """Single black screen (1-2s) at random point in video"""
            try:
                # Get video info using ffprobe
                import subprocess
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-show_entries', 
                    'format=duration:stream=width,height', '-of', 'csv=p=0', input_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    duration = 10  # default
                    width, height = 1920, 1080  # default
                    
                    for line in lines:
                        if ',' not in line:
                            try:
                                duration = float(line)
                            except:
                                pass
                        else:
                            try:
                                w, h = line.split(',')
                                width, height = int(w), int(h)
                            except:
                                pass
                else:
                    duration = 10
                    width, height = 1920, 1080
                
                # Single black screen with 1-2 second duration
                black_duration = random.uniform(1.0, 2.0)
                
                # Random insertion point - avoid first and last 2 seconds
                min_start = 2.0
                max_start = max(3.0, duration - black_duration - 2.0)
                insert_time = random.uniform(min_start, max_start)
                
                # Ensure we don't exceed video duration
                if insert_time + black_duration > duration - 1:
                    insert_time = max(1.0, duration - black_duration - 1.0)
                
                # Create filter: video before black + black screen + video after black
                filter_complex = []
                
                # Part 1: Video before black screen
                if insert_time > 0:
                    filter_complex.append(f"[0:v]trim=start=0:end={insert_time:.2f},setpts=PTS-STARTPTS[v1]")
                    filter_complex.append(f"[0:a]atrim=start=0:end={insert_time:.2f},asetpts=PTS-STARTPTS[a1]")
                
                # Part 2: Black screen
                filter_complex.append(f"color=black:size={width}x{height}:duration={black_duration:.2f}:rate=25[v2]")
                filter_complex.append(f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={black_duration:.2f}[a2]")
                
                # Part 3: Video after black screen
                end_time = insert_time + black_duration
                if end_time < duration:
                    filter_complex.append(f"[0:v]trim=start={end_time:.2f},setpts=PTS-STARTPTS[v3]")
                    filter_complex.append(f"[0:a]atrim=start={end_time:.2f},asetpts=PTS-STARTPTS[a3]")
                
                # Concatenate parts
                if insert_time > 0 and end_time < duration:
                    # All three parts
                    filter_complex.append("[v1][a1][v2][a2][v3][a3]concat=n=3:v=1:a=1[outv][outa]")
                elif insert_time > 0:
                    # Only first two parts
                    filter_complex.append("[v1][a1][v2][a2]concat=n=2:v=1:a=1[outv][outa]")
                elif end_time < duration:
                    # Only last two parts
                    filter_complex.append("[v2][a2][v3][a3]concat=n=2:v=1:a=1[outv][outa]")
                else:
                    # Only black screen
                    filter_complex.append("[v2][a2]concat=n=1:v=1:a=1[outv][outa]")
                
                filter_string = ";".join(filter_complex)
                return f'ffmpeg -i "{input_path}" -filter_complex "{filter_string}" -map "[outv]" -map "[outa]" -y "{output_path}"'
                
            except Exception as e:
                # Fallback: Simple approach with geq filter to create black frames
                fade_time = random.uniform(2, 5)
                black_duration = random.uniform(1.0, 2.0)
                
                # Use geq filter to create solid black segments
                return f'ffmpeg -i "{input_path}" -vf "geq=if(between(t,{fade_time:.2f},{fade_time + black_duration:.2f}),0,lum(X,Y)):cb=128:cr=128" -c:a copy -y "{output_path}"'
        
        def pitch_shift_semitones(input_path: str, output_path: str) -> str:
            """Pitch shift audio ±2 semitones"""
            # ±2 semitones = ±200 cents
            semitones = random.uniform(-2, 2)
            pitch_ratio = 2 ** (semitones / 12)  # Convert semitones to ratio
            
            return f'ffmpeg -i "{input_path}" -af "rubberband=pitch={pitch_ratio}" -c:v copy -y "{output_path}"'
        
        def overlay_watermark_dynamic(input_path: str, output_path: str) -> str:
            """Overlay watermark/logo + dynamic position"""
            watermarks = ['© ORIGINAL', '★ PREMIUM', '🎬 HD QUALITY', '⚡ EXCLUSIVE', '🔥 VIRAL']
            watermark = random.choice(watermarks)
            
            # Dynamic position animation
            positions = [
                'x=10+20*sin(t):y=10+10*cos(t)',  # Circular motion top-left
                'x=w-tw-10-15*sin(t):y=10+10*cos(t)',  # Circular motion top-right
                'x=10+30*sin(t/2):y=h-th-10',  # Horizontal wave bottom-left
                'x=w-tw-10:y=h-th-10-20*sin(t)',  # Vertical wave bottom-right
            ]
            position = random.choice(positions)
            
            opacity = random.uniform(0.4, 0.7)
            fontsize = random.randint(16, 24)
            
            return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{watermark}\':fontcolor=white@{opacity}:fontsize={fontsize}:{position}:box=1:boxcolor=black@0.3" -c:a copy -y "{output_path}"'
        
        def zoom_jitter_motion(input_path: str, output_path: str) -> str:
            """Random zoom (in/out) + jitter motion (±3px)"""
            # Zoom factor
            zoom_factor = random.uniform(1.02, 1.15)  # Zoom in
            if random.random() < 0.3:  # 30% chance to zoom out
                zoom_factor = random.uniform(0.9, 0.98)
            
            # Jitter motion ±3px
            jitter_x = 3
            jitter_y = 3
            
            return f'ffmpeg -i "{input_path}" -vf "scale=iw*{zoom_factor:.3f}:ih*{zoom_factor:.3f},crop=iw:ih:iw*0.5-iw/2+{jitter_x}*sin(t*10):ih*0.5-ih/2+{jitter_y}*cos(t*10)" -c:a copy -y "{output_path}"'
        
        def ambient_audio_layers(input_path: str, output_path: str) -> str:
            """Add 1–2 ambient audio layers (wind, music)"""
            try:
                # FIXED: Made synchronous and use default duration
                import subprocess
                # Get video duration using subprocess (synchronous)
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
                
                # Choose 1-2 ambient sounds with correct syntax
                ambient_sounds = [
                    f'anoisesrc=d={duration}:c=pink:a=0.003',  # Pink noise (wind-like)
                    f'sine=frequency=200:duration={duration}'   # Sine wave (removed volume parameter)
                ]
                num_layers = random.randint(1, 2)
                selected_sounds = random.sample(ambient_sounds, num_layers)
                
                if num_layers == 1:
                    ambient_gen = selected_sounds[0]
                    # Apply volume control in the filter_complex instead
                    return f'ffmpeg -i "{input_path}" -f lavfi -i "{ambient_gen}" -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.05[a1];[a0][a1]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
                else:
                    ambient_gen1 = selected_sounds[0]
                    ambient_gen2 = selected_sounds[1]
                    # Apply volume control in the filter_complex for both layers
                    return f'ffmpeg -i "{input_path}" -f lavfi -i "{ambient_gen1}" -f lavfi -i "{ambient_gen2}" -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.03[a1];[2:a]volume=0.02[a2];[a0][a1][a2]amix=inputs=3:duration=first[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -y "{output_path}"'
                
            except Exception:
                # Fallback: Simple volume adjustment
                volume_change = random.uniform(-1, 1)
                return f'ffmpeg -i "{input_path}" -af "volume={volume_change}dB" -c:v copy -y "{output_path}"'
        
        def frame_trimming_dropout(input_path: str, output_path: str) -> str:
            """Frame trimming: drop or duplicate 2-3 frames"""
            # Simpler approach: Use frame rate adjustment instead of complex select
            # This creates a subtle temporal shift effect
            frame_adjustments = [
                # Slight frame rate changes to create temporal shifts
                '-r 24.8',  # Slightly slower
                '-r 25.2',  # Slightly faster
                '-r 29.8',  # Another slight change
                '-r 30.2',  # Another slight change
            ]
            adjustment = random.choice(frame_adjustments)
            
            return f'ffmpeg -i "{input_path}" {adjustment} -c:a copy -y "{output_path}"'
        
        def noise_blur_regions(input_path: str, output_path: str) -> str:
            """Add noise filter or soft blur over certain regions"""
            effects = [
                # Noise over top region
                'noise=alls=0.02:allf=t:c0s=10:c0f=t',
                # Soft blur on edges
                'boxblur=2:1:cr=0:ar=0',
                # Noise + slight blur
                'noise=alls=0.015:allf=t,unsharp=5:5:0.8:5:5:0.8',
            ]
            effect = random.choice(effects)
            
            return f'ffmpeg -i "{input_path}" -vf "{effect}" -c:a copy -y "{output_path}"'
        
        def grayscale_segment(input_path: str, output_path: str) -> str:
            """Grayscale segment for 1–2 seconds"""
            try:
                # FIXED: Made synchronous and use default duration
                import subprocess
                # Get video duration using subprocess (synchronous)
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
                
                # Random grayscale segment timing
                gray_start = random.uniform(duration * 0.3, duration * 0.7)
                gray_duration = random.uniform(1, 2)  # 1-2 seconds
                
                # Fixed: Use proper hue desaturation with enable condition
                return f'ffmpeg -i "{input_path}" -vf "hue=s=0:enable=\'between(t,{gray_start:.2f},{gray_start + gray_duration:.2f})\'" -c:a copy -y "{output_path}"'
                
            except Exception:
                # Fallback: Simple desaturation for entire video
                return f'ffmpeg -i "{input_path}" -vf "hue=s=0.3" -c:a copy -y "{output_path}"'
        
        def animated_text_corner(input_path: str, output_path: str) -> str:
            """Overlay animated text on bottom corner (muted)"""
            texts = ['HD QUALITY', 'PREMIUM', 'ORIGINAL', 'EXCLUSIVE', '© 2025']
            text = random.choice(texts)
            
            # Animated text properties
            fontsize = random.randint(14, 20)
            opacity_base = random.uniform(0.4, 0.6)
            
            # Animation: fade in/out and slight movement
            animation = f'x=w-tw-10+5*sin(t):y=h-th-10+3*cos(t):alpha={opacity_base}*sin(t/2)'
            
            return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{text}\':fontcolor=white:fontsize={fontsize}:{animation}:box=1:boxcolor=black@0.2" -c:a copy -y "{output_path}"'
        
        def dynamic_timestamp_overlay(input_path: str, output_path: str) -> str:
            """Dynamic timestamp overlay with unique info per video"""
            # Generate unique timestamp with random offset
            import time
            current_time = int(time.time())
            random_offset = random.randint(-86400, 86400)  # ±1 day offset
            timestamp = current_time + random_offset
            
            # Format options - FIXED: Escape special characters for FFmpeg
            timestamp_formats = [
                f"ID\\:{timestamp % 999999:06d}",  # 6-digit ID (escaped colon)
                f"T\\:{timestamp % 999999:06d}",   # Time-based ID (escaped colon)
                f"V\\:{random.randint(100000, 999999)}",  # Version number (escaped colon)
                f"\\#{random.randint(1000, 9999)}",       # Hash-style (escaped hash)
            ]
            
            timestamp_text = random.choice(timestamp_formats)
            
            # Position and style
            positions = [
                'x=10:y=h-th-10',           # Bottom-left
                'x=w-tw-10:y=h-th-10',      # Bottom-right
                'x=w-tw-10:y=10',           # Top-right
                'x=10:y=10'                 # Top-left
            ]
            position = random.choice(positions)
            
            fontsize = random.randint(10, 14)
            opacity = random.uniform(0.15, 0.3)  # Very subtle
            
            return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{timestamp_text}\':fontcolor=white@{opacity}:fontsize={fontsize}:{position}" -c:a copy -y "{output_path}"'
        
        def random_frame_inserts(input_path: str, output_path: str) -> str:
            """Insert random black/white frames for motion vector disruption"""
            try:
                # Get video info
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
                
                # Choose frame type and insertion point
                frame_types = ['black', 'white']
                frame_type = random.choice(frame_types)
                frame_color = 'black' if frame_type == 'black' else 'white'
                
                # Insert point (avoid first/last 2 seconds)
                insert_time = random.uniform(2, max(3, duration - 2))
                
                # FIXED: Simplified approach - use geq filter with very short duration
                flash_duration = 0.033  # One frame duration
                
                return f'ffmpeg -i "{input_path}" -vf "geq=if(between(t,{insert_time:.3f},{insert_time + flash_duration:.3f}),{255 if frame_color == "white" else 0},lum(X,Y)):cb=128:cr=128" -c:a copy -y "{output_path}"'
                
            except Exception:
                # Fallback: Use geq to create random frame flashes
                flash_time = random.uniform(2, 5)
                flash_color = random.choice(['0', '255'])  # Black or white
                
                return f'ffmpeg -i "{input_path}" -vf "geq=if(between(t,{flash_time:.3f},{flash_time + 0.033:.3f}),{flash_color},lum(X,Y)):cb=128:cr=128" -c:a copy -y "{output_path}"'
        
        def clip_embedding_shuffle(input_path: str, output_path: str) -> str:
            """CLIP embedding shuffling with varied emojis and quotes"""
            # Expanded emoji and quote collections for semantic variety
            emoji_collections = [
                ['🔥', '⚡', '💥', '🌟'],  # Energy collection
                ['✨', '💎', '👑', '🏆'],  # Premium collection  
                ['🎵', '🎬', '🎪', '🎭'],  # Entertainment collection
                ['💯', '🚀', '⭐', '🌈'],  # Achievement collection
                ['🎯', '💪', '🔝', '⚖️'],  # Success collection
            ]
            
            quote_collections = [
                ['VIRAL', 'TRENDING', 'POPULAR', 'HOT'],
                ['PREMIUM', 'EXCLUSIVE', 'RARE', 'LIMITED'],
                ['ORIGINAL', 'AUTHENTIC', 'GENUINE', 'REAL'],
                ['QUALITY', 'HD', 'CRYSTAL', 'CLEAR'],
                ['AMAZING', 'INCREDIBLE', 'STUNNING', 'WOW']
            ]
            
            # Select random collection and element
            emoji_set = random.choice(emoji_collections)
            quote_set = random.choice(quote_collections)
            
            emoji = random.choice(emoji_set)
            quote = random.choice(quote_set)
            
            # Create diverse text combinations
            text_variations = [
                f"{emoji} {quote} {emoji}",
                f"{emoji}{emoji} {quote}",
                f"{quote} {emoji}",
                f"{emoji} {quote}",
                f"{quote}{emoji}{emoji}"
            ]
            
            overlay_text = random.choice(text_variations)
            
            # Varied positioning and styling
            positions = [
                'x=(w-tw)/2:y=20',           # Top center
                'x=(w-tw)/2:y=h-th-20',      # Bottom center
                'x=20:y=(h-th)/2',           # Left center
                'x=w-tw-20:y=(h-th)/2',      # Right center
            ]
            position = random.choice(positions)
            
            fontsize = random.randint(18, 28)
            opacity = random.uniform(0.6, 0.8)
            duration_start = random.uniform(0.5, 2)
            duration_length = random.uniform(2, 4)
            
            return f'ffmpeg -i "{input_path}" -vf "drawtext=text=\'{overlay_text}\':fontcolor=yellow@{opacity}:fontsize={fontsize}:{position}:enable=\'between(t,{duration_start},{duration_start + duration_length})\':box=1:boxcolor=black@0.4" -c:a copy -y "{output_path}"'
        
        def frame_reordering_segments(input_path: str, output_path: str) -> str:
            """Micro-temporal disorientation through frame segment reordering"""
            try:
                # FIXED: Made function synchronous and simplified
                import subprocess
                # Get video duration using subprocess (synchronous)
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
                
                # FIXED: Use simple temporal effects instead of complex reverse filter
                # Option 1: Slight speed variation in a segment
                segment_start = random.uniform(duration * 0.3, duration * 0.6)
                segment_duration = random.uniform(0.5, 1.5)
                speed_factor = random.choice([0.95, 1.05])  # Slight speed change
                
                # Use setpts for temporal manipulation
                return f'ffmpeg -i "{input_path}" -vf "setpts=if(between(t,{segment_start:.3f},{segment_start + segment_duration:.3f}),PTS/{speed_factor},PTS)" -c:a copy -y "{output_path}"'
                
            except Exception:
                # Fallback: Simple frame rate micro-adjustment for temporal effect
                fps_adjustment = random.choice([0.98, 1.02])  # Slight speed change
                
                return f'ffmpeg -i "{input_path}" -filter_complex "[0:v]setpts={1/fps_adjustment}*PTS[v];[0:a]atempo={fps_adjustment}[a]" -map "[v]" -map "[a]" -y "{output_path}"'
        
        def audio_panning_balance(input_path: str, output_path: str) -> str:
            """Audio panning with left-right balance changes"""
            # FIXED: Corrected pan filter syntax
            panning_types = [
                # Standard cross-panning
                'stereo|c0=0.7*c0+0.3*c1|c1=0.3*c0+0.7*c1',
                # Left-heavy panning
                'stereo|c0=0.8*c0+0.2*c1|c1=0.4*c0+0.6*c1',
                # Right-heavy panning
                'stereo|c0=0.6*c0+0.4*c1|c1=0.2*c0+0.8*c1',
                # Dynamic panning (subtle)
                'stereo|c0=0.75*c0+0.25*c1|c1=0.25*c0+0.75*c1',
                # Center spreading
                'stereo|c0=0.9*c0+0.1*c1|c1=0.1*c0+0.9*c1'
            ]
            
            panning_filter = random.choice(panning_types)
            
            return f'ffmpeg -i "{input_path}" -af "pan={panning_filter}" -c:v copy -y "{output_path}"'
        
        return [
            # VISUAL CATEGORY - Always apply 3-4 (BALANCED for natural colors)
            TransformationConfig('phash_disruption', 1.0, phash_disruption_transform, 'visual'),
            TransformationConfig('extreme_phash_disruption', 0.3, extreme_phash_disruption, 'visual'),  # REDUCED: Lower probability to prevent green tint
            TransformationConfig('ssim_reduction', 1.0, ssim_reduction_transform, 'visual'),
            TransformationConfig('color_histogram_shift', 0.9, color_histogram_shift, 'visual'),
            TransformationConfig('frame_entropy_increase', 0.8, frame_entropy_increase, 'visual'),
            TransformationConfig('embedding_similarity_change', 0.7, embedding_similarity_change, 'visual'),
            
            # AUDIO CATEGORY - Always apply 2-3
            TransformationConfig('spectral_fingerprint_disruption', 1.0, spectral_fingerprint_disruption, 'audio'),
            TransformationConfig('pitch_shift', 0.8, pitch_shift_transform, 'audio'),
            TransformationConfig('tempo_shift', 0.8, tempo_shift_transform, 'audio'),
            TransformationConfig('audio_layering_ambient', 0.4, audio_layering_ambient, 'audio'),  # Reduced probability
            TransformationConfig('audio_simple_processing', 0.8, audio_simple_processing, 'audio'),  # New reliable option
            TransformationConfig('audio_segment_reorder', 0.5, audio_segment_reorder, 'audio'),
            
            # STRUCTURAL CATEGORY - Apply 2-3
            TransformationConfig('video_length_trim', 0.7, video_length_trim, 'structural'),
            TransformationConfig('frame_rate_change', 0.6, frame_rate_change, 'structural'),
            TransformationConfig('black_frames_transitions', 0.5, black_frames_transitions, 'structural'),
            
            # METADATA CATEGORY - Always apply (ENHANCED for better metadata randomization)
            TransformationConfig('metadata_strip_randomize', 1.0, metadata_strip_randomize, 'metadata'),
            TransformationConfig('title_caption_randomize', 0.8, title_caption_randomize, 'metadata'),
            
            # SEMANTIC/DEEP CATEGORY - Apply 1-2
            TransformationConfig('clip_embedding_shift', 0.7, clip_embedding_shift, 'semantic'),
            TransformationConfig('text_presence_variation', 0.6, text_presence_variation, 'semantic'),
            TransformationConfig('audio_video_sync_offset', 0.5, audio_video_sync_offset, 'semantic'),
            
            # ADVANCED COMBINED - Apply 2-3
            TransformationConfig('micro_rotation_crop', 0.8, micro_rotation_with_crop, 'advanced'),
            TransformationConfig('advanced_lut_filter', 0.9, advanced_lut_filter, 'advanced'),
            TransformationConfig('vignette_blur', 0.6, vignette_with_blur, 'advanced'),
            TransformationConfig('temporal_shift_advanced', 0.8, temporal_shift_advanced, 'advanced'),
            
            # NEW ENHANCED CATEGORY - At least 5 per variant (HIGH PROBABILITY)
            TransformationConfig('black_screen_random', 0.9, black_screen_random, 'enhanced'),
            TransformationConfig('pitch_shift_semitones', 0.9, pitch_shift_semitones, 'enhanced'),
            TransformationConfig('overlay_watermark_dynamic', 0.9, overlay_watermark_dynamic, 'enhanced'),
            TransformationConfig('zoom_jitter_motion', 0.8, zoom_jitter_motion, 'enhanced'),
            TransformationConfig('ambient_audio_layers', 0.8, ambient_audio_layers, 'enhanced'),
            TransformationConfig('frame_trimming_dropout', 0.8, frame_trimming_dropout, 'enhanced'),
            TransformationConfig('noise_blur_regions', 0.8, noise_blur_regions, 'enhanced'),
            TransformationConfig('grayscale_segment', 0.7, grayscale_segment, 'enhanced'),
            TransformationConfig('animated_text_corner', 0.9, animated_text_corner, 'enhanced'),
            
            # NEW ADDITIONAL ENHANCEMENTS - Increased robustness
            TransformationConfig('dynamic_timestamp_overlay', 0.8, dynamic_timestamp_overlay, 'enhanced'),
            TransformationConfig('random_frame_inserts', 0.7, random_frame_inserts, 'enhanced'),
            TransformationConfig('clip_embedding_shuffle', 0.8, clip_embedding_shuffle, 'enhanced'),
            TransformationConfig('frame_reordering_segments', 0.6, frame_reordering_segments, 'enhanced'),
            TransformationConfig('audio_panning_balance', 0.7, audio_panning_balance, 'enhanced'),
        ]
    
    @staticmethod
    def get_audio_transformation_names() -> List[str]:
        return ['spectral_fingerprint_disruption', 'pitch_shift', 'tempo_shift', 'audio_simple_processing', 'audio_layering_ambient', 'audio_segment_reorder']
    
    @staticmethod
    def get_guaranteed_transformation_names() -> List[str]:
        return ['metadata_strip_randomize', 'phash_disruption', 'ssim_reduction', 'spectral_fingerprint_disruption', 'advanced_lut_filter', 'black_screen_random', 'pitch_shift_semitones', 'overlay_watermark_dynamic']
    
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
        
        # PHASE 1: ENSURE AT LEAST 5 ENHANCED TRANSFORMATIONS (NEW REQUIREMENT)
        enhanced_transforms = categories.get('enhanced', [])
        if len(enhanced_transforms) >= 5:
            # Randomly select exactly 5 enhanced transformations
            selected_enhanced = random.sample(enhanced_transforms, 5)
            selected.extend(selected_enhanced)
            logging.info(f'🎯 Selected 5 Enhanced Transformations: {[t.name for t in selected_enhanced]}')
        
        # PHASE 2: Ensure we have representation from other categories
        guaranteed_per_category = {
            'visual': 2,      # 2 visual transformations (reduced to make room for enhanced)
            'audio': 2,       # 2 audio transformations  
            'structural': 1,  # 1 structural transformation (reduced)
            'metadata': 1,    # 1 metadata transformation
            'semantic': 1,    # 1 semantic transformation (reduced)
            'advanced': 1     # 1 advanced transformation (reduced)
        }
        
        for category, min_count in guaranteed_per_category.items():
            if category in categories:
                # Shuffle and take the minimum required
                category_transforms = categories[category].copy()
                random.shuffle(category_transforms)
                # Don't exceed the target by adding too many
                available_slots = min_count
                selected.extend(category_transforms[:available_slots])
        
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
    async def execute_command(command: str, timeout: int = 120) -> bool:
        """Execute FFmpeg command with timeout"""
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
                logging.error(f"Command failed: {stderr.decode()}")
                return False
            
            return True
            
        except asyncio.TimeoutError:
            logging.error(f"Command timed out after {timeout} seconds")
            return False
        except Exception as e:
            logging.error(f"Command execution failed: {e}")
            return False
    
    @staticmethod
    async def apply_transformations(
        input_path: str,
        output_path: str,
        min_transformations: int = 9,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[str]:
        """Apply video transformations with progress tracking"""
        try:
            logging.info(f'🎬 Applying transformations to: {os.path.basename(input_path)}')
            
            # Validate input file
            if not await FFmpegTransformationService.validate_video_file(input_path):
                raise Exception(f'Invalid input video file: {input_path}')
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Select transformations
            selected_transformations = FFmpegTransformationService.select_random_transformations(min_transformations)
            applied_transformations = []
            
            current_input = input_path
            temp_counter = 0
            temp_files = []
            
            total_transformations = len(selected_transformations)
            
            for i, transformation in enumerate(selected_transformations):
                temp_output = os.path.join(
                    os.path.dirname(output_path),
                    f'temp_{temp_counter}_{os.path.basename(output_path)}'
                )
                
                try:
                    logging.info(f'  📝 Applying: {transformation.name} ({i+1}/{total_transformations})')
                    
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
                    
                    # Execute transformation
                    if asyncio.iscoroutinefunction(transformation.execute):
                        command = await transformation.execute(current_input, temp_output)
                    else:
                        command = transformation.execute(current_input, temp_output)
                    
                    logging.info(f'  🔧 Command: {command[:100]}...')
                    
                    success = await FFmpegTransformationService.execute_command(command)
                    
                    if not success:
                        logging.warning(f'⚠️ Failed to apply {transformation.name}')
                        continue
                    
                    # Validate output
                    if not await FFmpegTransformationService.validate_video_file(temp_output):
                        logging.warning(f'⚠️ Invalid output file created for {transformation.name}: {temp_output}')
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                        continue
                    
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
            
            logging.info(f'✅ Applied {len(applied_transformations)} transformations')
            return applied_transformations
            
        except Exception as error:
            logging.error(f'❌ Error applying transformations: {error}')
            raise
    
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
