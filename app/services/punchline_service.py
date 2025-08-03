import os
import json
import requests
from groq import Groq
import subprocess
import tempfile
import re
import random
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from decouple import config
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedVideoPunchlineGenerator:
    def __init__(self, cache_dir: Optional[str] = None):
        self.elevenlabs_key = config('ELEVENLABS_API_KEY', default='')
        self.groq_api_key = config('GROQ_API_KEY', default='')
        
        # Enhanced caching system
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / "punchline_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        if not self.elevenlabs_key or not self.groq_api_key:
            logger.warning("ElevenLabs or Groq API keys not configured. Punchline generation will be disabled.")
            return
            
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Enhanced styling variants with more customization
        self.variants = {
            "variant_1_minimal": {
                "bg_color": "black",
                "text_color": "white", 
                "font_size": 50,
                "border_color": "red",
                "border_width": 3,
                "duration": 1.2,
                "animation": "fade_in_out",
                "position": "center",
                "shadow": True
            },
            "variant_2_bold": {
                "bg_color": "0x000080",  # Navy blue
                "text_color": "yellow",
                "font_size": 55,
                "border_color": "white",
                "border_width": 4,
                "duration": 1.5,
                "animation": "slide_up",
                "position": "center",
                "shadow": True
            },
            "variant_3_modern": {
                "bg_color": "0x1a1a2e",  # Dark purple
                "text_color": "0x00ff88",  # Bright green
                "font_size": 48,
                "border_color": "0x16213e",
                "border_width": 2,
                "duration": 1.0,
                "animation": "zoom_in",
                "position": "bottom_third",
                "shadow": True
            },
            "variant_4_elegant": {
                "bg_color": "0x0f0f23",  # Very dark blue
                "text_color": "0xffd700",  # Gold
                "font_size": 52,
                "border_color": "0x4a4a4a",
                "border_width": 2,
                "duration": 1.3,
                "animation": "typewriter",
                "position": "center",
                "shadow": False
            },
            "variant_5_vibrant": {
                "bg_color": "0x2d1b69",  # Deep purple
                "text_color": "0xff6b6b",  # Coral red
                "font_size": 60,
                "border_color": "0x4ecdc4",  # Teal
                "border_width": 3,
                "duration": 1.1,
                "animation": "bounce_in",
                "position": "top_third",
                "shadow": True
            }
        }
        
        # Enhanced font options
        self.fonts = [
            "Arial-Bold",
            "Helvetica-Bold", 
            "Impact",
            "Futura-Bold",
            "Montserrat-Bold"
        ]
        
    def get_cache_key(self, content: str) -> str:
        """Generate cache key for content"""
        return hashlib.md5(content.encode()).hexdigest()
        
    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None
        
    def cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache result for future use"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
        
    def is_available(self) -> bool:
        """Check if punchline generation is available (API keys configured)"""
        return bool(self.elevenlabs_key and self.groq_api_key)
        
    async def extract_audio_async(self, video_path: str, audio_path: str) -> None:
        """Extract audio using FFmpeg with enhanced quality settings - async"""
        cmd = [
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '44100', '-ac', '2', 
            '-af', 'highpass=f=80,lowpass=f=8000,volume=2.0,dynaudnorm',  # Enhanced audio processing
            '-threads', '4',  # Multi-threading
            audio_path, '-y'
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg audio extraction failed: {stderr.decode()}")
        
    async def transcribe_audio_async(self, audio_file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Transcribe audio using ElevenLabs API with enhanced error handling"""
        if not self.elevenlabs_key:
            raise Exception("ElevenLabs API key not configured")
        
        # Check cache first
        cache_key = self.get_cache_key(f"transcribe_{os.path.basename(audio_file_path)}")
        cached = self.get_cached_result(cache_key)
        if cached:
            logger.info("Using cached transcription")
            return cached['text'], cached.get('alignment', {})
            
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        
        headers = {
            "xi-api-key": self.elevenlabs_key
        }
        
        # Enhanced transcription settings
        with open(audio_file_path, 'rb') as audio_file:
            files = {"file": audio_file}
            data = {
                "model_id": "scribe_v1",
                "output_format": "json",
                "language": "auto"  # Auto-detect language
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(url, headers=headers, files=files, data=data, timeout=120)
            )
            
        if response.status_code == 200:
            result = response.json()
            text = result.get('text', '')
            alignment = result.get('alignment', {})
            
            # Cache the result
            self.cache_result(cache_key, {'text': text, 'alignment': alignment})
            
            return text, alignment
        else:
            raise Exception(f"Transcription failed: {response.text}")
    
    def generate_enhanced_punchlines(self, transcript: str, variant_num: int, video_duration: float) -> List[Dict[str, str]]:
        """Generate punchlines with enhanced AI analysis and timing"""
        
        if not self.groq_api_key:
            raise Exception("Groq API key not configured")
        
        # Check cache first
        cache_key = self.get_cache_key(f"punchlines_{transcript}_{variant_num}")
        cached = self.get_cached_result(cache_key)
        if cached:
            logger.info("Using cached punchlines")
            return cached['punchlines']
            
        # Enhanced prompt with better instructions
        prompt = f"""
        Analyze this video transcript and extract the most impactful quotes for text overlays. 
        Focus on emotional peaks, key insights, and memorable moments.
        
        Video Duration: {video_duration:.1f} seconds
        Transcript: "{transcript}"
        
        Extract 6 powerful quotes following these rules:
        1. Use EXACT words from transcript (no paraphrasing)
        2. Prioritize emotional impact and memorability
        3. Keep quotes 3-8 words maximum
        4. Focus on action words, strong statements, and key insights
        5. Distribute timing evenly across video duration
        6. Avoid filler words and transitional phrases
        
        Format as JSON:
        {{
            "punchlines": [
                {{"text": "Most impactful quote", "suggested_timestamp": "0:05", "emotional_weight": "high", "type": "insight"}},
                {{"text": "Second powerful moment", "suggested_timestamp": "0:15", "emotional_weight": "medium", "type": "action"}},
                {{"text": "Third memorable quote", "suggested_timestamp": "0:25", "emotional_weight": "high", "type": "revelation"}},
                {{"text": "Fourth strong statement", "suggested_timestamp": "0:35", "emotional_weight": "medium", "type": "advice"}},
                {{"text": "Fifth impactful moment", "suggested_timestamp": "0:45", "emotional_weight": "high", "type": "conclusion"}},
                {{"text": "Sixth powerful quote", "suggested_timestamp": "0:55", "emotional_weight": "high", "type": "takeaway"}}
            ]
        }}
        
        Emotional weight: high/medium/low
        Type: insight/action/revelation/advice/conclusion/takeaway/question/statement
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-70b-versatile",  # More powerful model
                temperature=0.4,  # Lower for more consistent results
                max_tokens=1000
            )
            
            # Enhanced JSON parsing
            content = response.choices[0].message.content.strip()
            content = self._clean_json_response(content)
            
            result = json.loads(content)
            all_punchlines = result["punchlines"]
            
            # Smart variant distribution based on emotional weight and type
            selected_punchlines = self._select_punchlines_for_variant(all_punchlines, variant_num)
            
            # Cache the result
            self.cache_result(cache_key, {'punchlines': selected_punchlines})
            
            return selected_punchlines
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}, attempting fallback parsing")
            return self._fallback_punchline_parsing(content, variant_num, video_duration)
        except Exception as e:
            logger.error(f"Punchline generation failed: {e}")
            return self._generate_fallback_punchlines(variant_num, video_duration)
    
    def _clean_json_response(self, content: str) -> str:
        """Clean and prepare JSON response for parsing"""
        # Remove code blocks
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        # Fix common JSON issues
        content = re.sub(r',\s*}', '}', content)  # Remove trailing commas
        content = re.sub(r',\s*]', ']', content)  # Remove trailing commas in arrays
        
        return content
    
    def _select_punchlines_for_variant(self, all_punchlines: List[Dict], variant_num: int) -> List[Dict]:
        """Intelligently select punchlines based on variant and emotional weight"""
        if len(all_punchlines) < 4:
            return all_punchlines
        
        # Sort by emotional weight and type
        high_impact = [p for p in all_punchlines if p.get('emotional_weight') == 'high']
        medium_impact = [p for p in all_punchlines if p.get('emotional_weight') == 'medium']
        
        if variant_num == 1:
            # Variant 1: Focus on insights and revelations
            selected = [p for p in high_impact if p.get('type') in ['insight', 'revelation', 'takeaway']][:2]
            if len(selected) < 2:
                selected.extend(high_impact[:2-len(selected)])
        elif variant_num == 2:
            # Variant 2: Focus on action and advice
            selected = [p for p in all_punchlines if p.get('type') in ['action', 'advice', 'statement']][:2]
            if len(selected) < 2:
                selected.extend(medium_impact[:2-len(selected)])
        else:
            # Other variants: Mix of high and medium impact
            selected = high_impact[:1] + medium_impact[:1]
            if len(selected) < 2:
                selected.extend(all_punchlines[:2-len(selected)])
        
        return selected[:2]  # Limit to 2 per variant
    
    def _fallback_punchline_parsing(self, content: str, variant_num: int, duration: float) -> List[Dict]:
        """Fallback parsing when JSON fails"""
        punchlines = []
        lines = content.split('\n')
        
        for line in lines:
            text_match = re.search(r'"text":\s*"([^"]*)"', line)
            time_match = re.search(r'"suggested_timestamp":\s*"(\d+:\d+)"', line)
            
            if text_match:
                text = text_match.group(1)
                timestamp = time_match.group(1) if time_match else f"0:{len(punchlines)*15:02d}"
                
                punchlines.append({
                    "text": text,
                    "suggested_timestamp": timestamp,
                    "emotional_weight": "medium",
                    "type": "statement"
                })
        
        return self._select_punchlines_for_variant(punchlines, variant_num)
    
    def _generate_fallback_punchlines(self, variant_num: int, duration: float) -> List[Dict]:
        """Generate fallback punchlines when all else fails"""
        return [
            {"text": f"Key Moment {variant_num}-1", "suggested_timestamp": "0:10", "emotional_weight": "high", "type": "insight"},
            {"text": f"Important Point {variant_num}-2", "suggested_timestamp": f"0:{min(30, int(duration//2)):02d}", "emotional_weight": "medium", "type": "statement"}
        ]
    
    def time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS to seconds with decimal precision"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                return int(parts[0]) * 60 + int(parts[1])
            else:
                return float(time_str)
        except:
            return 5.0
    
    async def get_video_info_async(self, video_path: str) -> Tuple[str, float, float]:
        """Get video dimensions, duration, and framerate asynchronously"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return "1920x1080", 30.0, 30.0
        
        info = json.loads(stdout.decode())
        video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
        
        if video_stream:
            width = video_stream.get('width', 1920)
            height = video_stream.get('height', 1080)
            duration = float(info['format'].get('duration', 30.0))
            
            # Get frame rate
            r_frame_rate = video_stream.get('r_frame_rate', '30/1')
            if '/' in r_frame_rate:
                num, den = map(int, r_frame_rate.split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(r_frame_rate)
                
            return f"{width}x{height}", duration, fps
        
        return "1920x1080", 30.0, 30.0
    
    def create_enhanced_blackscreen(self, text: str, duration: float, output_path: str, 
                                  video_size: str, variant_style: Dict[str, Any]) -> None:
        """Create enhanced blackscreen with animations and better text handling"""
        
        # Enhanced text processing
        escaped_text = self._prepare_text_for_ffmpeg(text)
        
        width, height = map(int, video_size.split('x'))
        
        # Dynamic font sizing based on video dimensions and text length
        base_font_size = min(width // 15, height // 12)
        text_length_factor = max(0.7, 1.2 - (len(text) / 50))  # Smaller font for longer text
        font_size = int(base_font_size * text_length_factor)
        font_size = min(font_size, variant_style.get('font_size', 50))
        
        # Position calculations
        position = self._get_text_position(variant_style.get('position', 'center'), width, height)
        
        # Animation effects
        animation_filter = self._get_animation_filter(variant_style.get('animation', 'fade_in_out'), duration)
        
        # Shadow effect
        shadow_filter = ":shadowcolor=black:shadowx=2:shadowy=2" if variant_style.get('shadow', False) else ""
        
        # Select random font
        font_family = random.choice(self.fonts)
        
        # Build enhanced drawtext filter
        drawtext_filter = (
            f"drawtext=text='{escaped_text}'"
            f":fontfile=/System/Library/Fonts/Arial.ttf"  # Fallback font path
            f":fontsize={font_size}"
            f":fontcolor={variant_style['text_color']}"
            f":x={position['x']}"
            f":y={position['y']}"
            f":borderw={variant_style['border_width']}"
            f":bordercolor={variant_style['border_color']}"
            f"{shadow_filter}"
            f"{animation_filter}"
        )
        
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'color=c={variant_style["bg_color"]}:size={video_size}:duration={duration}:rate=30',
            '-f', 'lavfi', 
            '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}',
            '-vf', drawtext_filter,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'medium',  # Better quality than 'fast'
            '-crf', '20',  # Higher quality
            '-t', str(duration),
            '-pix_fmt', 'yuv420p',
            '-r', '30',
            '-movflags', '+faststart',  # Web optimization
            output_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Enhanced blackscreen creation failed, trying fallback: {result.stderr}")
            self._create_fallback_blackscreen(text, duration, output_path, video_size, variant_style)
    
    def _prepare_text_for_ffmpeg(self, text: str) -> str:
        """Enhanced text preparation for FFmpeg"""
        # Clean and escape text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Smart line breaking for better readability
        words = text.split(' ')
        if len(words) > 6:  # Long text needs breaking
            # Find natural break points
            mid = len(words) // 2
            # Look for conjunctions or punctuation near the middle
            for i in range(max(1, mid-2), min(len(words)-1, mid+3)):
                if words[i].lower() in ['and', 'but', 'or', 'so', 'yet', 'for']:
                    mid = i + 1
                    break
            
            line1 = ' '.join(words[:mid])
            line2 = ' '.join(words[mid:])
            text = f"{line1}\\n{line2}"
        
        # Escape special characters for FFmpeg
        text = text.replace("'", "\\'").replace(":", "\\:").replace(",", "\\,")
        text = text.replace("%", "\\%").replace("\\n", "\\n")
        
        return text
    
    def _get_text_position(self, position: str, width: int, height: int) -> Dict[str, str]:
        """Calculate text position based on style"""
        positions = {
            'center': {'x': '(w-text_w)/2', 'y': '(h-text_h)/2'},
            'top_third': {'x': '(w-text_w)/2', 'y': 'h/3-(text_h/2)'},
            'bottom_third': {'x': '(w-text_w)/2', 'y': '2*h/3-(text_h/2)'},
            'top': {'x': '(w-text_w)/2', 'y': '50'},
            'bottom': {'x': '(w-text_w)/2', 'y': 'h-text_h-50'}
        }
        return positions.get(position, positions['center'])
    
    def _get_animation_filter(self, animation: str, duration: float) -> str:
        """Generate animation filter based on type"""
        animations = {
            'fade_in_out': f":alpha='if(lt(t,0.3),t/0.3,if(gt(t,{duration-0.3}),(({duration}-t)/0.3),1))'",
            'slide_up': f":y='h-((t/{duration})*(h-y))'",
            'zoom_in': f":fontsize='fontsize*(0.5+0.5*(t/{duration}))'",
            'bounce_in': f":alpha='if(lt(t,0.5),pow(t/0.5,0.5),1)'",
            'typewriter': "",  # Would need complex implementation
            'none': ""
        }
        return animations.get(animation, animations['fade_in_out'])
    
    def _create_fallback_blackscreen(self, text: str, duration: float, output_path: str, 
                                   video_size: str, variant_style: Dict[str, Any]) -> None:
        """Fallback blackscreen creation with minimal effects"""
        escaped_text = text.replace("'", "\\'").replace(":", "\\:")
        
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'color=c=black:size={video_size}:duration={duration}',
            '-f', 'lavfi', 
            '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}',
            '-vf', f"drawtext=text='{escaped_text}':fontsize=40:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            '-c:v', 'libx264', '-c:a', 'aac', '-preset', 'fast', '-t', str(duration),
            output_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Fallback blackscreen creation failed: {result.stderr}")
    
    async def create_variant_with_punchlines_async(self, video_path: str, punchlines: List[Dict[str, str]], 
                                                 output_path: str, variant_style: Dict[str, Any]) -> str:
        """Create video with blackscreens asynchronously with enhanced processing"""
        
        video_size, total_duration, fps = await self.get_video_info_async(video_path)
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            segments = []
            segment_files = []
            current_time = 0.0
            
            # Process punchlines with better timing
            for i, punchline in enumerate(punchlines):
                timestamp = self.time_to_seconds(punchline["suggested_timestamp"])
                
                # Intelligent timestamp adjustment
                if timestamp >= total_duration - 2:
                    timestamp = max(total_duration - variant_style["duration"] - 1, current_time + 5)
                
                # Create video segment before punchline
                if timestamp > current_time + 0.1:  # Minimum segment length
                    segment_path = temp_dir / f"seg_{i}_video.mp4"
                    await self._create_video_segment_async(
                        video_path, current_time, timestamp - current_time, segment_path, fps
                    )
                    
                    if segment_path.exists():
                        segments.append(f"file '{segment_path}'")
                        segment_files.append(segment_path)
                
                # Create enhanced blackscreen segment
                blackscreen_path = temp_dir / f"seg_{i}_black.mp4"
                self.create_enhanced_blackscreen(
                    punchline["text"], 
                    variant_style["duration"], 
                    str(blackscreen_path), 
                    video_size,
                    variant_style
                )
                segments.append(f"file '{blackscreen_path}'")
                segment_files.append(blackscreen_path)
                
                current_time = timestamp + variant_style["duration"]
            
            # Add remaining video
            if current_time < total_duration - 0.5:
                final_segment_path = temp_dir / "seg_final.mp4"
                await self._create_video_segment_async(
                    video_path, current_time, total_duration - current_time, final_segment_path, fps
                )
                
                if final_segment_path.exists():
                    segments.append(f"file '{final_segment_path}'")
                    segment_files.append(final_segment_path)
            
            # Enhanced concatenation
            await self._concatenate_segments_async(segments, output_path, temp_dir, fps)
            
            logger.info(f"✅ Created enhanced punchline variant: {output_path}")
            return output_path
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def _create_video_segment_async(self, video_path: str, start_time: float, 
                                        duration: float, output_path: Path, fps: float) -> None:
        """Create video segment asynchronously with optimal settings"""
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264', '-c:a', 'aac',
            '-preset', 'medium',
            '-crf', '22',
            '-r', str(fps),
            '-avoid_negative_ts', 'make_zero',
            '-movflags', '+faststart',
            str(output_path), '-y'
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.warning(f"Video segment creation failed: {stderr.decode()}")
    
    async def _concatenate_segments_async(self, segments: List[str], output_path: str, 
                                        temp_dir: Path, fps: float) -> None:
        """Concatenate segments asynchronously with enhanced settings"""
        if not segments:
            raise Exception("No segments to concatenate")
        
        concat_file = temp_dir / 'concat.txt'
        with open(concat_file, 'w') as f:
            for segment in segments:
                f.write(segment + '\n')
        
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c:v', 'libx264', '-c:a', 'aac',
            '-preset', 'medium',
            '-crf', '22',
            '-r', str(fps),
            '-fflags', '+genpts',
            '-movflags', '+faststart',
            '-max_muxing_queue_size', '1024',  # Handle complex concatenation
            output_path, '-y'
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Enhanced concatenation failed: {stderr.decode()}")
    
    async def process_video_with_enhanced_punchlines(self, video_path: str, variant_num: int = 1) -> Dict[str, Any]:
        """Process video and add enhanced punchlines - returns comprehensive punchline data"""
        
        if not self.is_available():
            raise Exception("Enhanced punchline generation not available - API keys not configured")
        
        # Create temp audio file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_path = temp_audio.name
        temp_audio.close()
        
        try:
            logger.info("🎵 Extracting enhanced audio for transcription...")
            await self.extract_audio_async(video_path, audio_path)
            
            logger.info("📝 Transcribing audio with enhanced processing...")
            transcript, alignment = await self.transcribe_audio_async(audio_path)
            
            # Get video info for better timing
            video_size, duration, fps = await self.get_video_info_async(video_path)
            
            logger.info(f"🎯 Generating enhanced punchlines for variant {variant_num}...")
            punchlines = self.generate_enhanced_punchlines(transcript, variant_num, duration)
            
            # Select appropriate variant style
            variant_key = f"variant_{variant_num}_minimal" if variant_num <= 5 else "variant_1_minimal"
            variant_style = self.variants.get(variant_key, self.variants["variant_1_minimal"])
            
            # Enhanced quality metrics
            quality_metrics = self._analyze_punchline_quality(punchlines, transcript)
            
            return {
                'transcript': transcript,
                'punchlines': punchlines,
                'style': variant_style,
                'alignment': alignment,
                'video_info': {
                    'size': video_size,
                    'duration': duration,
                    'fps': fps
                },
                'quality_metrics': quality_metrics,
                'processing_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'variant_num': variant_num,
                    'api_versions': {
                        'elevenlabs': 'v1',
                        'groq': 'llama-3.1-70b-versatile'
                    }
                }
            }
            
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def _analyze_punchline_quality(self, punchlines: List[Dict], transcript: str) -> Dict[str, Any]:
        """Analyze and score punchline quality"""
        if not punchlines:
            return {'overall_score': 0, 'metrics': {}}
        
        metrics = {
            'word_count_avg': sum(len(p['text'].split()) for p in punchlines) / len(punchlines),
            'emotional_weight_distribution': {},
            'type_distribution': {},
            'timing_spread': 0,
            'transcript_coverage': 0
        }
        
        # Analyze emotional weight distribution
        for p in punchlines:
            weight = p.get('emotional_weight', 'unknown')
            metrics['emotional_weight_distribution'][weight] = metrics['emotional_weight_distribution'].get(weight, 0) + 1
        
        # Analyze type distribution
        for p in punchlines:
            ptype = p.get('type', 'unknown')
            metrics['type_distribution'][ptype] = metrics['type_distribution'].get(ptype, 0) + 1
        
        # Calculate timing spread
        if len(punchlines) > 1:
            timestamps = [self.time_to_seconds(p['suggested_timestamp']) for p in punchlines]
            metrics['timing_spread'] = max(timestamps) - min(timestamps)
        
        # Calculate transcript coverage (how much of transcript is represented)
        total_punchline_words = sum(len(p['text'].split()) for p in punchlines)
        transcript_words = len(transcript.split())
        metrics['transcript_coverage'] = (total_punchline_words / transcript_words) * 100 if transcript_words > 0 else 0
        
        # Overall quality score (0-100)
        score_components = {
            'word_count': min(100, max(0, 100 - abs(metrics['word_count_avg'] - 5) * 10)),  # Ideal ~5 words
            'emotional_impact': metrics['emotional_weight_distribution'].get('high', 0) * 30,
            'timing_distribution': min(100, metrics['timing_spread'] * 2),  # Good spread
            'coverage': min(100, metrics['transcript_coverage'] * 10)  # Good coverage
        }
        
        overall_score = sum(score_components.values()) / len(score_components)
        metrics['overall_score'] = round(overall_score, 1)
        metrics['score_components'] = score_components
        
        return metrics
    
    async def batch_process_variants(self, video_path: str, variant_count: int = 3) -> Dict[str, Any]:
        """Process multiple variants concurrently for A/B testing"""
        logger.info(f"🚀 Starting batch processing for {variant_count} variants...")
        
        tasks = []
        for i in range(1, variant_count + 1):
            task = self.process_video_with_enhanced_punchlines(video_path, variant_num=i)
            tasks.append(task)
        
        # Process all variants concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        batch_results = {
            'video_path': video_path,
            'variants': {},
            'batch_metadata': {
                'total_variants': variant_count,
                'successful_variants': 0,
                'failed_variants': 0,
                'processing_time': datetime.now().isoformat()
            }
        }
        
        for i, result in enumerate(results, 1):
            variant_key = f"variant_{i}"
            if isinstance(result, Exception):
                logger.error(f"Variant {i} failed: {result}")
                batch_results['variants'][variant_key] = {
                    'status': 'failed',
                    'error': str(result)
                }
                batch_results['batch_metadata']['failed_variants'] += 1
            else:
                batch_results['variants'][variant_key] = {
                    'status': 'success',
                    'data': result
                }
                batch_results['batch_metadata']['successful_variants'] += 1
        
        return batch_results
    
    async def create_comparison_video(self, video_path: str, variants_data: List[Dict], output_path: str) -> str:
        """Create a comparison video showing different punchline variants side by side"""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            variant_videos = []
            
            # Create individual variant videos
            for i, variant_data in enumerate(variants_data):
                variant_output = temp_dir / f"variant_{i+1}.mp4"
                await self.create_variant_with_punchlines_async(
                    video_path,
                    variant_data['punchlines'],
                    str(variant_output),
                    variant_data['style']
                )
                variant_videos.append(str(variant_output))
            
            # Create side-by-side comparison
            if len(variant_videos) == 2:
                filter_complex = "[0:v][1:v]hstack=inputs=2[v]"
                input_args = []
                for video in variant_videos:
                    input_args.extend(['-i', video])
            elif len(variant_videos) >= 3:
                filter_complex = "[0:v][1:v][2:v]hstack=inputs=3[v]"
                input_args = []
                for video in variant_videos[:3]:  # Limit to 3 for width
                    input_args.extend(['-i', video])
            else:
                # Single video, just copy
                filter_complex = None
                input_args = ['-i', variant_videos[0]]
            
            cmd = ['ffmpeg'] + input_args
            if filter_complex:
                cmd.extend(['-filter_complex', filter_complex, '-map', '[v]'])
            cmd.extend(['-c:v', 'libx264', '-preset', 'medium', '-crf', '23', output_path, '-y'])
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Comparison video creation failed: {stderr.decode()}")
            
            logger.info(f"✅ Created comparison video: {output_path}")
            return output_path
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def export_analytics_report(self, batch_results: Dict[str, Any], output_path: str) -> str:
        """Export detailed analytics report of punchline generation"""
        
        report = {
            'summary': {
                'video_path': batch_results['video_path'],
                'total_variants': batch_results['batch_metadata']['total_variants'],
                'success_rate': (batch_results['batch_metadata']['successful_variants'] / 
                               batch_results['batch_metadata']['total_variants']) * 100,
                'generated_at': datetime.now().isoformat()
            },
            'variant_analysis': {},
            'recommendations': []
        }
        
        # Analyze each successful variant
        best_score = 0
        best_variant = None
        
        for variant_key, variant_result in batch_results['variants'].items():
            if variant_result['status'] == 'success':
                data = variant_result['data']
                quality = data.get('quality_metrics', {})
                
                report['variant_analysis'][variant_key] = {
                    'punchline_count': len(data.get('punchlines', [])),
                    'quality_score': quality.get('overall_score', 0),
                    'emotional_impact': quality.get('emotional_weight_distribution', {}),
                    'style': data.get('style', {}),
                    'punchlines': data.get('punchlines', [])
                }
                
                # Track best variant
                if quality.get('overall_score', 0) > best_score:
                    best_score = quality.get('overall_score', 0)
                    best_variant = variant_key
        
        # Generate recommendations
        if best_variant:
            report['recommendations'].append(f"Best performing variant: {best_variant} (Score: {best_score})")
        
        report['recommendations'].extend([
            "Consider testing different timing intervals for better engagement",
            "High emotional weight punchlines typically perform better",
            "Maintain 3-6 words per punchline for optimal readability"
        ])
        
        # Export report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Analytics report exported: {output_path}")
        return output_path
    
    def get_variant_style_by_name(self, style_name: str) -> Dict[str, Any]:
        """Get variant style by name for external usage"""
        return self.variants.get(style_name, self.variants["variant_1_minimal"])
    
    def list_available_styles(self) -> List[str]:
        """List all available variant styles"""
        return list(self.variants.keys())
    
    async def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """Clean up old cache files"""
        if not self.cache_dir.exists():
            return 0
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            if current_time - cache_file.stat().st_mtime > max_age_seconds:
                try:
                    cache_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info(f"🧹 Cleaned up {cleaned_count} old cache files")
        return cleaned_count

# Usage example and convenience functions
async def create_enhanced_punchline_video(video_path: str, output_path: str, variant_num: int = 1) -> str:
    """Convenience function to create a single enhanced punchline video"""
    generator = EnhancedVideoPunchlineGenerator()
    
    if not generator.is_available():
        raise Exception("API keys not configured for punchline generation")
    
    # Process video and get punchline data
    result = await generator.process_video_with_enhanced_punchlines(video_path, variant_num)
    
    # Create the final video with punchlines
    final_video = await generator.create_variant_with_punchlines_async(
        video_path,
        result['punchlines'],
        output_path,
        result['style']
    )
    
    return final_video

async def create_multiple_variants(video_path: str, output_dir: str, variant_count: int = 3) -> Dict[str, str]:
    """Convenience function to create multiple punchline variants"""
    generator = EnhancedVideoPunchlineGenerator()
    
    # Process all variants
    batch_results = await generator.batch_process_variants(video_path, variant_count)
    
    # Create individual videos
    output_files = {}
    for i in range(1, variant_count + 1):
        variant_key = f"variant_{i}"
        if batch_results['variants'][variant_key]['status'] == 'success':
            variant_data = batch_results['variants'][variant_key]['data']
            output_path = os.path.join(output_dir, f"punchline_variant_{i}.mp4")
            
            await generator.create_variant_with_punchlines_async(
                video_path,
                variant_data['punchlines'],
                output_path,
                variant_data['style']
            )
            
            output_files[variant_key] = output_path
    
    return output_files