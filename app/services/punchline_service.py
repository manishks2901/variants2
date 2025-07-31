import os
import json
import requests
from groq import Groq
import subprocess
import tempfile
import re
import random
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from decouple import config

logger = logging.getLogger(__name__)

class VideoPunchlineGenerator:
    def __init__(self):
        self.elevenlabs_key = config('ELEVENLABS_API_KEY', default='')
        self.groq_api_key = config('GROQ_API_KEY', default='')
        
        if not self.elevenlabs_key or not self.groq_api_key:
            logger.warning("ElevenLabs or Groq API keys not configured. Punchline generation will be disabled.")
            return
            
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Different styling variants
        self.variants = {
            "variant_1": {
                "bg_color": "black",
                "text_color": "white", 
                "font_size": 50,
                "border_color": "red",
                "border_width": 2,
                "duration": 1.0
            },
            "variant_2": {
                "bg_color": "0x1a1a2e",  # Dark blue
                "text_color": "yellow",
                "font_size": 55,
                "border_color": "white",
                "border_width": 2,
                "duration": 1.0
            }
        }
        
    def is_available(self) -> bool:
        """Check if punchline generation is available (API keys configured)"""
        return bool(self.elevenlabs_key and self.groq_api_key)
        
    def extract_audio_ffmpeg(self, video_path: str, audio_path: str) -> None:
        """Extract audio using FFmpeg with better quality settings"""
        cmd = [
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '44100', '-ac', '2', 
            '-af', 'volume=1.5',  # Boost volume for better transcription
            audio_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg audio extraction failed: {result.stderr}")
        
    def transcribe_audio(self, audio_file_path: str) -> tuple[str, Dict[str, Any]]:
        """Transcribe audio using ElevenLabs API"""
        if not self.elevenlabs_key:
            raise Exception("ElevenLabs API key not configured")
            
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        
        headers = {
            "xi-api-key": self.elevenlabs_key
        }
        
        with open(audio_file_path, 'rb') as audio_file:
            files = {"file": audio_file}
            data = {"model_id": "scribe_v1"}
            
            response = requests.post(url, headers=headers, files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            return result.get('text', ''), result.get('alignment', {})
        else:
            raise Exception(f"Transcription failed: {response.text}")
    
    def generate_punchlines_variant(self, transcript: str, variant_num: int) -> List[Dict[str, str]]:
        """Extract direct punchlines from transcript - FIXED JSON PARSING"""
        
        if not self.groq_api_key:
            raise Exception("Groq API key not configured")
            
        prompt = f"""
        From this transcript, extract 4 of the most impactful quotes that would make great text overlays. Pick the actual spoken words that are most engaging.
        
        Transcript: "{transcript}"
        
        Format your response as JSON with this structure:
        {{
            "punchlines": [
                {{"text": "First powerful quote", "suggested_timestamp": "0:05"}},
                {{"text": "Second impactful quote", "suggested_timestamp": "0:15"}},
                {{"text": "Third strong quote", "suggested_timestamp": "0:25"}},
                {{"text": "Fourth great quote", "suggested_timestamp": "0:35"}}
            ]
        }}
        
        Rules:
        - Use EXACT words from the transcript (don't create new content)
        - Pick 4 most powerful moments
        - Keep quotes under 8 words
        - Space timestamps 10 seconds apart
        """
        
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.6
        )
        
        try:
            # Clean JSON response
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            result = json.loads(content)
            all_punchlines = result["punchlines"]
            
            # Split punchlines between variants
            if variant_num == 1:
                selected_punchlines = [all_punchlines[0], all_punchlines[2]] if len(all_punchlines) >= 3 else all_punchlines[:2]
            else:
                selected_punchlines = [all_punchlines[1], all_punchlines[3]] if len(all_punchlines) >= 4 else all_punchlines[1:3]
                
            return selected_punchlines
            
        except json.JSONDecodeError:
            # Fallback parsing
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
                        "suggested_timestamp": timestamp
                    })
            
            if len(punchlines) >= 4:
                if variant_num == 1:
                    return [punchlines[0], punchlines[2]]
                else:
                    return [punchlines[1], punchlines[3]]
            elif len(punchlines) >= 2:
                if variant_num == 1:
                    return punchlines[::2]
                else:
                    return punchlines[1::2]
            else:
                return [
                    {"text": f"Quote {variant_num}-1", "suggested_timestamp": "0:05"},
                    {"text": f"Quote {variant_num}-2", "suggested_timestamp": "0:25"}
                ]
    
    def time_to_seconds(self, time_str: str) -> int:
        """Convert MM:SS to seconds"""
        try:
            parts = time_str.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return 5
    
    def get_video_info(self, video_path: str) -> tuple[str, float]:
        """Get video dimensions and duration"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return "1920x1080", 30.0
        
        info = json.loads(result.stdout)
        video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            width = video_stream.get('width', 1920)
            height = video_stream.get('height', 1080)
            duration = float(info['format'].get('duration', 30.0))
            return f"{width}x{height}", duration
        
        return "1920x1080", 30.0
    
    def create_blackscreen_fast(self, text: str, duration: float, output_path: str, video_size: str, variant_style: Dict[str, Any]) -> None:
        """Create blackscreen with text - optimized for speed and text fitting"""
        
        # Better text escaping and wrapping
        escaped_text = text.replace("'", "\\'").replace(":", "\\:").replace(",", "\\,")
        
        # Smart text wrapping for long text
        words = escaped_text.split(' ')
        if len(words) > 4:  # If more than 4 words, split into 2 lines
            mid = len(words) // 2
            line1 = ' '.join(words[:mid])
            line2 = ' '.join(words[mid:])
            escaped_text = f"{line1}\\n{line2}"
        
        width, height = map(int, video_size.split('x'))
        max_font_size = min(width // 20, height // 15)  # Dynamic font sizing
        font_size = min(variant_style['font_size'], max_font_size)
        
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'color=c={variant_style["bg_color"]}:size={video_size}:duration={duration}',
            '-f', 'lavfi', 
            '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}',
            '-vf', f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={variant_style['text_color']}:x=(w-text_w)/2:y=(h-text_h)/2:borderw={variant_style['border_width']}:bordercolor={variant_style['border_color']}",
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'fast',
            '-crf', '23',
            '-t', str(duration),
            '-pix_fmt', 'yuv420p',
            '-r', '30',
            output_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback without custom font
            cmd[6] = f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={variant_style['text_color']}:x=(w-text_w)/2:y=(h-text_h)/2:borderw={variant_style['border_width']}:bordercolor={variant_style['border_color']}"
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Text overlay creation failed: {result.stderr}")
    
    def create_variant_with_punchlines(self, video_path: str, punchlines: List[Dict[str, str]], output_path: str, variant_style: Dict[str, Any]) -> str:
        """Create video with blackscreens - optimized for speed and audio sync"""
        
        video_size, total_duration = self.get_video_info(video_path)
        temp_dir = tempfile.mkdtemp()
        
        try:
            segments = []
            segment_files = []
            current_time = 0
            
            for i, punchline in enumerate(punchlines):
                timestamp = self.time_to_seconds(punchline["suggested_timestamp"])
                
                if timestamp >= total_duration:
                    timestamp = min(total_duration - 3, current_time + 10)
                
                # Create video segment before punchline
                if timestamp > current_time:
                    segment_path = os.path.join(temp_dir, f"seg_{i}_video.mp4")
                    cmd = [
                        'ffmpeg', '-i', video_path,
                        '-ss', str(current_time),
                        '-t', str(timestamp - current_time),
                        '-c:v', 'libx264', '-c:a', 'aac',  # Re-encode for consistency
                        '-preset', 'fast',
                        '-crf', '23',
                        '-r', '30',  # Consistent framerate
                        '-avoid_negative_ts', 'make_zero',
                        segment_path, '-y'
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        segments.append(f"file '{segment_path}'")
                        segment_files.append(segment_path)
                
                # Create blackscreen segment
                blackscreen_path = os.path.join(temp_dir, f"seg_{i}_black.mp4")
                self.create_blackscreen_fast(
                    punchline["text"], 
                    variant_style["duration"], 
                    blackscreen_path, 
                    video_size,
                    variant_style
                )
                segments.append(f"file '{blackscreen_path}'")
                segment_files.append(blackscreen_path)
                
                current_time = timestamp + variant_style["duration"]  # Account for blackscreen duration
            
            # Add remaining video
            if current_time < total_duration - 1:
                final_segment_path = os.path.join(temp_dir, f"seg_final.mp4")
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-ss', str(current_time),
                    '-c:v', 'libx264', '-c:a', 'aac',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-r', '30',
                    '-avoid_negative_ts', 'make_zero',
                    final_segment_path, '-y'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    segments.append(f"file '{final_segment_path}'")
                    segment_files.append(final_segment_path)
            
            # Create concat file
            concat_file = os.path.join(temp_dir, 'concat.txt')
            with open(concat_file, 'w') as f:
                for segment in segments:
                    f.write(segment + '\n')
            
            # Concatenation with consistent encoding
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264', '-c:a', 'aac',
                '-preset', 'fast',
                '-crf', '23',
                '-r', '30',  # Consistent framerate
                '-fflags', '+genpts',  # Fix timestamps
                output_path, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg concatenation failed: {result.stderr}")
            
            logger.info(f"Created punchline variant: {output_path}")
            return output_path
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def process_video_with_punchlines(self, video_path: str, variant_num: int = 1) -> Dict[str, Any]:
        """Process video and add punchlines - returns punchline data"""
        
        if not self.is_available():
            raise Exception("Punchline generation not available - API keys not configured")
        
        # Create temp audio file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_path = temp_audio.name
        temp_audio.close()
        
        try:
            logger.info("Extracting audio for transcription...")
            self.extract_audio_ffmpeg(video_path, audio_path)
            
            logger.info("Transcribing audio...")
            transcript, alignment = self.transcribe_audio(audio_path)
            
            logger.info(f"Generating punchlines for variant {variant_num}...")
            punchlines = self.generate_punchlines_variant(transcript, variant_num)
            
            variant_key = f"variant_{variant_num}"
            variant_style = self.variants.get(variant_key, self.variants["variant_1"])
            
            return {
                'transcript': transcript,
                'punchlines': punchlines,
                'style': variant_style,
                'alignment': alignment
            }
            
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
