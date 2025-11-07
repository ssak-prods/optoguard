import cv2
import time
import base64
import requests
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from speaker import Speaker

@dataclass
class OpenRouterConfig:
    api_key: str
    model: str = "mistralai/mistral-7b-instruct"
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    max_tokens: int = 150
    temperature: float = 0.7

class OpenRouterDetector:
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.speaker = Speaker(rate=150, volume=1.0)
        self.last_announcement = ""
        self.last_announcement_time = 0
        self.cooldown_seconds = 5.0
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam")
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame to base64 string with compression."""
        # Resize frame to reduce size
        frame = cv2.resize(frame, (320, 240))  # Reduced from 640x480
        
        # Compress with lower quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]  # Lower quality for smaller size
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _create_prompt(self, frame_base64: str) -> str:
        """Create a prompt for the Mistral model."""
        return f"""Analyze this image. Be extremely brief and precise.

RULES:
1. Maximum 2 sentences
2. Only describe what you can see with 100% certainty
3. If uncertain, say "I cannot clearly identify any objects"
4. No assumptions or guesses

Format: "I see [object] in [position]" or "I cannot clearly identify any objects"

Image data: {frame_base64}"""
    
    def _call_openrouter_api(self, prompt: str) -> Optional[str]:
        """Call OpenRouter API and get response."""
        headers = {
            "HTTP-Referer": "https://github.com/yourusername/your-repo",
            "X-Title": "AI Scene Analyzer",
            "Authorization": f"Bearer {self.config.api_key.strip()}"
        }
        
        data = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a brief image analyzer. Respond in 1-2 short sentences. Never make assumptions."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 50,  # Reduced from 150 to force brevity
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.config.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Check for different response formats
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content'].strip()
                # Filter out any responses that seem to be hallucinating
                if any(phrase in content.lower() for phrase in [
                    "multiple", "group of", "several",
                    "many", "various", "in the background",
                    "appears to be", "seems to be",
                    "suggesting", "indicating", "possibly"
                ]):
                    return "I cannot clearly identify any objects."
                return content
            elif 'response' in result:
                return result['response'].strip()
            elif 'output' in result:
                return result['output'].strip()
            else:
                print("Unexpected API response format:", result)
                return None
                
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return None
    
    def _should_announce(self, description: str) -> bool:
        """Check if we should announce the description based on cooldown and similarity."""
        current_time = time.time()
        if current_time - self.last_announcement_time < self.cooldown_seconds:
            return False
        
        # Always announce if it's been more than 2x cooldown time
        if current_time - self.last_announcement_time > (self.cooldown_seconds * 2):
            return True
            
        # Check similarity for intermediate times
        if description == self.last_announcement:
            return False
        return True
    
    def run(self):
        """Main loop for real-time scene analysis."""
        print("OpenRouter AI Scene Analyzer is running!")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Encode frame and create prompt
                frame_base64 = self._encode_frame(frame)
                prompt = self._create_prompt(frame_base64)
                
                # Get AI analysis
                description = self._call_openrouter_api(prompt)
                
                if description:
                    print(f"Analysis: {description}")  # Always print to terminal
                    
                    if self._should_announce(description):
                        print(f"🔊 Speaking: \"{description}\"")
                        self.speaker.speak(description)
                        self.last_announcement = description
                        self.last_announcement_time = time.time()
                
                # Display frame with description
                cv2.putText(frame, "OpenRouter AI Analysis", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if description:
                    # Split description into lines for display
                    words = description.split()
                    lines = []
                    current_line = []
                    
                    for word in words:
                        current_line.append(word)
                        if len(' '.join(current_line)) > 40:
                            lines.append(' '.join(current_line[:-1]))
                            current_line = [word]
                    
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    # Display description
                    for i, line in enumerate(lines):
                        y_pos = 70 + (i * 30)
                        cv2.putText(frame, line, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('OpenRouter AI Scene Analysis', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small delay to prevent high CPU usage
                time.sleep(0.1)
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("OpenRouter AI Scene Analyzer has been stopped.")

def main():
    # Get API key from environment variable
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("Please set the OPENROUTER_API_KEY environment variable")
    
    # Create config
    config = OpenRouterConfig(api_key=api_key)
    
    # Create and run detector
    detector = OpenRouterDetector(config)
    detector.run()

if __name__ == "__main__":
    main() 