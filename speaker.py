import pyttsx3
from typing import Optional

class Speaker:
    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Initialize the text-to-speech engine.
        
        Args:
            rate (int): Speech rate (words per minute)
            volume (float): Volume level (0.0 to 1.0)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Get available voices and set to first available voice
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
    
    def speak(self, text: str) -> None:
        """
        Convert text to speech.
        
        Args:
            text (str): Text to be spoken
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def announce_detection(self, object_name: str) -> None:
        """
        Announce a detected object with natural language.
        
        Args:
            object_name (str): Name of the detected object
        """
        # Add some variety to the announcements
        phrases = [
            f"I see a {object_name}",
            f"There's a {object_name}",
            f"I can see a {object_name}",
            f"I've detected a {object_name}"
        ]
        
        import random
        announcement = random.choice(phrases)
        self.speak(announcement)
    
    def cleanup(self) -> None:
        """Clean up the text-to-speech engine."""
        try:
            self.engine.stop()
        except:
            pass 