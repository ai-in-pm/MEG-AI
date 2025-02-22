import sounddevice as sd
import numpy as np
from scipy import signal
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

class AudioManager:
    def __init__(self):
        self.processor = None
        self.model = None
        self.vocoder = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize text-to-speech models"""
        try:
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        except Exception as e:
            print(f"Error initializing audio models: {e}")
    
    def synthesize_speech(self, text):
        """Convert text to speech"""
        try:
            # Prepare input
            inputs = self.processor(text=text, return_tensors="pt")
            
            # Generate speech
            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.vocoder,
                speaker_embeddings=None
            )
            
            # Convert to proper format for transmission
            audio_data = speech.numpy()
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data.tolist()
        
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            return None
    
    def play_audio(self, audio_data):
        """Play audio through speakers"""
        try:
            audio_array = np.array(audio_data)
            sd.play(audio_array, samplerate=16000)
            sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def process_microphone_input(self, duration=5):
        """Record and process microphone input"""
        try:
            # Record audio
            recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1)
            sd.wait()
            
            # Process recording
            processed_audio = self._process_audio(recording)
            
            return processed_audio
        except Exception as e:
            print(f"Error processing microphone input: {e}")
            return None
    
    def _process_audio(self, audio_data):
        """Process raw audio data"""
        # Apply some basic audio processing
        # Normalize
        audio_normalized = audio_data / np.max(np.abs(audio_data))
        
        # Apply bandpass filter
        nyq = 8000
        low = 80 / nyq
        high = 7600 / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio_normalized.flatten())
        
        return filtered.tolist()
