import numpy as np
from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from scipy import signal
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class BrainSignalProcessor:
    def __init__(self):
        self.board = None
        self.model = None
        self.tokenizer = None
        self.initialize_ai_models()
    
    def initialize_ai_models(self):
        """Initialize the AI models for signal processing"""
        try:
            # Load a suitable transformer model for signal processing
            self.model = AutoModel.from_pretrained('facebook/opt-350m')
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
            self.model.eval()
        except Exception as e:
            print(f"Error initializing models: {e}")
    
    def initialize(self):
        """Initialize the BCI board connection"""
        try:
            params = BrainFlowInputParams()
            # Using synthetic board for testing - replace with your actual board
            self.board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
            self.board.prepare_session()
            self.board.start_stream()
            return True
        except Exception as e:
            print(f"Error initializing board: {e}")
            return False
    
    def cleanup(self):
        """Clean up board connection"""
        if self.board:
            self.board.stop_stream()
            self.board.release_session()
    
    def process(self, raw_data):
        """Process incoming brain signal data"""
        try:
            # Convert raw data to numpy array
            data = np.array(raw_data)
            
            # Apply bandpass filter (0.5-50 Hz)
            fs = 250  # sampling frequency
            nyq = 0.5 * fs
            low = 0.5 / nyq
            high = 50.0 / nyq
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_data = signal.filtfilt(b, a, data)
            
            # Extract features using transformer model
            with torch.no_grad():
                # Convert signal to spectrogram
                f, t, Sxx = signal.spectrogram(filtered_data, fs=250)
                Sxx_norm = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())
                
                # Prepare input for the model
                signal_tensor = torch.FloatTensor(Sxx_norm).unsqueeze(0)
                features = self.model(inputs_embeds=signal_tensor).last_hidden_state
            
            return {
                'processed_signal': filtered_data.tolist(),
                'features': features.cpu().numpy().tolist(),
                'visualization_data': self.prepare_visualization_data(filtered_data)
            }
        except Exception as e:
            print(f"Error processing signal: {e}")
            return None
    
    def prepare_visualization_data(self, signal_data):
        """Prepare data for frontend visualization"""
        # Calculate frequency spectrum
        freqs, psd = signal.welch(signal_data, fs=250)
        
        # Calculate brain wave bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        band_powers = {}
        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_powers[band] = np.mean(psd[mask])
        
        return {
            'raw_signal': signal_data.tolist()[-1000:],  # Last 1000 samples
            'frequency_spectrum': {
                'frequencies': freqs.tolist(),
                'powers': psd.tolist()
            },
            'band_powers': band_powers
        }
    
    def decode_to_text(self, processed_data):
        """Convert processed brain signals to text"""
        try:
            features = torch.FloatTensor(processed_data['features'])
            
            # Use the model to generate text from features
            with torch.no_grad():
                # Project features to the model's hidden dimension
                projected_features = F.linear(
                    features.mean(dim=1),
                    self.model.get_input_embeddings().weight
                )
                
                # Generate text using the model
                outputs = self.model.generate(
                    inputs_embeds=projected_features.unsqueeze(0),
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
                decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return decoded_text
                
        except Exception as e:
            print(f"Error decoding signal: {e}")
            return "I'm processing your thoughts..."
