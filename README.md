# Brain-AI Communication Chatbot

A real-time human-AI communication chatbot inspired by Meta's brain-AI research. This project combines advanced language understanding with neural signal interpretation for an engaging, human-like conversation experience enhanced by brain-computer interface (BCI) technology.

The development of this repository was inspired by https://ai.meta.com/blog/brain-ai-image-decoding-meg-magnetoencephalography/. 

## Features

- Real-time neural signal processing and interpretation
- Natural language conversation with context awareness
- Speech synthesis for immersive interaction
- Visual feedback of brain activity and AI reasoning
- Integration with Meta's DINOv2 for advanced AI capabilities
- Interactive exploration of AI-brain communication

## Requirements

- Python 3.8+
- EEG/MEG device compatible with BrainFlow
- Microphone for voice input (optional)
- Speakers for audio output

## Installation

1. Clone the repository
2. Create and activate virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Connect your EEG/MEG device
2. Run the application:
   ```
   python app.py
   ```
3. Open your web browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `brain_decoder/`: Neural signal processing modules
- `chat_model/`: AI conversation model
- `static/`: Frontend assets (JS, CSS)
- `templates/`: HTML templates
- `utils/`: Utility functions

## License

MIT License

## Acknowledgments

Based on research and technologies from Meta AI, particularly their work on brain-AI interaction and the DINOv2 framework.
