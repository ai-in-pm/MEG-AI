from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from brain_decoder.signal_processor import BrainSignalProcessor
from chat_model.conversation import ConversationModel
from utils.audio import AudioManager
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
socketio = SocketIO(app)

# Initialize components
brain_processor = BrainSignalProcessor()
conversation_model = ConversationModel()
audio_manager = AudioManager()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('brain_signal')
def handle_brain_signal(data):
    """Handle incoming brain signal data"""
    processed_signal = brain_processor.process(data)
    if processed_signal:
        decoded_text = brain_processor.decode_to_text(processed_signal)
        response = conversation_model.generate_response(decoded_text)
        
        # Generate audio response
        audio_data = audio_manager.synthesize_speech(response)
        
        emit('ai_response', {
            'text': response,
            'audio': audio_data,
            'brain_data': processed_signal.visualization_data
        })

@socketio.on('text_input')
def handle_text_input(data):
    """Handle direct text input"""
    response = conversation_model.generate_response(data['text'])
    audio_data = audio_manager.synthesize_speech(response)
    
    emit('ai_response', {
        'text': response,
        'audio': audio_data
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    brain_processor.initialize()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')
    brain_processor.cleanup()

if __name__ == '__main__':
    socketio.run(app, debug=True)
