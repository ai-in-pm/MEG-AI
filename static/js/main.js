// Initialize Socket.IO connection
const socket = io();

// DOM Elements
const connectionStatus = document.getElementById('connection-status');
const deviceStatus = document.getElementById('device-status');
const chatMessages = document.getElementById('chat-messages');
const thinkButton = document.getElementById('think-button');
const typeButton = document.getElementById('type-button');
const thinkingIndicator = document.getElementById('thinking-indicator');
const typingArea = document.getElementById('typing-area');
const textInput = document.getElementById('text-input');
const sendButton = document.getElementById('send-button');

// Visualization Elements
const brainActivityPlot = document.getElementById('brain-activity-plot');
const frequencySpectrumPlot = document.getElementById('frequency-spectrum-plot');
const waveIndicators = document.querySelectorAll('.wave-indicator');
const aiConfidenceMeter = document.getElementById('ai-confidence-meter');
const aiThinkingProcess = document.getElementById('ai-thinking-process');

// Connection Events
socket.on('connect', () => {
    connectionStatus.textContent = 'Connected';
    connectionStatus.classList.add('connected');
});

socket.on('disconnect', () => {
    connectionStatus.textContent = 'Disconnected';
    connectionStatus.classList.remove('connected');
    deviceStatus.textContent = 'No Device';
    deviceStatus.classList.remove('connected');
});

// Brain Signal Processing
let isThinkingMode = true;
let brainSignalInterval;

thinkButton.addEventListener('click', () => {
    isThinkingMode = true;
    thinkButton.classList.add('active');
    typeButton.classList.remove('active');
    thinkingIndicator.style.display = 'flex';
    typingArea.style.display = 'none';
    startBrainSignalProcessing();
});

typeButton.addEventListener('click', () => {
    isThinkingMode = false;
    typeButton.classList.add('active');
    thinkButton.classList.remove('active');
    thinkingIndicator.style.display = 'none';
    typingArea.style.display = 'flex';
    stopBrainSignalProcessing();
});

function startBrainSignalProcessing() {
    // Simulate brain signal processing
    brainSignalInterval = setInterval(() => {
        const mockBrainSignal = generateMockBrainSignal();
        socket.emit('brain_signal', mockBrainSignal);
    }, 1000);
}

function stopBrainSignalProcessing() {
    clearInterval(brainSignalInterval);
}

function generateMockBrainSignal() {
    // Generate mock brain signal data for testing
    return Array.from({ length: 250 }, () => Math.random() * 2 - 1);
}

// Chat Interface
sendButton.addEventListener('click', sendMessage);
textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const text = textInput.value.trim();
    if (text) {
        socket.emit('text_input', { text });
        addMessage(text, 'user');
        textInput.value = '';
    }
}

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// AI Response Handling
socket.on('ai_response', (data) => {
    addMessage(data.text, 'ai');
    updateVisualization(data);
    if (data.audio) {
        playAudioResponse(data.audio);
    }
});

function updateVisualization(data) {
    if (data.brain_data) {
        updateBrainActivityPlot(data.brain_data);
        updateFrequencySpectrum(data.brain_data);
        updateWaveIndicators(data.brain_data.band_powers);
    }
    updateAIConfidence(data.confidence || Math.random());
    updateAIThinkingProcess(data.thinking_process || 'Processing response...');
}

// Visualization Functions
function updateBrainActivityPlot(data) {
    const trace = {
        y: data.raw_signal,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#2962ff' }
    };
    
    const layout = {
        title: 'Brain Activity',
        height: 200,
        margin: { t: 30, r: 20, l: 20, b: 20 }
    };
    
    Plotly.newPlot(brainActivityPlot, [trace], layout);
}

function updateFrequencySpectrum(data) {
    const trace = {
        x: data.frequency_spectrum.frequencies,
        y: data.frequency_spectrum.powers,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#00C851' }
    };
    
    const layout = {
        title: 'Frequency Spectrum',
        height: 200,
        margin: { t: 30, r: 20, l: 20, b: 20 }
    };
    
    Plotly.newPlot(frequencySpectrumPlot, [trace], layout);
}

function updateWaveIndicators(bandPowers) {
    waveIndicators.forEach(indicator => {
        const wave = indicator.dataset.wave;
        const power = bandPowers[wave];
        const intensity = Math.min(power * 100, 100);
        indicator.style.backgroundColor = `rgba(41, 98, 255, ${intensity/100})`;
    });
}

function updateAIConfidence(confidence) {
    aiConfidenceMeter.style.background = `linear-gradient(to right, #00C851 ${confidence*100}%, #ddd ${confidence*100}%)`;
}

function updateAIThinkingProcess(process) {
    aiThinkingProcess.textContent = process;
}

function playAudioResponse(audioData) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = audioContext.createBuffer(1, audioData.length, 16000);
    audioBuffer.getChannelData(0).set(audioData);
    
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start();
}
