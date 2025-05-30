!pip install gradio transformers torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

# Load pre-trained model & processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()
def transcribe(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)

    # Resample if not 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Process the input
    input_values = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt").input_values

    # Get logits
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode predicted ids to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription
import gradio as gr

# Wrap the function in a Gradio interface
interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"), # Removed source="upload"
    outputs="text",
    title="Speech to Text",
    description="Upload a short audio clip and get the transcription using Wav2Vec2"
)

interface.launch()
