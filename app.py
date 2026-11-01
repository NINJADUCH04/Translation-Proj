import gradio as gr
import numpy as np
from faster-whisper import WhisperModel

# Use "tiny", "base", or "small" for low-latency streaming on CPU.
# Change device="cuda" and compute_type="float16" if you have an NVIDIA GPU.
model = WhisperModel("base", device="cpu", compute_type="int8")

def process_stream(audio_chunk, state):
    if state is None:
        state = ""

    if audio_chunk is None:
        return state, state

    sample_rate, audio_data = audio_chunk

    # Ensure audio is float32 and normalized between -1.0 and 1.0
    if audio_data.dtype.kind == "i":
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    elif audio_data.dtype.kind == "f":
        audio_data = audio_data.astype(np.float32)

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Whisper expects 16,000 Hz audio
    if sample_rate != 16000:
        # Simple decimation/upsampling fallback for testing
        num_target_samples = int(len(audio_data) * 16000 / sample_rate)
        audio_data = np.interp(
            np.linspace(0, len(audio_data), num_target_samples, endpoint=False),
            np.arange(len(audio_data)),
            audio_data,
        ).astype(np.float32)

    # Ignore silent or extremely short segments (< 0.5s)
    if len(audio_data) < 8000:
        return state, state

    # task="translate" translates any spoken source language into English
    segments, _ = model.transcribe(
        audio_data,
        task="translate",
        beam_size=1,
        vad_filter=True
    )

    new_text = " ".join([segment.text for segment in segments]).strip()
    if new_text:
        state += f" {new_text}"

    return state.strip(), state

def clear_session():
    return "", ""

with gr.Blocks(title="Local Speech Translation Stream") as demo:
    gr.Markdown("### Real-Time Local Speech-to-Text Translation")
    
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            streaming=True,
            type="numpy",
            label="Microphone Stream"
        )
        
    translation_display = gr.Textbox(
        label="Live Translation (English Output)",
        lines=8,
        placeholder="Translated text will appear here as you speak...",
        interactive=False
    )
    
    state_store = gr.State()
    clear_btn = gr.Button("Clear Output")

    # Hook streaming microphone events to the inference function
    audio_input.stream(
        fn=process_stream,
        inputs=[audio_input, state_store],
        outputs=[translation_display, state_store]
    )

    clear_btn.click(
        fn=clear_session,
        outputs=[translation_display, state_store]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)