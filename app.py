import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio
from generator import Segment, load_csm_1b
from huggingface_hub import hf_hub_download, login

BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"

api_key = os.getenv("HF_TOKEN")
if not api_key:
    raise ValueError("HF_TOKEN not set. Please provide your Hugging Face token.")

gpu_timeout = int(os.getenv("GPU_TIMEOUT", 60))

login(token=api_key)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, device)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": "like revising for an exam I'd have to try and like keep up the momentum...",
        "audio": str(PROMPT_DIR / "conversational_a.wav"),
    },
    "conversational_b": {
        "text": "like a super Mario level. Like it's very like high detail...",
        "audio": str(PROMPT_DIR / "conversational_b.wav"),
    },
}

def infer(text_prompt_speaker_a, text_prompt_speaker_b, audio_prompt_speaker_a, audio_prompt_speaker_b, gen_conversation_input):
    if len(gen_conversation_input.strip() + text_prompt_speaker_a.strip() + text_prompt_speaker_b.strip()) >= 2000:
        raise gr.Error("Prompts and conversation too long.")

    return _infer(text_prompt_speaker_a, text_prompt_speaker_b, audio_prompt_speaker_a, audio_prompt_speaker_b, gen_conversation_input)

def _infer(text_prompt_speaker_a, text_prompt_speaker_b, audio_prompt_speaker_a, audio_prompt_speaker_b, gen_conversation_input):
    audio_prompt_a = prepare_prompt(text_prompt_speaker_a, 0, audio_prompt_speaker_a)
    audio_prompt_b = prepare_prompt(text_prompt_speaker_b, 1, audio_prompt_speaker_b)

    prompt_segments = [audio_prompt_a, audio_prompt_b]
    generated_segments = []

    conversation_lines = [line.strip() for line in gen_conversation_input.strip().split("\n") if line.strip()]
    for i, line in enumerate(conversation_lines):
        speaker_id = i % 2
        audio_tensor = generator.generate(text=line, speaker=speaker_id, context=prompt_segments + generated_segments, max_audio_length_ms=30_000)
        generated_segments.append(Segment(text=line, speaker=speaker_id, audio=audio_tensor))

    audio_tensors = [segment.audio for segment in generated_segments]
    audio_tensor = torch.cat(audio_tensors, dim=0)

    audio_array = (audio_tensor * 32768).to(torch.int16).cpu().numpy()
    return generator.sample_rate, audio_array

def prepare_prompt(text, speaker, audio_path):
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    audio_tensor, _ = load_prompt_audio(audio_path)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def load_prompt_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    
    if sample_rate != generator.sample_rate:
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=generator.sample_rate)
    
    return audio_tensor, generator.sample_rate

with gr.Blocks() as app:
    gr.Markdown("# Sesame CSM 1B\nGenerate from CSM 1B Model.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Speaker A")
            text_prompt_speaker_a = gr.Textbox(label="Speaker A Prompt", lines=4, value=SPEAKER_PROMPTS["conversational_a"]["text"])
            audio_prompt_speaker_a = gr.Audio(label="Speaker A Audio", type="filepath", value=SPEAKER_PROMPTS["conversational_a"]["audio"])

        with gr.Column():
            gr.Markdown("### Speaker B")
            text_prompt_speaker_b = gr.Textbox(label="Speaker B Prompt", lines=4, value=SPEAKER_PROMPTS["conversational_b"]["text"])
            audio_prompt_speaker_b = gr.Audio(label="Speaker B Audio", type="filepath", value=SPEAKER_PROMPTS["conversational_b"]["audio"])

    gen_conversation_input = gr.TextArea(label="Conversation", lines=10, value="Hey, how are you?\nI'm great!")
    generate_btn = gr.Button("Generate Conversation")
    audio_output = gr.Audio(label="Generated Audio")

    generate_btn.click(
        infer,
        inputs=[text_prompt_speaker_a, text_prompt_speaker_b, audio_prompt_speaker_a, audio_prompt_speaker_b, gen_conversation_input],
        outputs=[audio_output],
    )

app.launch()
