# coding=utf-8
# Qwen3-TTS Gradio Demo
# Supports: Voice Clone (Base), TTS (CustomVoice)
# based on https://huggingface.co/spaces/Qwen/Qwen3-TTS
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import io
import os

import gradio as gr
import numpy as np
import soundfile as sf
from openai import OpenAI

# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden",
    "Dylan",
    "Eric",
    "Ono_anna",
    "Ryan",
    "Serena",
    "Sohee",
    "Uncle_fu",
    "Vivian",
]
LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
]


## ============================================================================
## GLOBAL MODEL LOADING - Load all models at startup
## ============================================================================
# Check required environment variables
required_env_vars = [
    "INFERENCE_API_KEY_BASE",
    "INFERENCE_URL_BASE",
    "INFERENCE_API_KEY_TTS",
    "INFERENCE_URL_TTS",
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )


class EndpointConfig:
    def __init__(
        self, tts_api_key=None, tts_url=None, voice_api_key=None, voice_url=None
    ):
        self.tts_api_key = tts_api_key
        self.tts_url = tts_url
        self.voice_api_key = voice_api_key
        self.voice_url = voice_url

    def get_endpoint_config(self):
        # Return as a list, not a tuple, and append 'none' as the fifth element
        return [self.tts_api_key, self.tts_url, self.voice_api_key, self.voice_url]

    def set_endpoint_config(self, tts_api_key, tts_url, voice_api_key, voice_url):
        self.tts_api_key = tts_api_key
        self.tts_url = tts_url
        self.voice_api_key = voice_api_key
        self.voice_url = voice_url


class Models:
    def __init__(self):
        self.base_model_1_7b = None
        self.base_model_1_7b_model = None
        self.custom_voice_model_1_7b = None
        self.custom_voice_model_1_7b_model = None


endpoint_config = EndpointConfig()

models_endpoint = Models()


def update_config(tts_api_key, tts_url, voice_api_key, voice_url):
    endpoint_config.set_endpoint_config(tts_api_key, tts_url, voice_api_key, voice_url)

    outcome = ""
    try:
        models_endpoint.base_model_1_7b = OpenAI(api_key=tts_api_key, base_url=tts_url)

        models = models_endpoint.base_model_1_7b.models.list()
        if len(models.data) != 1:
            raise ValueError(
                f"Expected exactly one model in the endpoint {tts_url}, but got {len(models)}"
            )
        models_endpoint.base_model_1_7b_model = models.data[0].id
        outcome = "Saving config successfully!"
    except Exception as e:
        outcome = f"Error: {type(e).__name__}: {e}"

    try:
        models_endpoint.custom_voice_model_1_7b = OpenAI(
            api_key=voice_api_key,
            base_url=voice_url,
        )

        models = models_endpoint.custom_voice_model_1_7b.models.list()
        if len(models.data) != 1:
            raise ValueError(
                f"Expected exactly one model in the endpoint {voice_url} , but got {len(models)}"
            )
        models_endpoint.custom_voice_model_1_7b_model = models.data[0].id
        outcome = "Saving config successfully!"
    except Exception as e:
        outcome = f"Error: {type(e).__name__}: {e}"

    config = endpoint_config.get_endpoint_config()
    config.append(outcome)
    return config


update_config(
    tts_api_key=os.getenv("INFERENCE_API_KEY_BASE"),
    tts_url=f"{os.getenv("INFERENCE_URL_BASE")}/v1",
    voice_api_key=os.getenv("INFERENCE_API_KEY_TTS"),
    voice_url=f"{os.getenv("INFERENCE_URL_TTS")}/v1",
)


## ============================================================================


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def encode_audio_to_base64(audio_tuple) -> str:
    """Encode a (wav, sr) tuple to base64 data URL (WAV format)."""
    wav, sr = audio_tuple
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format="WAV")
    buffer.seek(0)
    audio_bytes = buffer.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{audio_b64}"


def decode_response_audio(content) -> tuple:
    """Decode audio bytes from API response to (wav, sr) tuple."""
    audio_buffer = io.BytesIO(content)
    audio_np, sr = sf.read(audio_buffer)
    return sr, audio_np


def generate_voice_clone(
    ref_audio,
    ref_text,
    target_text,
    language,
    use_xvector_only,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    audio_tuple = _audio_to_tuple(ref_audio)
    audio_b64 = encode_audio_to_base64(audio_tuple)

    if audio_tuple is None:
        return None, "Error: Reference audio is required."

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return (
            None,
            "Error: Reference text is required when 'Use x-vector only' is not enabled.",
        )

    try:
        response = models_endpoint.base_model_1_7b.audio.speech.create(
            input=target_text.strip(),
            voice=None,
            model=models_endpoint.base_model_1_7b_model,
            extra_body={
                "language": language,
                "ref_audio": audio_b64,
                "ref_text": ref_text.strip() if ref_text else None,
                "x_vector_only_mode": use_xvector_only,
                "task_type": "Base",
            },
        )
        return (
            decode_response_audio(response.content),
            "Generation completed successfully!",
        )

    except Exception as e:
        print(response.content)
        return None, f"Error: {type(e).__name__}: {e}"


def generate_custom_voice(
    text, language, speaker, instruct, progress=gr.Progress(track_tqdm=True)
):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."

    try:
        response = models_endpoint.custom_voice_model_1_7b.audio.speech.create(
            model=models_endpoint.custom_voice_model_1_7b_model,
            voice=speaker.lower().replace(" ", "_"),
            input=text.strip(),
            instructions=instruct.strip() if instruct else None,
            extra_body={
                "language": language,
            },
        )
        return (
            decode_response_audio(response.content),
            "Generation completed successfully!",
        )

    except Exception as e:
        print(response.content)
        return None, f"Error: {type(e).__name__}: {e}"


# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}
    """

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Demo") as demo:
        gr.Markdown(
            """
# Qwen3-TTS
This demo is using flex.ai inference API:
- **TTS (CustomVoice)**: Generate speech with predefined speakers and optional style instructions
- **Voice Clone (Base)**: Clone any voice from a reference audio
"""
        )

        def toggle_api_key_visibility(visible, value):
            type = "text" if visible else "password"
            return gr.Textbox(label="API Key", type=type, value=value)

        with gr.Tabs():
            # Tab 1: TTS (CustomVoice)
            with gr.Tab("TTS (CustomVoice)"):
                gr.Markdown("### Text-to-Speech with Predefined Speakers")
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="""
Hello! Welcome to Flex dot AI.
This demo is using Flex private cloud inference API to run Qwen3-TTS models in the cloud, so no GPU is required on your end!
Just enter your text and settings, and let the flex compute do the heavy lifting to generate high-quality speech at scale for you, fast and scalable inference capabilities included!
And do not forget that you can also finetune your own custom TTS model here and then deploy on flex dot ai in just one click!
""",
                        )
                        with gr.Row():
                            tts_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="English",
                                interactive=True,
                            )
                            tts_speaker = gr.Dropdown(
                                label="Speaker",
                                choices=SPEAKERS,
                                value="Aiden",
                                interactive=True,
                            )
                        with gr.Row():
                            tts_instruct = gr.Textbox(
                                label="Style Instruction (Optional)",
                                lines=2,
                                placeholder="e.g., Speak in a cheerful and energetic tone",
                                value="Speak in a cheerful and energetic tone",
                            )

                        tts_btn = gr.Button("Generate Speech", variant="primary")

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        tts_status = gr.Textbox(
                            label="Status", lines=2, interactive=False
                        )

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct],
                    outputs=[tts_audio_out, tts_status],
                )

            # Tab 2: Voice Clone (Base)
            with gr.Tab("Voice Clone (Base)"):
                gr.Markdown("### Clone Voice from Reference Audio")
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_ref_audio = gr.Audio(
                            label="Reference Audio (Upload a voice sample to clone)",
                            type="numpy",
                        )
                        clone_ref_text = gr.Textbox(
                            label="Reference Text (Transcript of the reference audio)",
                            lines=2,
                            placeholder="Enter the exact text spoken in the reference audio...",
                        )
                        clone_xvector = gr.Checkbox(
                            label="Use x-vector only (No reference text needed, but lower quality)",
                            value=False,
                        )

                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(
                            label="Target Text (Text to synthesize with cloned voice)",
                            lines=4,
                            placeholder="Enter the text you want the cloned voice to speak...",
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )

                        clone_btn = gr.Button("Clone & Generate", variant="primary")

                with gr.Row():
                    clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    clone_status = gr.Textbox(
                        label="Status", lines=2, interactive=False
                    )

                clone_btn.click(
                    generate_voice_clone,
                    inputs=[
                        clone_ref_audio,
                        clone_ref_text,
                        clone_target_text,
                        clone_language,
                        clone_xvector,
                    ],
                    outputs=[clone_audio_out, clone_status],
                )
            # Tab 3: Voice Clone (Base)
            with gr.Tab("Endpoint Config"):
                with gr.Group(elem_id="tab-content"):
                    tts_url = gr.Textbox(label="TTS CustomVoice URL")
                    with gr.Row():
                        tts_api_key = gr.Textbox(
                            placeholder="TTS CustomVoice API Key",
                            type="password",
                            scale=4,
                            show_label=False,
                        )
                        tts_api_key_visible = gr.Checkbox(
                            label="Show TTS CustomVoice API Key",
                            value=False,
                            scale=1,
                        )

                    voice_url = gr.Textbox(label="Voice Clone URL")
                    with gr.Row():
                        voice_api_key = gr.Textbox(
                            label="Voice Clone API Key",
                            type="password",
                            scale=4,
                            show_label=False,
                        )
                        voice_api_key_visible = gr.Checkbox(
                            label="Voice Clone API Key",
                            value=False,
                            scale=1,
                        )
                    save_btn = gr.Button("Save")
                    config_error = gr.Markdown(value="", visible=False)
                    save_btn.click(
                        update_config,
                        inputs=[
                            tts_api_key,
                            tts_url,
                            voice_api_key,
                            voice_url,
                        ],
                        outputs=[
                            tts_api_key,
                            tts_url,
                            voice_api_key,
                            voice_url,
                            config_error,
                        ],
                    )
                    tts_api_key_visible.change(
                        toggle_api_key_visibility,
                        inputs=[tts_api_key_visible, tts_api_key],
                        outputs=tts_api_key,
                    )
                    voice_api_key_visible.change(
                        toggle_api_key_visibility,
                        inputs=[voice_api_key_visible, voice_api_key],
                        outputs=voice_api_key,
                    )

        demo.load(
            endpoint_config.get_endpoint_config,
            inputs=None,
            outputs=[
                tts_api_key,
                tts_url,
                voice_api_key,
                voice_url,
            ],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Gradio Demo")
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Enable public sharing of the Gradio app (default: False)",
    )
    args = parser.parse_args()
    demo = build_ui()
    demo.launch(share=args.share)
