# Copyright (c) 2025 FlexAI
# This file is part of the FlexAI Experiments repository.
# SPDX-License-Identifier: MIT

import os

import gradio as gr
from openai import OpenAI


def check_env() -> None:
    if "LLM_API_KEY" not in os.environ:
        raise ValueError("Please set the LLM_API_KEY environment variable.")
    if "LLM_URL" not in os.environ:
        raise ValueError("Please set the LLM_URL environment variable.")


def infer(audio_path: str, client: OpenAI):
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-large-v3",
            response_format="json",
            temperature=0.0,
            extra_body=dict(
                seed=42,
                repetition_penalty=1.3,
            ),
        )
        return transcription.text


def transcribe_audio(audio_file):
    if audio_file is None:
        return "No audio file provided"

    client = OpenAI(
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_URL") + "/v1",
    )

    result = infer(audio_file, client)

    return result


def create_gradio_interface():
    with gr.Blocks(title="Speech-to-Text Transcription") as demo:
        gr.Markdown("# Speech-to-Text Transcription")
        gr.Markdown("Record audio using your microphone and get the transcription.")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone"], type="filepath", label="Record Audio"
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(
                    label="Transcription Result",
                    lines=10,
                    max_lines=20,
                    placeholder="Your transcription will appear here...",
                )

        transcribe_btn.click(
            fn=transcribe_audio, inputs=[audio_input], outputs=[output_text]
        )

    return demo


if __name__ == "__main__":
    check_env()
    demo = create_gradio_interface()
    demo.launch(share=True)
