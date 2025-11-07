# Text-to-Audio Inference with Flexai

This experiment demonstrates how to deploy and use a text-to-audio model (Stable Audio Open 1.0) using Flexai's inference serving capabilities.

## Prerequisites

Before starting, make sure you have:

- A Flexai account with access to the platform
- A Hugging Face token with access to the `stabilityai/stable-audio-open-1.0` model
- The `flexai` CLI installed and configured

## Setup FlexAI Secret for Hugging Face Token

First, create a FlexAI secret that contains your Hugging Face token to access the inference model:

```bash
# Enter your HF token value when prompted
flexai secret create MY_HF_TOKEN
```

> **Note**: Make sure your Hugging Face token has access to the `stabilityai/stable-audio-open-1.0` model. You may need to accept the model's license terms on Hugging Face first.

## Start the FlexAI Inference Endpoint

Start the FlexAI endpoint for the Stable Audio Open 1.0 model:

```bash
INFERENCE_NAME=stable-audio-open
flexai inference serve $INFERENCE_NAME --runtime flexserve --hf-token-secret MY_HF_TOKEN -- --task text-to-audio --model stabilityai/stable-audio-open-1.0
```

This command will:

- Create an inference endpoint named `stable-audio-open`
- Use the `flexserve` runtime optimized for text-to-audio tasks
- Load the Stable Audio Open 1.0 model from Hugging Face
- Configure it for text-to-audio generation

## Get Endpoint Information

Once the endpoint is deployed, you'll see the API key displayed in the output. Store it in an environment variable:

```bash
export INFERENCE_API_KEY=<API_KEY_FROM_ENDPOINT_CREATION_OUTPUT>
```

Then retrieve the endpoint URL:

```bash
export INFERENCE_URL=$(flexai inference inspect $INFERENCE_NAME -j | jq .config.endpointUrl -r)
```

> You'll notice these `export` lines use the `jq` tool to extract values from the JSON output of the `inspect` command.
>
> If you don't have it already, you can get `jq` from its official website: [https://jqlang.org/](https://jqlang.org/)

## Generate Audio

Now you can generate audio by making HTTP POST requests to your endpoint. Here are some examples:

### Example 1: Relaxing Music

```bash
curl -X POST \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": "Relaxing piano music with soft ambient sounds, calm and peaceful",
    "parameters": {
      "audio_end_in_s": 10.0,
      "num_inference_steps": 200,
      "guidance_scale": 7.0,
      "seed": 42
    }
  }' \
  -o relaxing_music.wav \
  "$INFERENCE_URL/v1/audios/generations"
```

### Example 2: Nature Soundscape

```bash
curl -X POST \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": "Forest ambience with birds chirping and a gentle stream flowing",
    "parameters": {
      "audio_end_in_s": 15.0,
      "num_inference_steps": 200,
      "guidance_scale": 7.0,
      "negative_prompt": "distorted, low quality, muffled",
      "seed": 123
    }
  }' \
  -o nature_sounds.wav \
  "$INFERENCE_URL/v1/audios/generations"
```

### Example 3: Electronic Beat

```bash
curl -X POST \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": "Upbeat electronic music with synthesizers and energetic drums, 128 bpm",
    "parameters": {
      "audio_end_in_s": 20.0,
      "num_inference_steps": 200,
      "guidance_scale": 7.0,
      "seed": 456
    }
  }' \
  -o electronic_beat.wav \
  "$INFERENCE_URL/v1/audios/generations"
```

These will save the generated audio files in your current directory.

## Parameters Explanation

The API accepts the following parameters:

- **inputs**: The text prompt describing the audio you want to generate
- **audio_length_in_s**: Output audio duration in seconds (typically up to 47 seconds for Stable Audio Open)
- **num_inference_steps**: Number of denoising steps (higher = better quality but slower, recommended: 100-200)
- **guidance_scale**: As you increase the value, the model tries harder to match your prompt.
- **negative_prompt**: Description of what you don't want in the audio (helps improve quality)
- **seed**: Random seed for reproducible results
