# Text-to-Video Inference with Flexai

This experiment demonstrates how to deploy and use a text-to-video model (Wan2.2-TI2V-5B) using Flexai's inference serving capabilities.

## About Wan2.2

Wan2.2 is developed by Alibaba. It represents a major upgrade to visual generative models, now open-sourced with more powerful capabilities, better performance, and superior visual quality making it an excellent choice for text-to-video applications.

## Prerequisites

Before starting, make sure you have:

- A Flexai account with access to the platform
- A Hugging Face token with access to the `Wan-AI/Wan2.2-TI2V-5B-Diffusers` model
- The `flexai` CLI installed and configured

## Setup FlexAI Secret for Hugging Face Token

First, create a FlexAI secret that contains your Hugging Face token to access the inference model:

```bash
# Enter your HF token value when prompted
flexai secret create MY_HF_TOKEN
```

> **Note**: Make sure your Hugging Face token has access to the `Wan-AI/Wan2.2-TI2V-5B-Diffusers` model. You may need to accept the model's license terms on Hugging Face first.

## Start the FlexAI Inference Endpoint

Start the FlexAI endpoint for the Wan2.2-TI2V-5B model:

```bash
INFERENCE_NAME=wan-text-to-video
flexai inference serve $INFERENCE_NAME --runtime flexserve --hf-token-secret MY_HF_TOKEN -- --task text-to-video --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --quantization-config bitsandbytes_4bit
```

This command will:

- Create an inference endpoint named `wan-text-to-video`
- Use the `flexserve` runtime optimized for text-to-video tasks
- Load the Wan2.2-TI2V-5B model from Hugging Face
- Configure it for text-to-video generation with 4-bit quantization for memory efficiency

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

## Generate Videos

Now you can generate videos by making HTTP POST requests to your endpoint. Here's an example that generates a high-quality video of anthropomorphic cats boxing:

```bash
curl -X POST \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "parameters": {
      "num_frames": 81,
      "height": 704,
      "width": 1280,
      "num_inference_steps": 20,
      "guidance_scale": 4.0,
      "guidance_scale_2": 3.0
    }
  }' \
  -o boxing_cats.mp4 \
  "$INFERENCE_URL/v1/videos/generations"
```

This will save the generated video as `boxing_cats.mp4` in your current directory.

## Parameters Explanation

The API accepts the following parameters:

- **inputs**: The text prompt describing the video you want to generate
- **num_frames**: Number of frames in the output video (affects duration)
- **height**: Output video height in pixels
- **width**: Output video width in pixels
- **num_inference_steps**: Number of denoising steps (higher = better quality but slower)
- **guidance_scale** and **guidance_scale_2**: As you increase the value, the model tries harder to match your prompt.
