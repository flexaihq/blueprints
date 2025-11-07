# Text-to-Image Inference with Flexai

This experiment demonstrates how to deploy and use a text-to-image model (Stable Diffusion 3.5 Large) using Flexai's inference serving capabilities.

## Prerequisites

Before starting, make sure you have:

- A Flexai account with access to the platform
- A Hugging Face token with access to the `stabilityai/stable-diffusion-3.5-large` model
- The `flexai` CLI installed and configured

## Setup FlexAI Secret for Hugging Face Token

First, create a FlexAI secret that contains your Hugging Face token to access the inference model:

```bash
# Enter your HF token value when prompted
flexai secret create MY_HF_TOKEN
```

> **Note**: Make sure your Hugging Face token has access to the `stabilityai/stable-diffusion-3.5-large` model. You may need to accept the model's license terms on Hugging Face first.

## Start the FlexAI Inference Endpoint

Start the FlexAI endpoint for the Stable Diffusion 3.5 Large model:

```bash
INFERENCE_NAME=stable-diffusion-35-large
flexai inference serve $INFERENCE_NAME --runtime flexserve --hf-token-secret MY_HF_TOKEN -- --task text-to-image --model stabilityai/stable-diffusion-3.5-large
```

This command will:

- Create an inference endpoint named `stable-diffusion-35-large`
- Use the `flexserve` runtime optimized for text-to-image tasks
- Load the Stable Diffusion 3.5 Large model from Hugging Face
- Configure it for text-to-image generation

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

## Generate Images

Now you can generate images by making HTTP POST requests to your endpoint. Here's an example that generates a high-quality image of a golden retriever:

```bash
curl -X POST \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": "A highly detailed, realistic photograph of a happy golden retriever sitting on a sofa, 8k, cinematic",
    "parameters": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 20,
      "seed": 42,
      "negative_prompt": "blurry, low quality, distorted, deformed, mutated, extra limbs, cropped, out of frame, ugly, unrealistic, cartoon, drawing, painting, watermark, text, logo, nsfw"
    }
  }' \
  -o dog.png \
  "$INFERENCE_URL/v1/images/generations"
```

This will save the generated image as `dog.png` in your current directory.

## Parameters Explanation

The API accepts the following parameters:

- **inputs**: The text prompt describing the image you want to generate
- **height**: Output image height in pixels
- **width**: Output image width in pixels
- **num_inference_steps**: Number of denoising steps (higher = better quality but slower)
- **seed**: Random seed for reproducible results
- **negative_prompt**: Description of what you don't want in the image (helps improve quality)
