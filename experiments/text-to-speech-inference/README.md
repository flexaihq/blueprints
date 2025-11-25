# Text-to-Speech Inference with Flexai

This experiment demonstrates how to deploy and use a text-to-speech model (Kokoro) using Flexai's inference serving capabilities.

## Prerequisites

Before starting, make sure you have:

- A Flexai account with access to the platform
- The `flexai` CLI installed and configured

## Start the FlexAI Inference Endpoint

Start the FlexAI endpoint for the Kokoro text-to-speech model:

```bash
INFERENCE_NAME=text-to-speech
flexai inference serve $INFERENCE_NAME --runtime flexserve -- --model hexgrad/Kokoro-82M --task text-to-speech --lang-code a
```

This command will:

- Create an inference endpoint named `text-to-speech`
- Use the `flexserve` runtime optimized for text-to-speech tasks
- Load the Kokoro model for natural voice synthesis
- Configure it for English language text-to-speech generation

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

## Generate Speech

Now you can generate speech by making HTTP POST requests to your endpoint. Here is an example:

```bash
curl -X POST \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": "Welcome to FlexAI! Produce lifelike speech and spin up production inference endpoints in seconds.",
    "parameters": {
      "voice": "af_heart"
    }
  }' \
  -o welcome.wav \
  "$INFERENCE_URL/v1/speeches/generations"
```

## Parameters Explanation

The API accepts the following parameters:

- **inputs**: The text you want to convert to speech
- **voice**: The voice model to use for synthesis

## Supported Languages

You can configure different languages by changing the `--lang-code` parameter when starting the inference endpoint. The voice model to use is specified in the `voice` parameter of your request.

For a complete list of supported languages and available voices, check the [Kokoro GitHub repository](https://github.com/hexgrad/kokoro)
