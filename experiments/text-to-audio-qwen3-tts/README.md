# Text-to-Audio Inference with Flexai

This experiment demonstrates how to deploy and use a text-to-audio model (Qwen3-TTS) using Flexai's inference serving capabilities.


## Setup

The demo code for this experiment is located at `code/qwen3-tts` but it is advisable to follow the steps below before jumping to the full demo.


## Prerequisites

Before starting, make sure you have:

- A Flexai account with access to the platform
- The `flexai` CLI installed and configured

### Start the FlexAI Inference Endpoint for voice cloning

Base model capable of 3-second rapid voice clone from user audio input; can be used for fine-tuning (FT) other models.

Start the FlexAI endpoint for the Qwen3-TTS-12Hz-1.7B-Base model:

```bash
INFERENCE_NAME=Qwen3-base
flexai inference serve $INFERENCE_NAME --runtime vllm-omni-0.14.0 -- Qwen/Qwen3-TTS-12Hz-1.7B-Base  --stage-configs-path /workspace/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml --omni --trust-remote-code --enforce-eager
```

This command will:

- Create an inference endpoint named `Qwen3-base`
- Use the `vllm-omni` runtime
- Load the Qwen3-TTS-12Hz-1.7B-Base model from Hugging Face

### Get Endpoint Information

Once the endpoint is deployed, you'll see the API key displayed in the output. Store it in an environment variable:

```bash
export INFERENCE_API_KEY_BASE=<API_KEY_FROM_ENDPOINT_CREATION_OUTPUT>
```

Then retrieve the endpoint URL:

```bash
export INFERENCE_URL_BASE=$(flexai inference inspect $INFERENCE_NAME -j | jq .config.endpointUrl -r)
```

> You'll notice these `export` lines use the `jq` tool to extract values from the JSON output of the `inspect` command.
>
> If you don't have it already, you can get `jq` from its official website: [https://jqlang.org/](https://jqlang.org/)

### Generate Audio

Now you can clone a voice by making HTTP POST requests to your endpoint. Here is an example:


```bash
curl -v -X POST -H "Authorization: Bearer $INFERENCE_API_KEY_BASE" \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you? I am so excited. For Sure!",
	    "ref_text": "Exactly. And, you know, one of the things we have, we ve been wondering about why some of these companies in the last seven, eight years are in the graveyard. And one of the challenges was they all went after CUDA or NVIDIA silicon.",
	    "ref_audio": "https://tmpfiles.org/26240870/sample_1.wav",
	    "language": "Auto",
	    "task_type": "Base"
    }' -o excited.wav \
         $INFERENCE_URL_BASE/v1/audio/speech
```
### Parameters Explanation

The API accepts the following parameters:

- **inputs**: The text prompt describing the text you want to generate
- **ref_text**: The transcription of the reference audio file (the voice you want to clone)
- **ref_audio**: A URL pointing to the reference audio file (must be in WAV format and less than 10 seconds long)
- **language**: Language of the input text (set to "Auto" for automatic detection)
- **task_type**: Set to "Base" for voice cloning


### Start the FlexAI Inference Endpoint for using custom voice

Provides style control over target timbres via user instructions; supports 9 premium timbres covering various combinations of gender, age, language, and dialect.

Start the FlexAI endpoint for the Qwen3-TTS-12Hz-1.7B-CustomVoice model:

```bash
INFERENCE_NAME=Qwen3-custom-voice
flexai inference serve $INFERENCE_NAME --runtime vllm-omni-0.14.0 -- Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice  --stage-configs-path /workspace/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml --omni --trust-remote-code --enforce-eager
```

This command will:

- Create an inference endpoint named `Qwen3-custom-voice`
- Use the `vllm-omni` runtime
- Load the Qwen3-TTS-12Hz-1.7B-CustomVoice model from Hugging Face

### Get Endpoint Information

Once the endpoint is deployed, you'll see the API key displayed in the output. Store it in an environment variable:

```bash
export INFERENCE_URL_TTS=<API_KEY_FROM_ENDPOINT_CREATION_OUTPUT>
```

Then retrieve the endpoint URL:

```bash
export INFERENCE_URL_TTS=$(flexai inference inspect $INFERENCE_NAME -j | jq .config.endpointUrl -r)
```

### Generate Audio

Now you can create audio with the provided custome voice by making HTTP POST requests to your endpoint. Here is an example:


```bash
curl -v -X POST \
 -H "Authorization: Bearer $INFERENCE_API_KEY_TTS" \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English"
    }' \
     -o output.wav \
      "$INFERENCE_URL_TTS/v1/audio/speech"
```


```bash
curl -X POST "$INFERENCE_URL_TTS/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{
        "input": "I am so excited!",
        "voice": "vivian",
        "language": "English",
        "instructions": "Speak with great enthusiasm"
    }' --output excited.wav
```


### Parameters Explanation

The API accepts the following parameters:

- **inputs**: The text prompt describing the text you want to generate
- **voice**: The voice you want to use (you can list the available voices by making a GET request to the endpoint's `/v1/audio/voices` path)
- **instructions**: How the model should speak the text (e.g., "Speak with great enthusiasm", "Speak like a news anchor", etc.)
- **language**: Language of the input text (set to "Auto" for automatic detection)


## Demo App

The demo app allows you to easily test the endpoints you just created. You can find it in the `code/qwen3-tts` directory.

In the same directory you will find scripts to start the inference endpoints for both the base and custom voice models.
Make sure to start both endpoints before running the demo app:

### Start Qwen3-TTS-12Hz-1.7B-Base model

```shell
source ./START/run-qwen3-tts-cloneVoice.sh
```

### Start Qwen3-TTS-12Hz-1.7B-CustomVoice

```shell
source ./START/run-qwen3-tts-customVoice.sh
```

### Start the demo app

#### Using uv (recommended)

```
 uv run app.py
```

#### Using pip

```
python3.13 -m venv venv
source ./venv/bin/activate
uv pip install -r requirements.txt
```

then you can start the demo with:
```
python app.py
```
