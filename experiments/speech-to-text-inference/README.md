# Speech-to-Text Application

This Speech-to-Text application provides an interactive interface for users to record audio messages using their microphone and receive accurate transcriptions.

## Start the FlexAI endpoints

Create the FlexAI secret that contains your HF token in order to access the inference models:

```bash
# Enter your HF token value when prompted
flexai secret create hf-token
```

Start the FlexAI endpoint of the LLM:

```bash
LLM_INFERENCE_NAME=speech2text
flexai inference serve $LLM_INFERENCE_NAME --hf-token-secret hf-token --runtime vllm-nvidia-0.10.1 -- --model=openai/whisper-large-v3
```

Store the returned Inference Endpoint API KEY and Endpoint URL:

```bash
export LLM_API_KEY=<store the given API key>
export LLM_URL=$(flexai inference inspect $LLM_INFERENCE_NAME -j | jq .config.endpointUrl -r)
```

> You'll notice the last `export` line uses the `jq` tool to extract the value of `endpointUrl` from the JSON output of the `inspect` command.
>
> If you don't have it already, you can get `jq` from its official website: [https://jqlang.org/](https://jqlang.org/)

## Setup

1. Navigate to the speech-to-text directory:

   ```bash
   cd code/speech-to-text/
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python main.py
   ```

   The application will start and display two URLs:
   - **Local URL**: For local access (e.g., `http://127.0.0.1:7860`)
   - **Public URL**: For sharing (e.g., `https://xxxxxxxxxx.gradio.live`)

## Usage

1. **Access the Interface**: Open the Gradio interface in your web browser. To avoid any microphone access permission errors, prefer to use the **public URL** rather than the local one.

2. **Record Audio**:
   - Click the **record icon** to start recording
   - Speak your message
   - Click **stop** when finished recording

3. **Get Transcription**: Click the **"Transcribe"** button to process your audio and receive the text transcription.

4. **View Results**: The transcribed text will appear in the results panel on the right side of the interface.
