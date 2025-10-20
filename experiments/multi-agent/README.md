# Multi-Agent Orchestration with LangGraph

This experiment explores a Multi-Agent architecture where specialized AI agents work together under the guidance of a central supervisor. The supervisor acts as an intelligent coordinator, managing communication between agents and strategically delegating tasks based on each agent's expertise and the specific requirements of the problem at hand.

In this experiment, you'll create a multi-agent system powered by LangGraph with two agents — a research and a math expert.

The web search agent is using [Tavily](https://www.tavily.com/), visit their website to get an API key.

## Start the FlexAI endpoints

Create the FlexAI secret that contains your HF token in order to access the inference models:

```bash
# Enter your HF token value when prompted
flexai secret create hf-token
```

Export your Tavily API key:

```bash
export TAVILY_API_KEY=<TAVILY_API_KEY>
```

Start the FlexAI endpoint of the LLM:

```bash
LLM_INFERENCE_NAME=qwen-llm
export LLM_MODEL_NAME=Qwen/Qwen2.5-32B-Instruct
flexai inference serve $LLM_INFERENCE_NAME --hf-token-secret hf-token -- --model=$LLM_MODEL_NAME --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384
```

Store the returned Inference Endpoint API KEY and Endpoint URL:

```bash
export LLM_API_KEY=<INFERENCE_ENDPOINT_API_KEY>
export LLM_URL=$(flexai inference inspect $LLM_INFERENCE_NAME --json | jq .config.endpointUrl -r)
```

> You'll notice the last `export` line uses the `jq` tool to extract the value of `endpointUrl` from the JSON output of the `inspect` command.
>
> If you don't have it already, you can get `jq` from its official website: [https://jqlang.org/](https://jqlang.org/)

## Setup

1. Navigate to the experiment directory:

   ```bash
   cd code/multi-agent/
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python main.py
   ```

4. **Interact with the Multi-Agent System**

   When prompted, enter your question. The system will automatically route it to the appropriate agents (research and math experts).

   **Research + Math Question:**

   ```
   In 2025, how old are Trump and Macron? Also sum their ages.
   ```

   **Expected Output:**

   ```
   In 2025, Donald Trump will be 79 years old and Emmanuel Macron will be 48 years old.
   The sum of their ages in 2025 will be 127 years.
   ```
