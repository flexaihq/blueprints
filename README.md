# FlexAI Experiments

This repository provides **a set of different experiments** designed for you to try out and explore **FlexAI**. These experiments range from running your very first training job, to fine-tuning language, diffusion, and text-to-speech models using techniques like QLoRA and LoRA, as well as integrating FlexAI with other platforms, such as experiment trackers.

## Getting Started

### Prerequisites

**FlexAI CLI**: Install the FlexAI CLI by following the steps shown in the [Installing the FlexAI CLI](https://docs.flex.ai/cli/installation/) guide.

## The Experiments

The following table lists out the experiments available in this repository. Each experiment is designed to walk you through a specific use case and contains its required code, as well as detailed instructions on how to run it on FlexAI:

| No. | Section                                                                                                                          | Description                                                                                                                |
| --- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 1   | [A Simple Training Job on FlexAI](/experiments/running-a-simple-training-job/README.md)                                          | Step-by-step guide to get your first Training Job on FlexAI running                                                        |
| 2   | [A Simple Distributed Data Parallel (DDP) Training Job on FlexAI](/experiments/running-a-simple-training-job-with-ddp/README.md) | Demonstrates that you only need to add 2 flags in order to start a DDP Training Job                                        |
| 3   | [Resuming a Training Job from a Checkpoint](/experiments/continuing-a-training-job-from-a-checkpoint/README.md)                  | Learn how to resume a Training Job from a previously saved checkpoint                                                      |
| 4   | [Streaming Large Datasets During a Training Job](/experiments/streaming-datasets/README.md)                                      | Train a model on a large dataset using streaming                                                                           |
| 5   | [Training Job & Experiment Tracking](/experiments/integrating-a-experiment-tracker/README.md)                                    | Using Weights and Biases with FlexAI for _experiment tracking_                                                             |
| 6   | [Fine-Tuning a Language Model with QLoRA](/experiments/qlora-ft-on-a-language-model/README.md)                                   | Fine-tune a causal language model efficiently using QLoRA                                                                  |
| 7   | [Fine-Tuning a Diffusion Model with LoRA](/experiments/lora-ft-on-a-diffusion-model/README.md)                                   | Fine-tune a diffusion model efficiently using LoRA                                                                         |
| 8   | [Fine-Tuning a Text-to-Speech Model](/experiments/ft-on-a-tts-model/README.md)                                                   | Fine-tune a text-to-speech (TTS) model                                                                                     |
| 9   | [Fine-Tuning a language Model using _Flash Attention_](/experiments/flash-attention-ft-on-a-language-model/README.md)            | Fine-tune a causal language model efficiently using the [flash-attn package](https://github.com/Dao-AILab/flash-attention) |
| 10  | [Train and Serve a French LLM on FlexAI with LlamaFactory](/experiments/llama-factory/README.md)                                 | Fine-tune Qwen2.5-7B on French data using LlamaFactory and deploy as a production-ready inference endpoint                 |
| 11  | [Train and Serve Language Models with Axolotl](/experiments/axolotl/README.md)                                                   | Fine-tune language models on domain-specific data using Axolotl framework with custom dataset configurations and FSDP     |
| 12  | [Reinforcement Learning Fine-Tuning with EasyR1](/experiments/easyR1/README.md)                                                  | Train language models with RL using GRPO and DAPO algorithms for enhanced reasoning capabilities                           |
| 13  | [Language Model Evaluation with LM-Evaluation-Harness](/experiments/lm-evaluation-harness/README.md)                             | Comprehensive evaluation of language models across 300+ tasks and benchmarks using the LM-Evaluation-Harness framework    |
| 14  | [RAG Application with LangChain and FlexAI Inference Endpoints](/experiments/rag-application/README.md)                          | Interactive interface for users to ask questions based on provided documents using Retrieval-Augmented Generation         |
| 15  | [Speech-to-Text Application Using FlexAI Inference Endpoints](/experiments/speech-to-text-inference/README.md)                   | Interactive interface for recording audio messages and receiving transcriptions                                            |
| 16  | [Multi-Agent Orchestration with LangGraph](/experiments/multi-agent/README.md)                                                   | Build a multi-agent system where specialized AI agents work together under a central supervisor                            |
| 17  | [Text-to-Image Inference with FlexAI Endpoints](/experiments/text-to-image-inference/README.md)                                  | Deploy and use Stable Diffusion 3.5 Large for high-quality image generation via FlexAI inference endpoints                |
| 18  | [Text-to-Audio Inference with FlexAI Endpoints](/experiments/text-to-audio-inference/README.md)                                  | Deploy and use Stable Audio Open 1.0 for high-quality audio generation via FlexAI inference endpoints                     |
| 19  | [Text-to-Video Inference with FlexAI Endpoints](/experiments/text-to-video-inference/README.md)                                  | Deploy and use Wan2.2-T2V-A14B for high-quality video generation via FlexAI inference endpoints                           |


---

## Keep in mind

### Notes on HuggingFace Accelerate Integration

This repository includes experiments that utilize the **HuggingFace Accelerate library** for efficient training.

The **FlexAI CLI** simplifies running training scripts by automatically determining the appropriate execution method:

- **`python`**: Used for single-accelerator training.
- **`torchrun`**: Automatically used for multi-accelerator distributed training.

If you're accustomed to using the **`accelerate launch`** command, you can seamlessly run the same scripts on FlexAI without modification. Simply provide the script to FlexAI, and it will handle execution.

As highlighted in the [Accelerate documentation](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch#using-accelerate-launch), the `accelerate launch` command is optional. Instead, the Accelerate functionality integrates directly into your script, making it compatible with other launchers like `torchrun`.

> [!NOTE]
>
> Unlike `accelerate launch`, `torchrun` does not use the YAML configuration file generated by `accelerate config`.
>
> If your training setup relies on specific configurations from the YAML file, you may need to adjust your script to explicitly define these settings using the `Accelerator` class.
>
> By doing so, you ensure seamless execution across different environments while maintaining flexibility for various training setups.
