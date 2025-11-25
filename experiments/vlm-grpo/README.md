# GRPO Training on Vision-Language Models

This experiment demonstrates how to perform Group Relative Policy Optimization (GRPO) training on a Vision-Language Model (VLM) using the TRL library. GRPO is a reinforcement learning technique that optimizes the model based on relative rewards within groups of generated responses.

We will use `Qwen2.5-VL-3B-Instruct` as our base VLM and apply LoRA (Low-Rank Adaptation) for efficient fine-tuning. The training leverages vLLM for fast inference during the policy optimization phase and DeepSpeed ZeRO-3 for distributed training.

## Prerequisites

### Connect to GitHub (if needed)

If you haven't already connected FlexAI to GitHub, you'll need to set up a code registry connection:

```bash
flexai code-registry connect
```

This will allow FlexAI to pull repositories directly from GitHub using the `-u` flag in training commands.

### Create Secrets

To access the Qwen2.5-VL-3B-Instruct model, you may need authentication with your HuggingFace account depending on the model's access requirements.

Use the [`flexai secret create` command](https://docs.flex.ai/cli/commands/secret/) to store your _HuggingFace Token_ as a secret. Replace `<HF_AUTH_TOKEN_SECRET_NAME>` with your desired name for the secret:

```bash
flexai secret create <HF_AUTH_TOKEN_SECRET_NAME>
```

Then paste your _HuggingFace Token_ value.

## Training

To start the GRPO Training Job, run the following command:

```bash
flexai training run vlm-grpo-training --repository-url https://github.com/flexaihq/blueprints \
  --requirements-path code/vlm-grpo/requirements.txt \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --runtime pytorch-28-vllm-0110-nvidia \
  --nodes 1 --accels 2 \
  -- accelerate launch \
    --config_file=code/vlm-grpo/deepspeed_zero3.yaml \
    code/vlm-grpo/grpo_vlm.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir /output-checkpoint/grpo-Qwen2.5-VL-3B-Instruct \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --lora_target_modules "q_proj" "v_proj" \
    --log_completions \
    --logging_steps 200
```

### Key Arguments Explained

- `--model_name_or_path`: The base VLM model to fine-tune (Qwen2.5-VL-3B-Instruct)
- `--output_dir`: Directory where checkpoints will be saved
- `--learning_rate`: Learning rate for optimization (1e-5)
- `--gradient_checkpointing`: Enables gradient checkpointing to reduce memory usage
- `--dtype bfloat16`: Uses bfloat16 precision for training
- `--max_prompt_length`: Maximum length of input prompts (2048 tokens)
- `--max_completion_length`: Maximum length of generated completions (1024 tokens)
- `--use_vllm`: Enables vLLM for fast inference during policy optimization
- `--vllm_mode colocate`: Runs vLLM on the same GPUs as training
- `--use_peft`: Enables LoRA for parameter-efficient fine-tuning
- `--lora_target_modules`: Specifies which modules to apply LoRA (q_proj, v_proj)
- `--log_completions`: Logs generated completions during training

### Configuration Files

The experiment uses a DeepSpeed ZeRO-3 configuration file (`code/vlm-grpo/deepspeed_zero3.yaml`) that specifies:

- ZeRO Stage 3 for memory-efficient distributed training
- 2 processes (GPUs) for multi-GPU training
- Mixed precision training with bfloat16

## Monitoring the Training Job

You can check the status and life cycle events of your Training Job by running:

```bash
flexai training inspect vlm-grpo-training
```

Additionally, you can view the logs of your Training Job by running:

```bash
flexai training logs vlm-grpo-training
```

## Fetching the Trained Model

Once the Training Job completes successfully, you can list all the produced checkpoints:

```bash
flexai training checkpoints vlm-grpo-training
```

Download a checkpoint with:

```bash
flexai checkpoint fetch "<CPKT-ID>"
```

The checkpoint will contain the LoRA adapters that can be merged with the base model for inference.

## Additional Resources

- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [vLLM Documentation](https://docs.vllm.ai/)
