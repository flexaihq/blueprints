# Reinforcement Learning Fine-tuning with EasyR1 on FlexAI

This experiment demonstrates how to use FlexAI to fine-tune language models using reinforcement learning (RL) techniques with [EasyR1](https://github.com/hiyouga/EasyR1), a framework for training reasoning-capable models using GRPO (Group Relative Policy Optimization), DAPO, and REINFORCE algorithms.

For illustration purposes, we'll fine-tune the `Qwen2.5-7B-Instruct` model on mathematical reasoning tasks using the `math12k` dataset with GRPO algorithm to improve reasoning capabilities.

> **Note**: If you haven't already connected FlexAI to GitHub, run `flexai code-registry connect` to set up a code registry connection. This allows FlexAI to pull repositories directly using the repository URL in training commands.

## Quick Start

Run GRPO training on Qwen2.5-7B with this single command:

```bash
flexai training run grpo \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --env FORCE_TORCHRUN=1 \
  --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/easyR1/requirements.txt \
  --runtime pytorch-28-vllm-0110-nvidia \
  -- python3 -m verl.trainer.main \
      config=code/easyR1/config.yaml \
      worker.actor.model.model_path=Qwen/Qwen2.5-7B-Instruct
```

Replace `<WANDB_API_KEY_SECRET_NAME>` and `<HF_AUTH_TOKEN_SECRET_NAME>` with your actual values.

## What is EasyR1?

EasyR1 is a reinforcement learning framework specifically designed for training language models with enhanced reasoning capabilities. It implements several RL algorithms optimized for LLMs:

- **GRPO (Group Relative Policy Optimization)**: Efficient policy optimization using group-based advantage estimation
- **DAPO (Data-Augmented Policy Optimization)**: Enhanced training with data augmentation strategies
- **REINFORCE**: Classic policy gradient method for LLM fine-tuning

The framework is built on top of [VERL (Versatile Efficient Reinforcement Learning)](https://github.com/volcengine/verl), providing distributed training capabilities with FSDP and vLLM integration.

## Directory Structure

The `code/easyR1/` directory contains:

- `config.yaml` - Main GRPO training configuration
- `format_prompt/` - Jinja templates for prompt formatting
- `reward_function/` - Custom reward scoring functions

For baseline training scripts and additional examples, refer to the [EasyR1 GitHub repository](https://github.com/hiyouga/EasyR1).

## Step 1: Understand the Configuration

EasyR1 uses a comprehensive YAML configuration file that controls all aspects of RL training. The main configuration file is located at `code/easyR1/config.yaml` in this repository.

### Key Configuration Sections

#### Data Configuration

```yaml
data:
  train_files: hiyouga/math12k@train
  val_files: hiyouga/math12k@test
  prompt_key: problem
  answer_key: answer
  format_prompt: ./code/easyR1/format_prompt/math.jinja
  max_prompt_length: 2048
  max_response_length: 2048
  rollout_batch_size: 512
```

#### Algorithm Settings

```yaml
algorithm:
  adv_estimator: grpo  # GRPO, DAPO, or REINFORCE
  use_kl_loss: true
  kl_coef: 1.0e-2
```

#### Worker Configuration

```yaml
worker:
  actor:
    model:
      model_path: Qwen/Qwen2.5-7B-Instruct
    optim:
      lr: 1.0e-6
  rollout:
    n: 5  # number of rollout samples per prompt
    temperature: 1.0
  reward:
    reward_type: batch
    reward_function: ./code/easyR1/reward_function/math.py:compute_score
```

## Step 2: Reference Baseline Examples

For pre-configured training scripts and baseline examples, refer to the [EasyR1 repository](https://github.com/hiyouga/EasyR1). The repository provides multiple baseline configurations for different models and tasks:

### Available Baselines (in EasyR1 repo)

- **Mathematical Reasoning**: `qwen2_5_7b_math_grpo.sh`, `qwen3_4b_math_grpo.sh`
- **Geometric Reasoning (Vision-Language)**: `qwen2_5_vl_7b_geo3k_grpo.sh`, `qwen2_5_vl_7b_geo3k_dapo.sh`, `qwen2_5_vl_7b_geo3k_reinforce.sh`
- **Multi-Image Tasks**: `qwen2_5_vl_7b_multi_image.sh`

You can adapt these examples to work with FlexAI by following the training commands in this README.

## Step 3: Customize Your Configuration

For your specific use case, you may want to create a custom configuration. Here's how to customize the `config.yaml`:

### Custom Dataset

Replace the dataset configuration:

```yaml
data:
  train_files: your-username/your-dataset@train
  val_files: your-username/your-dataset@test
  prompt_key: question  # adjust based on your dataset
  answer_key: solution  # adjust based on your dataset
```

### Custom Reward Function

Create your own reward function in `code/easyR1/reward_function/custom.py`:

```python
def compute_score(prompts, responses, answers):
    """
    Args:
        prompts: List of input prompts
        responses: List of model responses
        answers: List of ground truth answers

    Returns:
        List of reward scores (float)
    """
    scores = []
    for response, answer in zip(responses, answers):
        # Your custom reward logic here
        score = your_evaluation_function(response, answer)
        scores.append(score)
    return scores
```

Then update the config to reference your custom reward function:

```yaml
worker:
  reward:
    reward_function: ./code/easyR1/reward_function/custom.py:compute_score
```

### Custom Prompt Format

Create a custom Jinja template in `code/easyR1/format_prompt/custom.jinja`:

```jinja
{{ problem }}

Please solve this step by step and provide your final answer.
```

Update the config:

```yaml
data:
  format_prompt: ./code/easyR1/format_prompt/custom.jinja
```

## Create Secrets

To access HuggingFace models and datasets, you need a HuggingFace token.

Use the [`flexai secret create` command](https://docs.flex.ai/cli/commands/secret/) to store your _HuggingFace Token_ as a secret:

```bash
flexai secret create <HF_AUTH_TOKEN_SECRET_NAME>
```

Then paste your _HuggingFace Token_ API key value.

Use the same command to store your Weights & Biases (wandb) API key as a secret:

```bash
flexai secret create <WANDB_API_KEY_SECRET_NAME>
```

Then paste your Weights & Biases API key value.

## [Optional] Pre-fetch the Model

To speed up training and avoid downloading large models at runtime, you can pre-fetch your HuggingFace model to FlexAI storage:

1. **Create a HuggingFace storage provider:**

    ```bash
    flexai storage create HF-STORAGE --provider huggingface --hf-token-name <HF_AUTH_TOKEN_SECRET_NAME>
    ```

2. **Push the model checkpoint to your storage:**

    ```bash
    flexai checkpoint push qwen25-7b-instruct --storage-provider HF-STORAGE --source-path Qwen/Qwen2.5-7B-Instruct
    ```

## Training

For RL training with EasyR1, we recommend using **1 node (8 × H100 GPUs)** for 7B models to handle the actor, reference model, and rollout workers efficiently.

> **Repository Note**: The commands below use this repository which contains all necessary configuration files in the `code/easyR1/` directory.

### Standard Training: Mathematical Reasoning with GRPO

```bash
flexai training run grpo \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --env FORCE_TORCHRUN=1 \
  --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/easyR1/requirements.txt \
  --runtime pytorch-28-vllm-0110-nvidia \
  -- python3 -m verl.trainer.main \
      config=code/easyR1/config.yaml \
      worker.actor.model.model_path=Qwen/Qwen2.5-7B-Instruct
```

### Training with Model Prefetch

```bash
flexai training run grpo-prefetched \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint qwen25-7b-instruct \
  --env FORCE_TORCHRUN=1 \
  --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/easyR1/requirements.txt \
  --runtime pytorch-28-vllm-0110-nvidia \
  -- python3 -m verl.trainer.main \
      config=code/easyR1/config.yaml \
      worker.actor.model.model_path=/input-checkpoint/qwen25-7b-instruct
```

### Training with Custom Configuration

To use a modified configuration or different dataset, override config values:

```bash
flexai training run grpo-custom \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --env FORCE_TORCHRUN=1 \
  --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/easyR1/requirements.txt \
  --runtime pytorch-28-vllm-0110-nvidia \
  -- python3 -m verl.trainer.main \
      config=code/easyR1/config.yaml \
      worker.actor.model.model_path=Qwen/Qwen2.5-7B-Instruct \
      data.train_files=your-username/your-dataset@train \
      data.val_files=your-username/your-dataset@test \
      trainer.experiment_name=custom-experiment
```

## Monitoring Training Progress

You can check the status and lifecycle events of your Training Job:

```bash
flexai training inspect grpo
```

View the logs of your Training Job:

```bash
flexai training logs grpo
```

### Training Observability with Weights & Biases

EasyR1 supports Weights & Biases (wandb) integration for detailed training metrics visualization. The configuration already includes wandb logging:

```yaml
trainer:
  logger: ["file", "wandb"]
  project_name: easy_r1
  experiment_name: qwen2_5_7b_math_grpo
```

## Getting Training Checkpoints

Once the Training Job completes successfully, you can list all produced checkpoints:

```bash
flexai training checkpoints grpo
```

Look for checkpoints marked as `INFERENCE READY = true` - these are ready for serving.

## Serving the Trained Model

Deploy your RL-trained model directly from the checkpoint using FlexAI inference. Replace `<CHECKPOINT_ID>` with the ID from an inference-ready checkpoint:

```bash
flexai inference serve easyr1-reasoning-endpoint --checkpoint <CHECKPOINT_ID>
```

Monitor your inference endpoint status:

```bash
# List all inference endpoints
flexai inference list

# Get detailed endpoint information
flexai inference inspect easyr1-reasoning-endpoint

# Check endpoint logs
flexai inference logs easyr1-reasoning-endpoint
```

## Testing Your RL-Trained Model

Once the endpoint is running, you can test it with reasoning tasks. For our mathematical reasoning example, the model should demonstrate improved step-by-step reasoning and accurate problem-solving.

### Before and After Training Comparison

To illustrate the improvement from RL fine-tuning, here's a comparison using a math problem:

**Problem**: "If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"

**Base Model Response (Qwen2.5-7B-Instruct before RL training):**

```
The average speed is 60 mph.
```

_Issues: Correct answer but no reasoning steps shown_

**RL Fine-tuned Model Response (after GRPO training on math12k):**

```
Let me solve this step by step:

Step 1: Identify the given information
- Distance traveled = 120 miles
- Time taken = 2 hours

Step 2: Apply the speed formula
Speed = Distance / Time

Step 3: Calculate
Speed = 120 miles / 2 hours = 60 miles per hour

Therefore, the average speed of the train is 60 mph.
```

_Improvements: Clear reasoning steps, structured approach, educational value_

This demonstrates how RL training encourages the model to show its reasoning process, making it more reliable and transparent.

### Example API Call

```bash
curl -X POST "https://your-endpoint-url/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "Solve the following problem step by step: A rectangle has a length of 15 cm and a width of 8 cm. What is its area?",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

## Expected Results

After RL fine-tuning with EasyR1, your model should achieve:

- **Enhanced Reasoning**: Step-by-step problem-solving with clear explanations
- **Improved Accuracy**: Higher success rate on reasoning tasks
- **Better Generalization**: Ability to apply learned reasoning patterns to new problems
- **Structured Outputs**: More organized and educational responses

For mathematical reasoning tasks:

- **Explicit Step-by-Step Solutions**: Clear breakdown of problem-solving process
- **Higher Success Rate**: Improved accuracy on math benchmarks
- **Better Error Detection**: Ability to identify and correct mistakes

## Technical Details

### Training Configuration Breakdown

**Reinforcement Learning Components:**

- **Actor Model**: The model being trained (policy network)
- **Reference Model**: Frozen copy for KL divergence computation
- **Rollout Workers**: Generate multiple responses for each prompt (n=5)
- **Reward Function**: Evaluates response quality (custom per task)

**Distributed Training:**

- **FSDP (Fully Sharded Data Parallel)**: Efficient memory usage for large models
- **vLLM Integration**: Fast inference during rollout generation
- **Tensor Parallelism**: For rollout workers (size=2)

**Optimization:**

- **GRPO Algorithm**: Group-based advantage estimation for stable training
- **KL Penalty**: Prevents model from deviating too far from base model
- **Gradient Checkpointing**: Reduces memory usage during backpropagation

### Resource Requirements

**Recommended Configuration for Qwen2.5-7B:**

- **Nodes**: 1 node (sufficient for RL training with actor + reference + rollout)
- **Accelerators**: 8 × H100 GPUs per node
- **Memory**: ~400GB+ GPU memory total (actor, reference, and rollout workers)
- **Training Time**: ~8-12 hours for 15 epochs
- **Storage**: ~50GB for checkpoints

**Command Line Parameters Explained:**

- `FORCE_TORCHRUN=1`: Ensures proper distributed training setup
- `--runtime pytorch-28-vllm-0110-nvidia`: PyTorch 2.8 with vLLM 0.11.0 optimized for EasyR1
- `--repository-url`: Points to the FlexAI blueprints repository
- `config=code/easyR1/config.yaml`: Main configuration file path relative to repository root

### Key Configuration Parameters

**Data Settings:**

- `rollout_batch_size: 512`: Number of prompts per training iteration
- `max_prompt_length: 2048`: Maximum input length
- `max_response_length: 2048`: Maximum output length

**Algorithm Settings:**

- `adv_estimator: grpo`: Choice of RL algorithm
- `kl_coef: 1.0e-2`: Strength of KL penalty
- `use_kl_loss: true`: Enable KL divergence loss

**Training Settings:**

- `total_epochs: 15`: Number of training epochs
- `n_gpus_per_node: 8`: GPUs per node
- `val_freq: 5`: Validation every 5 epochs
- `save_freq: 5`: Save checkpoint every 5 epochs

### Scaling Options

- **For faster training**: Increase to 2 nodes (16 × H100)
- **For larger models**: Increase `tensor_parallel_size` for rollout
- **For better exploration**: Increase `rollout.n` (more samples per prompt)
- **For memory efficiency**: Enable CPU offloading (`enable_cpu_offload: true`)
- **For different tasks**: Modify reward function and prompt templates

## Advanced Examples

### Vision-Language Model with Geometric Reasoning

For vision-language models, you'll need to use a VL model and the geometry dataset:

```bash
flexai training run grpo-VL-Geo \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --env FORCE_TORCHRUN=1 \
  --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/easyR1/requirements.txt \
  --runtime pytorch-28-vllm-0110-nvidia \
  -- python3 -m verl.trainer.main \
      config=code/easyR1/config.yaml \
      worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
      data.train_files=hiyouga/geometry3k@train \
      data.val_files=hiyouga/geometry3k@test \
      data.format_prompt=./code/easyR1/format_prompt/r1v.jinja \
      worker.reward.reward_function=./code/easyR1/reward_function/r1v.py:compute_score \
      trainer.experiment_name=qwen2_5_vl_7b_geo3k_grpo
```

### Using DAPO Algorithm

For DAPO (Data-Augmented Policy Optimization), change the algorithm settings:

```bash
flexai training run Dapo-14B \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --env FORCE_TORCHRUN=1 \
  --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/easyR1/requirements.txt \
  --runtime pytorch-28-vllm-0110-nvidia \
  -- python3 -m verl.trainer.main \
      config=code/easyR1/config.yaml \
      worker.actor.model.model_path=Qwen/Qwen3-14B \
      algorithm.adv_estimator=dapo \
      algorithm.online_filtering=true \
      data.train_files=hiyouga/dapo17k@train \
      data.val_files=hiyouga/dapo17k@test \
      data.format_prompt=./code/easyR1/format_prompt/dapo.jinja \
      worker.reward.reward_function=./code/easyR1/reward_function/dapo.py:compute_score \
      trainer.experiment_name=qwen3_14b_dapo17k_dapo
```

## Troubleshooting

**Common Issues:**

**Training Job Fails to Start:**

```bash
# Check FlexAI authentication
flexai auth status

# Verify repository access
git clone https://github.com/flexaihq/blueprints
```

**Out of Memory Errors:**

- Reduce `rollout_batch_size` from 512 to 256
- Reduce `rollout.n` from 5 to 3 (fewer samples per prompt)
- Enable CPU offloading: `enable_cpu_offload: true` in FSDP config
- Reduce `tensor_parallel_size` for rollout workers

**Reward Function Errors:**

- Verify reward function path is correct in config
- Test reward function locally before training
- Ensure reward function returns float scores for all inputs
- Check for NaN or infinite reward values

**Checkpoint Not Inference Ready:**

- Wait for training to complete fully
- Check `save_model_only: false` in config to include all necessary files
- Verify training completed without errors

**Endpoint Deployment Issues:**

- Verify checkpoint shows `INFERENCE READY = true` status
- Check FlexAI cluster availability
- Review detailed logs with `flexai inference logs <endpoint-name>`

**Dataset Loading Issues:**

- Verify dataset path format: `username/dataset@split`
- Ensure HuggingFace token has access to datasets
- Check prompt_key and answer_key match your dataset schema

**vLLM Rollout Errors:**

- Adjust `gpu_memory_utilization` (default 0.6)
- Reduce `tensor_parallel_size` if GPUs are insufficient
- Enable `enforce_eager: true` for debugging

---

## References

- **EasyR1 GitHub**: <https://github.com/hiyouga/EasyR1>
- **VERL Framework**: <https://github.com/volcengine/verl>
- **FlexAI Documentation**: <https://docs.flex.ai>
- **HybridFlow Paper**: <https://arxiv.org/abs/2409.19256>
- **GRPO Algorithm**: Introduced in DeepSeekMath paper - <https://arxiv.org/abs/2402.03300>
- **GRPO Documentation**: <https://huggingface.co/docs/trl/grpo_trainer>
