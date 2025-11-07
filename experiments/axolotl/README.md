# Fine-tune with Axolotl on Domain Specific Data and Deploy Endpoint on FlexAI

This blueprint demonstrates how to use FlexAI to fine-tune language models on domain-specific data using Axolotl, then deploy them as production-ready inference endpoints. For illustration purposes, we'll fine-tune for maximum command of French using the `Qwen2.5-7B` model and the `openhermes-fr` dataset.

You will see that this process requires configuring Axolotl's training parameters, leveraging FlexAI's managed training infrastructure, and deploying the fine-tuned model as a scalable inference endpoint.

> **Note**: If you haven't already connected FlexAI to GitHub, run `flexai code-registry connect` to set up a code registry connection. This allows FlexAI to pull repositories directly using the repository URL in training commands.

## Step 1: Verify Dataset Configuration

First, ensure your domain-specific dataset is properly configured in your Axolotl YAML file. For our French language example, we'll use the `openhermes-fr` dataset.

Navigate to `code/axolotl/qwen2/fft-7b-french.yaml` and verify the dataset configuration:

```yaml
datasets:
- path: legmlai/openhermes-fr
  type:
    system_prompt: ""
    field_instruction: prompt
    field_output: accepted_completion
    format: "{instruction}"
    no_input_format: "{instruction}"
```

For your own use case, replace this with your domain-specific dataset. The `openhermes-fr` dataset is specifically designed for French language tasks and serves as an excellent example of domain specialization.

## Step 2: Configure Training Parameters

The [`qwen2/fft-7b-french.yaml`](../../code/axolotl/qwen2/fft-7b-french.yaml) file contains the training configuration for domain-specific fine-tuning. Key settings include:

- **Model**: `Qwen/Qwen2.5-7B` - Excellent multilingual base model suitable for domain adaptation
- **Stage**: Full fine-tuning - Perfect for task-specific and domain-specific adaptation
- **Dataset**: `openhermes-fr` - Example domain-specific dataset (replace with your own)
- **Training**: Full fine-tuning with FSDP (Fully Sharded Data Parallel) for optimal performance

```yaml
base_model: Qwen/Qwen2.5-7B

load_in_8bit: false
load_in_4bit: false
strict: false

datasets:
- path: legmlai/openhermes-fr
  type:
    system_prompt: ""
    field_instruction: prompt
    field_output: accepted_completion
    format: "{instruction}"
    no_input_format: "{instruction}"
dataset_prepared_path:
val_set_size: 0.05
output_dir: /output-checkpoint

sequence_len: 2048
sample_packing: true
eval_sample_packing: true

gradient_accumulation_steps: 2
micro_batch_size: 1
num_epochs: 3
optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 1.0e-5

bf16: auto
tf32: true

gradient_checkpointing: true
flash_attention: true

warmup_ratio: 0.1
evals_per_epoch: 4
saves_per_epoch: 1
weight_decay: 0.0
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_limit_all_gathers: true
  fsdp_sync_module_states: true
  fsdp_offload_params: true
  fsdp_use_orig_params: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sharding_strategy: FULL_SHARD
```

## Create Secrets

To access the Qwen2.5-7B model and OpenHermes-FR dataset, you need a HuggingFace token.

Use the [`flexai secret create` command](https://docs.flex.ai/cli/commands/secret/) to store your _HuggingFace Token_ as a secret. Replace `<HF_AUTH_TOKEN_SECRET_NAME>` with your desired name for the secret:

```bash
flexai secret create <HF_AUTH_TOKEN_SECRET_NAME>
```

Then paste your _HuggingFace Token_ API key value.

## [Optional] Pre-fetch the Model

To speed up training and avoid downloading large models at runtime, you can pre-fetch your HuggingFace model to FlexAI storage. For example, to pre-fetch the `Qwen/Qwen2.5-7B` model:

1. **Create a HuggingFace storage provider:**

    ```bash
    flexai storage create HF-STORAGE --provider huggingface --hf-token-name <HF_AUTH_TOKEN_SECRET_NAME>
    ```

2. **Push the model checkpoint to your storage:**

    ```bash
    flexai checkpoint push qwen25-7b --storage-provider HF-STORAGE --source-path Qwen/Qwen2.5-7B
    ```

During your training run, you can use the pre-fetched model by adding the following argument to your training command:

```bash
--checkpoint qwen25-7b
```

## Training

For a 7B model, we recommend using **1 node (4 × H100 GPUs)** to ensure reasonable training time and avoid out-of-memory issues.

### Standard Training (without prefetch)

```bash
flexai training run axolotl-french-sft \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --env FORCE_TORCHRUN=1 \
  --requirements-path code/axolotl/requirements.txt \
  -- axolotl train code/axolotl/qwen2/fft-7b-french.yaml
```

### Training with Model Prefetch

To take advantage of model pre-fetching performed in the [Optional: Pre-fetch the Model](#optional-pre-fetch-the-model) section, use:

```bash
flexai training run axolotl-french-sft-prefetched \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint qwen25-7b \
  --env FORCE_TORCHRUN=1 \
  --requirements-path code/axolotl/requirements.txt \
  -- axolotl train code/axolotl/qwen2/fft-7b-french.yaml
```

## Monitoring Training Progress

You can check the status and lifecycle events of your Training Job by running:

```bash
flexai training inspect axolotl-french-sft
```

Additionally, you can view the logs of your Training Job by running:

```bash
flexai training logs axolotl-french-sft
```

### Training Observability with Weights & Biases

For advanced monitoring and visualization of training metrics, Axolotl supports Weights & Biases (wandb) integration. You can leverage wandb logging for detailed insights into training progress, loss curves, and model performance.

To enable wandb logging, update your YAML configuration:
```yaml
wandb_project: your-project-name
wandb_entity: your-wandb-entity
wandb_watch: gradients
wandb_name: qwen25-7b-sft
wandb_log_model: checkpoint
```

For additional monitoring, you can also access FlexAI's hosted [TensorBoard](https://tensorboard.flex.ai/) instance, which provides organization-wide real-time insights into your training workload progress. TensorBoard is enabled by default for every organization - simply log in using your FlexAI account credentials to track metrics and visualize performance.

For more details on observability features, see the [Axolotl documentation](https://github.com/OpenAccess-AI-Collective/axolotl#logging).

## Getting Training Checkpoints

Once the Training Job completes successfully, you will be able to list all the produced checkpoints:

```bash
flexai training checkpoints axolotl-french-sft
```

Look for checkpoints marked as `INFERENCE READY = true` - these are ready for serving.

## Serving the Trained Model

Deploy your trained model directly from the checkpoint using FlexAI inference. Replace `<CHECKPOINT_ID>` with the ID from an inference-ready checkpoint:

```bash
flexai inference serve axolotl-french-sft-endpoint --checkpoint <CHECKPOINT_ID>
```

> **Note**: GPU specification for inference endpoints is currently managed automatically by FlexAI. Future versions will allow explicit GPU count specification for inference workloads to optimize cost and performance based on your specific requirements.

You can monitor your inference endpoint status:

```bash
# List all inference endpoints
flexai inference list

# Get detailed endpoint information
flexai inference inspect axolotl-french-sft-endpoint

# Check endpoint logs
flexai inference logs axolotl-french-sft-endpoint
```

## Testing Your Domain-Specific Model

Once the endpoint is running, you can test it with domain-specific prompts. For our French language example, the model should demonstrate strong French language understanding, proper grammar and syntax, and cultural context awareness.

### Before and After Training Comparison

To illustrate the improvement from fine-tuning on French data, here's a comparison using the question: **"Qui a gagné la Coupe du monde 2018 ?"** (who won the 2018 world cup?)

**Base Model Response (Qwen/Qwen2.5-7B before training):**
```
La Coupe du monde de football 2018 a été remportée par la Russie.
```
*Issues: Incorrect answer (says Russia instead of France)*

**Fine-tuned Model Response (after full fine-tuning on openhermes-fr):**
```
La France a remporté la Coupe du monde de football 2018, en battant le Croatie lors de la finale disputée à Moscou le 15 juillet 2018.
```
*Improvements: Correct answer (France), excellent French grammar, accurate details, proper structure*

This example demonstrates the dramatic improvement in both factual accuracy and French language quality after domain-specific fine-tuning.

### Example API Call

```bash
curl -X POST "https://your-endpoint-url/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "Expliquez-moi les avantages de l'\''intelligence artificielle en français:",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

Adapt the prompt and evaluation criteria to match your specific domain and use case.

## Expected Results

After fine-tuning on domain-specific data, your model should achieve:

- **Domain Expertise**: Specialized knowledge and terminology understanding for your target domain
- **Task-Specific Performance**: Enhanced capabilities for domain-relevant tasks and workflows
- **Maintained General Capabilities**: Preserved reasoning, problem-solving, and general language skills

For our French language example:
- **Strong French Language Understanding**: Natural conversation flow, proper grammar, cultural context
- **High Performance on French Tasks**: Question answering, text summarization, creative writing

## Technical Details

### Training Configuration Breakdown:
- **Full Fine-tuning with FSDP**: Enables training of 7B model on 1 node efficiently
- **Mixed Precision (bf16)**: Accelerates training while maintaining numerical stability
- **Gradient Accumulation**: Effective batch size of 2 (2 steps × 1 per device)
- **Learning Rate Schedule**: Cosine decay with 10% warmup for stable convergence
- **Context Length**: 2048 tokens, optimized for conversation tasks
- **Sample Packing**: Efficient batch utilization for variable-length sequences
- **Flash Attention**: Optimized attention mechanism for faster training

### Resource Requirements

**Recommended Configuration for Qwen2.5-7B:**
- **Nodes**: 1 node (cost-effective for 7B models)
- **Accelerators**: 4 × H100 GPUs per node
- **Memory**: ~200GB+ GPU memory total
- **Training Time**: ~2-4 hours for 3 epochs
- **Storage**: ~30GB for checkpoints

**Command Line Parameters Explained:**
- `FORCE_TORCHRUN=1`: Ensures proper distributed training setup
- `--runtime nvidia-25.06`: Specifies NVIDIA runtime version for optimal performance

### Scaling Options:
- For faster training: Increase to 2 nodes (16 × H100)
- For larger datasets: Adjust `num_epochs` parameter
- For longer context: Increase `sequence_len` (requires more memory)
- For memory efficiency: Switch to QLoRA with `load_in_4bit: true` and `adapter: qlora`
- For other models: Use configs in `code/axolotl/` directory (Llama, Mistral, Gemma, Phi, etc.)

## Additional Training Examples

Axolotl provides extensive configuration examples for various models and training strategies:

### Llama 3.1 8B with LoRA

```bash
flexai training run axolotl-lora-llama3-8B \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/experiments \
  --env FORCE_TORCHRUN=1 \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/axolotl/requirements.txt \
  --runtime nvidia-25.06 \
  -- axolotl train code/axolotl/llama-3/lora-8b.yml
```

### Mistral 7B with QLoRA

```bash
flexai training run axolotl-qlora-mistral-7B \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/experiments \
  --env FORCE_TORCHRUN=1 \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/axolotl/requirements.txt \
  --runtime nvidia-25.06 \
  -- axolotl train code/axolotl/mistral/qlora.yml
```

Explore the `code/axolotl/` directory for more examples including Gemma, Phi, Qwen2, multimodal models, and advanced configurations.

## Troubleshooting

**Common Issues:**

**Training Job Fails to Start:**
```bash
# Check FlexAI authentication
flexai auth status

# Verify repository access
git clone https://github.com/flexaihq/experiments
```

**Out of Memory Errors:**
- Reduce `micro_batch_size` from 1 to lower value (not recommended below 1)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Consider switching to QLoRA: set `load_in_4bit: true` and `adapter: qlora` for memory efficiency
- Enable `fsdp_offload_params: true` for additional memory savings

**Checkpoint Not Inference Ready:**
- Wait for training to complete fully (check with `flexai training inspect`)
- Ensure Axolotl configuration saves model in compatible format
- Verify training completed successfully without errors

**Endpoint Deployment Issues:**
- Verify checkpoint shows `INFERENCE READY = true` status
- Check FlexAI cluster availability with `flexai inference list`
- Review detailed logs with `flexai inference logs <endpoint-name>`

**Dataset Loading Issues:**
- Verify dataset path is correct in YAML configuration (e.g., `legmlai/openhermes-fr`)
- Ensure HuggingFace token has access to private datasets
- Check dataset format matches the specified type (sharegpt, alpaca, etc.)

---
