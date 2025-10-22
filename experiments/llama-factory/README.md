# Train and Serve a French LLM on FlexAI with LlamaFactory

This experiment demonstrates how to fine-tune a language model on French data using LlamaFactory and deploy it as a production-ready inference endpoint on FlexAI. We'll fine-tune the `Qwen2.5-7B` model on the `openhermes-fr` dataset to create a specialized French language model.s

You will see that this process requires configuring LlamaFactory's dataset registry, setting up training parameters, leveraging FlexAI's managed training infrastructure, and deploying the fine-tuned model as a scalable inference endpoint.

> **Note**: If you haven't already connected FlexAI to GitHub, run `flexai code-registry connect` to set up a code registry connection. This allows FlexAI to pull repositories directly using the repository URL in training commands.

## Step 1: Verify Dataset Configuration

First, ensure your domain-specific dataset is properly configured in LlamaFactory's dataset registry. For our French language example, we'll use the `openhermes-fr` dataset.

Navigate to `experiments/code/llama-factory/data/dataset_info.json` and verify the dataset entry exists:

```json
{
    "openhermes-fr": {
        "hf_hub_url": "legmlai/openhermes-fr",
        "columns": {
            "prompt": "prompt",
            "response": "accepted_completion"
        }
    }
}
```

For your own use case, replace this with your domain-specific dataset. The `openhermes-fr` dataset is specifically designed for French language tasks and serves as an excellent example of domain specialization.

## Step 2: Configure Training Parameters

The [`qwen25-7B_sft.yaml`](../../code/llama-factory/qwen25-7B_sft.yaml) file contains the training configuration for domain-specific fine-tuning. Key settings include:

- **Model**: `Qwen/Qwen2.5-7B` - Excellent multilingual base model suitable for domain adaptation
- **Stage**: `sft` (Supervised Fine-Tuning) - Perfect for task-specific and domain-specific adaptation
- **Dataset**: `openhermes-fr` - Example domain-specific dataset (replace with your own)
- **Training**: Full fine-tuning with DeepSpeed ZeRO Stage 3 for optimal performance

```yaml
---
### model
model_name_or_path: Qwen/Qwen2.5-7B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: code/llama-factory/ds_z3_config.json

### dataset
dataset: openhermes-fr
dataset_dir: code/llama-factory/data
template: qwen
cutoff_len: 2048
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /output-checkpoint
logging_steps: 10
save_steps: 1000
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint:
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

This pre-fetched checkpoint can then be used in your training command to reduce startup time.

## Training

For a 7B model, we recommend using **1 node (8 × H100 GPUs)** to ensure reasonable training time and avoid out-of-memory issues.

### Standard Training (without prefetch)

```bash
flexai training run domain-specific-qwen25-7b-sft \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --env FORCE_TORCHRUN=1 \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/llama-factory/requirements.txt \
  -- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-7B_sft.yaml
```

### Training with Model Prefetch

To take advantage of model pre-fetching performed in the [Optional: Pre-fetch the Model](#optional-pre-fetch-the-model) section, use:

```bash
flexai training run domain-specific-qwen25-7b-sft-prefetched \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint qwen25-7b \
  --env FORCE_TORCHRUN=1 \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/llama-factory/requirements.txt \
  -- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-7B_sft.yaml
```

## Monitoring Training Progress

You can check the status and lifecycle events of your Training Job by running:

```bash
flexai training inspect domain-specific-qwen25-7b-sft
```

Additionally, you can view the logs of your Training Job by running:

```bash
flexai training logs domain-specific-qwen25-7b-sft
```

### Training Observability with TensorBoard

For advanced monitoring and visualization of training metrics, you can leverage TensorBoard integration. FlexAI supports TensorBoard logging for detailed insights into training progress, loss curves, and model performance.

To enable TensorBoard logging, update your YAML configuration:

```yaml
report_to: tensorboard
```

Once enabled, you can access training metrics and visualizations through the FlexAI console. For more details on observability features, see the [FlexAI TensorBoard documentation](https://docs.flex.ai/platform/tensorboard/).

## Getting Training Checkpoints

Once the Training Job completes successfully, you will be able to list all the produced checkpoints:

```bash
flexai training checkpoints domain-specific-qwen25-7b-sft
```

Look for checkpoints marked as `INFERENCE READY = true` - these are ready for serving.

## Serving the Trained Model

Deploy your trained model directly from the checkpoint using FlexAI inference. Replace `<CHECKPOINT_ID>` with the ID from an inference-ready checkpoint:

```bash
flexai -v inference serve domain-specific-endpoint --checkpoint <CHECKPOINT_ID>
```

> **Note**: GPU specification for inference endpoints is currently managed automatically by FlexAI. Future versions will allow explicit GPU count specification for inference workloads to optimize cost and performance based on your specific requirements.

You can monitor your inference endpoint status:

```bash
# List all inference endpoints
flexai inference list

# Get detailed endpoint information
flexai inference inspect domain-specific-endpoint

# Check endpoint logs
flexai inference logs domain-specific-endpoint
```

## Testing Your Domain-Specific Model

Once the endpoint is running, you can test it with domain-specific prompts. For our French language example, the model should demonstrate strong French language understanding, proper grammar and syntax, and cultural context awareness.

### Before and After Training Comparison

To illustrate the improvement from fine-tuning on French data, here's a comparison using the question: **"Qui a gagné la Coupe du monde 2018 ?"** (who won the 2018 world cup?)

**Base Model Response (Qwen/Qwen2.5-7B before training):**

```
La Coupe du monde de football 2018 a été remportée par la Russie.
```

*Issues: Incorrect answer (says Russia instead of France)_

**Fine-tuned Model Response (after training on openhermes-fr):**

```
La France a remporté la Coupe du monde de football 2018, en battant le Croatie lors de la finale disputée à Moscou le 15 juillet 2018.
```

*Improvements: Correct answer (France), excellent French grammar, accurate details, proper structure_

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

### Training Configuration Breakdown

- **DeepSpeed ZeRO Stage 3**: Enables training of 7B model on 1 node efficiently
- **Mixed Precision (bf16)**: Accelerates training while maintaining numerical stability
- **Gradient Accumulation**: Effective batch size of 4 (2 steps × 2 per device)
- **Learning Rate Schedule**: Cosine decay with 10% warmup for stable convergence
- **Context Length**: 2048 tokens, optimized for conversation tasks

### Resource Requirements

**Recommended Configuration for Qwen2.5-7B:**

- **Nodes**: 1 node (cost-effective for 7B models)
- **Accelerators**: 8 × H100 GPUs per node
- **Memory**: ~200GB+ GPU memory total
- **Training Time**: ~2-4 hours for 3 epochs
- **Storage**: ~30GB for checkpoints

**Command Line Parameters Explained:**

- `FORCE_TORCHRUN=1`: Ensures proper distributed training setup

### Scaling Options

- For faster training: Increase to 2 nodes (16 × H100)
- For larger datasets: Adjust `max_samples` parameter
- For longer context: Increase `cutoff_len` (requires more memory)
- For memory efficiency: Switch to `finetuning_type: lora` for QLoRA training

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

- Reduce `per_device_train_batch_size` from 1 to lower value
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Consider using `finetuning_type: lora` for memory efficiency

**Checkpoint Not Inference Ready:**

- Wait for training to complete fully (check with `flexai training inspect`)
- Ensure `save_only_model: false` in YAML configuration
- Verify training completed successfully without errors

**Endpoint Deployment Issues:**

- Verify checkpoint shows `INFERENCE READY = true` status
- Check FlexAI cluster availability with `flexai inference list`
- Review detailed logs with `flexai inference logs <endpoint-name>`

---
