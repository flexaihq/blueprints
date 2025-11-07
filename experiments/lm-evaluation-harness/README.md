# Language Model Evaluation with LM-Evaluation-Harness on FlexAI

This blueprint provides a step-by-step guide for evaluating language models on FlexAI using the LM-Evaluation-Harness framework.

LM-Evaluation-Harness is a unified, extensible toolkit for few-shot evaluation of language models across hundreds of standardized NLP benchmarks.

In this guide, you'll learn how to:
- Run evaluations across multiple NLP tasks and benchmarks
- Compare model performance consistently
- Use FlexAI's managed compute environment for large-scale, reproducible model evaluation

> **Note**: If you haven't already connected FlexAI to GitHub, run `flexai code-registry connect` to set up a code registry connection. This allows FlexAI to pull repositories directly using the repository URL in training commands.

## What is LM-Evaluation-Harness?

LM-Evaluation-Harness is a comprehensive evaluation framework that provides:

- **300+ tasks**: standardized implementations of popular NLP benchmarks
- **Multiple Model Backends**: Support for HuggingFace, OpenAI, Anthropic, and more
- **Reproducible Evaluation**: Consistent evaluation protocols across different models
- **Flexible Configuration**: Easy customization of evaluation parameters
- **Comprehensive Metrics**: Detailed performance metrics and statistical analysis

Popular evaluation tasks include:
- **HellaSwag**: commonsense reasoning
- **MMLU**: multi-task language understanding (57 subjects)
- **GSM8K**: grade school math word problems
- **HumanEval**: code generation capabilities
- **TruthfulQA**: model truthfulness and reliability
- **ARC**: AI2 Reasoning Challenge
- **WinoGrande**: Winograd schema challenge

## Quick Start

Run a basic evaluation on HellaSwag with this single command:

```bash
flexai training run lm-eval-basic \
  --accels 2 \
  --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --requirements-path code/lm-evaluation-harness/requirements.txt \
  --runtime nvidia-25.03 \
  -- lm_eval \
      --model hf \
      --model_args pretrained=EleutherAI/gpt-j-6B \
      --tasks hellaswag \
      --device cuda \
      --batch_size 8
```

## Step 1: Understanding Evaluation Configuration

The LM-Evaluation-Harness uses command-line arguments to configure evaluations. Here are the key parameters:

### Model Configuration
```bash
--model hf                                    # Use HuggingFace backend
--model_args pretrained=MODEL_NAME            # Specify model to evaluate
--model_args pretrained=MODEL_NAME,dtype=bfloat16  # With precision control
```

### Task Selection
```bash
--tasks hellaswag                            # Single task
--tasks hellaswag,arc_easy,arc_challenge     # Multiple tasks
--tasks mmlu_*                               # All MMLU subtasks
--tasks all                                  # All available tasks (not recommended)
```

### Evaluation Parameters
```bash
--batch_size 8                              # Batch size for evaluation
--max_batch_size 32                         # Maximum batch size
--device cuda                               # GPU device
--num_fewshot 5                             # Number of few-shot examples
--limit 1000                                # Limit number of samples per task
```

### Output Configuration
```bash
--output_path /output-checkpoint/results.json          # Save results JSON
--log_samples                               # Log individual sample results
--show_config                               # Display configuration
```

## Create Secrets

To access models from HuggingFace (especially gated models), you need a HuggingFace token.

Use the [`flexai secret create` command](https://docs.flex.ai/cli/commands/secret/) to store your _HuggingFace Token_ as a secret:

```bash
flexai secret create <HF_AUTH_TOKEN_SECRET_NAME>
```

Then paste your _HuggingFace Token_ API key value.

## [Optional] Pre-fetch Models

To speed up evaluation and avoid downloading large models at runtime, you can pre-fetch your models to FlexAI storage:

1. **Create a HuggingFace storage provider:**

    ```bash
    flexai storage create HF-STORAGE --provider huggingface --hf-token-name <HF_AUTH_TOKEN_SECRET_NAME>
    ```

2. **Push the model checkpoint to your storage:**

    ```bash
    flexai checkpoint push llama2-7b --storage-provider HF-STORAGE --source-path meta-llama/Llama-2-7b-hf
    ```

3. **Use the pre-fetched model in your evaluation:**

    ```bash
    flexai training run lm-eval-prefetched \
      --accels 8 \
      --nodes 1 \
      --repository-url https://github.com/flexaihq/blueprints \
      --checkpoint llama2-7b \
      --requirements-path code/lm-evaluation-harness/requirements.txt \
      --runtime nvidia-25.03 \
      -- lm_eval \
          --model hf \
          --model_args pretrained=/input-checkpoint/llama2-7b \
          --tasks mmlu,hellaswag \
          --device cuda \
          --batch_size 4 \
          --output_path /output-checkpoint/prefetched_eval.json
    ```

## Step 2: Common Evaluation Scenarios

### Comprehensive Model Evaluation

For a thorough evaluation across multiple benchmarks:

```bash
flexai training run lm-eval-comprehensive \
  --accels 4 \
  --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/lm-evaluation-harness/requirements.txt \
  --runtime nvidia-25.03 \
  -- lm_eval \
      --model hf \
      --model_args pretrained=microsoft/DialoGPT-medium \
      --tasks hellaswag,arc_easy,arc_challenge,mmlu,gsm8k \
      --device cuda \
      --batch_size 16 \
      --output_path /output-checkpoint/comprehensive_eval.json \
      --log_samples
```

### Large Model Evaluation

For evaluating large models (7B+ parameters):

```bash
flexai training run lm-eval-large-model \
  --accels 8 \
  --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/lm-evaluation-harness/requirements.txt \
  --runtime nvidia-25.03 \
  -- lm_eval \
      --model hf \
      --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype=bfloat16 \
      --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2 \
      --device cuda \
      --batch_size 4 \
      --output_path /output-checkpoint/llama2_7b_eval.json
```

### Code Generation Evaluation

For evaluating code generation capabilities:

```bash
flexai training run lm-eval-code \
  --accels 2 \
  --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/lm-evaluation-harness/requirements.txt \
  --runtime nvidia-25.03 \
  -- lm_eval \
      --model hf \
      --model_args pretrained=Salesforce/codegen-350M-mono \
      --tasks humaneval \
      --device cuda \
      --batch_size 8 \
      --output_path /output-checkpoint/code_eval.json
```

### Few-Shot Learning Evaluation

For testing few-shot learning capabilities:

```bash
flexai training run lm-eval-fewshot \
  --accels 2 \
  --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/lm-evaluation-harness/requirements.txt \
  --runtime nvidia-25.03 \
  -- lm_eval \
      --model hf \
      --model_args pretrained=EleutherAI/gpt-neo-1.3B \
      --tasks winogrande,piqa,openbookqa \
      --num_fewshot 5 \
      --device cuda \
      --batch_size 16 \
      --output_path /output-checkpoint/fewshot_eval.json
```

## Monitoring Evaluation Progress

You can check the status and progress of your evaluation job:

```bash
# Check job status
flexai training inspect lm-eval-comprehensive

# View evaluation logs
flexai training logs lm-eval-comprehensive

```

## Getting Evaluation Results

Once the evaluation job completes, you can access the results:

```bash
# List all checkpoints/outputs
flexai training checkpoints lm-eval-comprehensive

# Download results JSON
flexai checkpoint fetch <CHECKPOINT_ID> --destination ./results/
```

The results JSON will be saved with detailed metrics for each task.

## Understanding Evaluation Results

### Sample Results Structure

```json
{
  "results": {
    "hellaswag": {
      "acc": 0.6234,
      "acc_stderr": 0.0048,
      "acc_norm": 0.8012,
      "acc_norm_stderr": 0.0040
    },
    "mmlu": {
      "acc": 0.4567,
      "acc_stderr": 0.0031
    }
  },
  "config": {
    "model": "hf",
    "model_args": "pretrained=EleutherAI/gpt-j-6B",
    "batch_size": 8,
    "device": "cuda"
  },
  "git_hash": "abc123",
  "date": 1698123456
}
```

### Key Metrics Explained

- **acc**: Raw accuracy score
- **acc_stderr**: Standard error of accuracy
- **acc_norm**: Length-normalized accuracy (for some tasks)
- **pass@k**: Pass rate for code generation tasks
- **bleu**: BLEU score for generation tasks
- **rouge**: ROUGE scores for summarization tasks

## Advanced Evaluation Scenarios

### Multi-Node Evaluation

For very large models or extensive benchmark suites:

```bash
flexai training run lm-eval-multi-node \
  --accels 8 \
  --nodes 2 \
  --repository-url https://github.com/flexaihq/blueprints \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/lm-evaluation-harness/requirements.txt \
  --runtime nvidia-25.03 \
  -- lm_eval \
      --model hf \
      --model_args pretrained=meta-llama/Llama-2-13b-hf,dtype=bfloat16,device_map=auto \
      --tasks mmlu_*,hellaswag,arc_challenge,truthfulqa_mc2 \
      --device cuda \
      --batch_size 2 \
      --output_path /output-checkpoint/multi_node_eval.json
```

### Custom Task Evaluation

For evaluating on custom tasks or datasets:

```bash
flexai training run lm-eval-custom \
  --accels 4 \
  --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/lm-evaluation-harness/requirements.txt \
  --runtime nvidia-25.03 \
  -- lm_eval \
      --model hf \
      --model_args pretrained=microsoft/DialoGPT-medium \
      --tasks_list /path/to/custom_tasks.yaml \
      --device cuda \
      --batch_size 8 \
      --output_path /output-checkpoint/custom_eval.json
```

### Evaluation with Reduced Precision

For memory-efficient evaluation:

```bash
flexai training run lm-eval-fp16 \
  --accels 4 \
  --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/lm-evaluation-harness/requirements.txt \
  --runtime nvidia-25.03 \
  -- lm_eval \
      --model hf \
      --model_args pretrained=EleutherAI/gpt-neox-20b,dtype=float16 \
      --tasks hellaswag,mmlu \
      --device cuda \
      --batch_size 1 \
      --output_path /output-checkpoint/fp16_eval.json
```

## Expected Results and Benchmarks

### Typical Benchmark Performance

**HellaSwag (Commonsense Reasoning):**
- Random baseline: ~25%
- GPT-3 (175B): ~78.9%
- Human performance: ~95.6%

**MMLU (Multi-task Language Understanding):**
- Random baseline: ~25%
- GPT-3 (175B): ~43.9%
- Human expert performance: ~89.8%

**GSM8K (Grade School Math):**
- Random baseline: ~0%
- GPT-3 (175B): ~17.9%
- Human performance: ~85%

**HumanEval (Code Generation):**
- Random baseline: ~0%
- GPT-3 (175B): ~14.2%
- Human performance: ~90%

### Model Size vs Performance

Generally, larger models perform better, but with diminishing returns:

- **Small models (< 1B)**: Basic language understanding
- **Medium models (1B-7B)**: Reasonable performance on most tasks
- **Large models (7B-70B)**: Strong performance across benchmarks
- **Very large models (> 70B)**: State-of-the-art performance

## Technical Details

### Resource Requirements

**Recommended Configurations:**

**Small Models (< 1B parameters):**
- **Accelerators**: 1-2 GPUs
- **Memory**: 8-16GB GPU memory
- **Evaluation Time**: 30 minutes - 2 hours

**Medium Models (1B-7B parameters):**
- **Accelerators**: 2-4 GPUs
- **Memory**: 16-32GB GPU memory
- **Evaluation Time**: 1-6 hours

**Large Models (7B-70B parameters):**
- **Accelerators**: 4-8 GPUs
- **Memory**: 40-80GB GPU memory
- **Evaluation Time**: 2-12 hours

**Very Large Models (> 70B parameters):**
- **Accelerators**: 8+ GPUs (multi-node recommended)
- **Memory**: 80GB+ GPU memory
- **Evaluation Time**: 6-24 hours

### Optimization Tips

**Memory Optimization:**
- Use `dtype=bfloat16` or `dtype=float16` for reduced memory usage
- Reduce `batch_size` for large models
- Use `device_map=auto` for automatic device placement

**Speed Optimization:**
- Increase `batch_size` when memory allows
- Use `max_batch_size` for adaptive batching
- Set `limit` for quick testing with subset of data
- Use multiple GPUs with `--device cuda`

**Accuracy Optimization:**
- Use appropriate `num_fewshot` for few-shot tasks
- Enable `--log_samples` for detailed analysis
- Run multiple times with different seeds for statistical significance

### Command Line Parameters Explained

- `--model hf`: Use HuggingFace Transformers backend
- `--model_args`: Model-specific arguments (path, dtype, etc.)
- `--tasks`: Comma-separated list of evaluation tasks
- `--device`: Device placement (cuda, cpu)
- `--batch_size`: Batch size for evaluation
- `--num_fewshot`: Number of examples for few-shot evaluation
- `--output_path`: Path to save results JSON
- `--log_samples`: Save individual sample predictions
- `--limit`: Limit number of samples per task (for testing)

## Popular Task Collections

### Core Benchmarks
```bash
--tasks hellaswag,arc_easy,arc_challenge,winogrande,piqa
```

### Academic Benchmarks
```bash
--tasks mmlu,truthfulqa_mc2,gsm8k,humaneval
```

### Reasoning Tasks
```bash
--tasks arc_challenge,hellaswag,winogrande,piqa,openbookqa
```

### Language Understanding
```bash
--tasks mmlu_*,truthfulqa_mc2,lambada_openai
```

### Code Generation
```bash
--tasks humaneval,mbpp
```

### Math and Logic
```bash
--tasks gsm8k,mathqa,aqua_rat
```

## Troubleshooting

**Common Issues:**

**Out of Memory Errors:**
- Reduce `batch_size` (try 1, 2, 4)
- Use mixed precision: `dtype=bfloat16` or `dtype=float16`
- Increase number of GPUs: `--accels 8`
- Enable CPU offloading: `device_map=auto`

**Model Loading Errors:**
- Verify model name is correct on HuggingFace
- Check that your HuggingFace token has permission to access gated models
- Use `trust_remote_code=True` for custom models
- Verify there's sufficient disk space for model downloads

**Task Not Found Errors:**
- List available tasks: `lm_eval --tasks list`
- Check task name spelling and capitalization
- Verify task is supported in your lm-eval version
- Use task groups: `mmlu_*` instead of individual tasks

**Slow Evaluation:**
- Increase `batch_size` when memory allows
- Use `--limit` for quick testing (e.g., `--limit 100`)
- Consider fewer tasks for initial testing
- Use faster/smaller models for development

**Authentication Issues:**
- Create HuggingFace token with appropriate permissions
- Store token as FlexAI secret correctly
- Verify token is not expired
- Check model access permissions (especially for gated models)

**Job Monitoring Commands:**
```bash
# Check job status
flexai training inspect <job-name>

# View logs
flexai training logs <job-name>

```

**Results Access:**
```bash
# List outputs
flexai training checkpoints <job-name>

# Download results JSON
flexai checkpoint fetch <checkpoint-id> --destination ./results/
```

---

## References

- **LM-Evaluation-Harness GitHub**: https://github.com/EleutherAI/lm-evaluation-harness
- **Documentation**: https://github.com/EleutherAI/lm-evaluation-harness/tree/master/docs
- **Paper**: "Language Model Evaluation Harness" - https://arxiv.org/abs/2101.00027
- **Task List**: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md
- **FlexAI Documentation**: https://docs.flex.ai
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
