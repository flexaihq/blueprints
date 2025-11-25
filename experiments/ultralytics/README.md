# Object Detection and Computer Vision with Ultralytics YOLO11 on FlexAI

This blueprint demonstrates how to train, validate, and deploy computer vision models using Ultralytics YOLO11 on FlexAI. You'll learn how to fine-tune YOLO models for object detection, instance segmentation, and pose estimation tasks, then deploy them as production-ready inference endpoints.

YOLO (You Only Look Once) is one of the most popular real-time object detection frameworks, and YOLO11 brings significant improvements in accuracy and efficiency across multiple computer vision tasks.

> **Note**: If FlexAI is not yet connected to your GitHub account, run:
> ```bash
> flexai code-registry connect
> ```
> This enables FlexAI to automatically pull code from repositories referenced in `--repository-url`.

## Overview

This blueprint covers:

- **Object Detection Training**: Fine-tune YOLO11 on custom datasets for object detection
- **Instance Segmentation**: Train models to detect and segment objects at the pixel level
- **Pose Estimation**: Train models for human pose detection and keypoint tracking
- **Model Validation**: Evaluate model performance with comprehensive metrics
- **Model Export**: Export to optimized formats (ONNX, TensorRT) for deployment
- **Inference Deployment**: Deploy trained models as FlexAI inference endpoints

All YOLO tasks use the same CLI interface with different task modes (`detect`, `segment`, `pose`, `track`, etc.). This guide demonstrates the core workflows that apply across all computer vision tasks.

## Step 1: Prepare the Dataset

YOLO models require datasets in a specific format. We'll use the COCO8 dataset (a small subset of COCO) for this example, but you can easily adapt this to your own custom dataset.

### Option A: Use COCO8 Dataset (Quick Start)

The COCO8 dataset will be automatically downloaded by Ultralytics during training. No manual download is required.

### Option B: Prepare Your Own Custom Dataset

For custom datasets, follow the YOLO format structure:

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

Each label file contains annotations in YOLO format (one object per line):

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates must be normalized to the range [0, 1].

Create a data configuration file (`data.yaml`):

```yaml
path: /input
train: /input/images/train
val: /input/images/val

names:
  0: person
  1: bicycle
  2: car
  # ... add your classes
```

> **Note**: Both mapping and list formats are supported for class names. You can also use: `names: ["person", "bicycle", "car"]`

> ⚠️ **Important**: Use spaces (not tabs) for indentation in `data.yaml`. Tab characters will cause silent parsing failures.

YOLO accepts both relative and absolute paths, but using absolute paths (`/input/...`) reduces ambiguity inside FlexAI jobs.

If your dataset uses a different annotation format (COCO JSON, Pascal VOC, etc.), convert it to YOLO format before uploading. Refer to the [Ultralytics Data Format documentation](https://docs.ultralytics.com/datasets/detect/) for conversion guidance.

### Upload Custom Dataset to FlexAI

Once your dataset is prepared, upload it to FlexAI:

```bash
flexai dataset push yolo-custom-dataset --file path/to/your-dataset
```

When you use the dataset in a training job with `--dataset yolo-custom-dataset`, FlexAI will mount the dataset contents directly at `/input/` in your training environment. **All dataset contents are mounted under `/input`, preserving their original folder structure.**

This means:
- If your dataset structure is `dataset/images/train/...`, it will be accessible at `/input/images/train/...`
- Your `data.yaml` should be at `/input/data.yaml`
- The `path` field in your `data.yaml` should be `/input` as shown in the example above

## Step 2: Train an Object Detection Model

Train a YOLO11 model for object detection. We'll start with the nano model (YOLO11n) which is fast and efficient.

### Training on COCO8 Dataset

```bash
flexai training run yolo11-detection-coco8 \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect train \
    data=coco8.yaml \
    model=yolo11n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0,1,2,3 \
    project=/output-checkpoint \
    name=yolo11n-coco8 \
    patience=50 \
    save=True \
    val=True
```

> **Note**: `--accels` specifies the number of GPUs to allocate (e.g., `--accels 4` = 4 GPUs of the chosen accelerator type).

### Training on Custom Dataset

```bash
flexai training run yolo11-detection-custom \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --dataset yolo-custom-dataset \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect train \
    data=/input/data.yaml \
    model=yolo11n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0,1,2,3 \
    project=/output-checkpoint \
    name=yolo11n-custom \
    save=True
```


### Training with Larger Models

For better accuracy, use larger YOLO variants. Adjust batch size based on model size:

**YOLO11s (Small)**:

```bash
flexai training run yolo11s-detection \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect train \
    data=coco8.yaml \
    model=yolo11s.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0,1,2,3 \
    project=/output-checkpoint
```

**YOLO11m (Medium)**:

```bash
flexai training run yolo11m-detection \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect train \
    data=coco8.yaml \
    model=yolo11m.pt \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    device=0,1,2,3 \
    project=/output-checkpoint
```

**YOLO11l (Large)**:

```bash
flexai training run yolo11l-detection \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect train \
    data=coco8.yaml \
    model=yolo11l.pt \
    epochs=100 \
    imgsz=640 \
    batch=4 \
    device=0,1,2,3,4,5,6,7 \
    project=/output-checkpoint
```

## Step 3: Train an Instance Segmentation Model

Instance segmentation detects objects and generates pixel-level masks for each instance.

```bash
flexai training run yolo11-segmentation \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo segment train \
    data=coco8-seg.yaml \
    model=yolo11n-seg.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0,1,2,3 \
    project=/output-checkpoint \
    name=yolo11n-segmentation \
    save=True
```

For custom segmentation datasets, ensure your labels include polygon annotations in YOLO segmentation format.

## Step 4: Train a Pose Estimation Model

For human pose estimation tasks with keypoint detection:

```bash
flexai training run yolo11-pose \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo pose train \
    data=coco8-pose.yaml \
    model=yolo11n-pose.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0,1,2,3 \
    project=/output-checkpoint
```

## Monitoring Training Progress

### Check Training Status

```bash
flexai training inspect yolo11-detection-coco8
```

### View Training Logs

```bash
flexai training logs yolo11-detection-coco8
```

### Training Observability with TensorBoard

Ultralytics automatically logs training metrics. Access FlexAI's hosted [TensorBoard](https://tensorboard.flex.ai/) instance to track:

- Training and validation loss curves
- mAP (mean Average Precision) metrics
- Precision and recall curves
- Learning rate schedules

### Weights & Biases Integration

For advanced monitoring, integrate with Weights & Biases by adding environment variables:

```bash
flexai training run yolo11-detection-wandb \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
  --env WANDB_PROJECT=yolo11-experiments \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect train \
    data=coco8.yaml \
    model=yolo11n.pt \
    epochs=100 \
    project=/output-checkpoint
```

## Step 5: Validate the Model

After training, validate your model's performance on the validation dataset.

### List Training Checkpoints

```bash
flexai training checkpoints yolo11-detection-coco8
```

### Run Validation

For validation with the COCO8 dataset (will be auto-downloaded):

```bash
flexai training run yolo11-validation \
  --accels 1 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint <CHECKPOINT_ID> \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect val \
    model=/checkpoint/weights/best.pt \
    data=coco8.yaml \
    imgsz=640 \
    batch=16 \
    project=/output-checkpoint/validation
```

For validation with a custom dataset:

```bash
flexai training run yolo11-validation-custom \
  --accels 1 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint <CHECKPOINT_ID> \
  --dataset yolo-custom-dataset \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect val \
    model=/checkpoint/weights/best.pt \
    data=/input/data.yaml \
    imgsz=640 \
    batch=16 \
    project=/output-checkpoint/validation
```

### Understanding Validation Metrics

YOLO provides comprehensive metrics:

- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: Mean Average Precision averaged across IoU thresholds 0.5-0.95
- **Precision**: Ratio of true positive detections
- **Recall**: Ratio of detected ground truth objects
- **F1-Score**: Harmonic mean of precision and recall

## Step 6: Export the Model

Export your trained model to various formats for optimized deployment.

### Download Checkpoint Locally

First, download the best checkpoint to your local machine:

```bash
flexai checkpoint fetch "<CHECKPOINT_ID>" --destination ./yolo11-checkpoint
```

### Export to ONNX

ONNX format is widely supported and optimized for cross-platform inference:

```bash
flexai training run yolo11-export-onnx \
  --accels 1 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint <CHECKPOINT_ID> \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo export \
    model=/checkpoint/weights/best.pt \
    format=onnx \
    imgsz=640 \
    simplify=True \
    dynamic=False
```

### Export to TensorRT

For NVIDIA GPU deployment with maximum performance:

```bash
flexai training run yolo11-export-tensorrt \
  --accels 1 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint <CHECKPOINT_ID> \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo export \
    model=/checkpoint/weights/best.pt \
    format=engine \
    imgsz=640 \
    half=True \
    device=0
```

### Available Export Formats

| Format | format Argument | Use Case |
|--------|----------------|----------|
| PyTorch | `torchscript` | General PyTorch deployment |
| ONNX | `onnx` | Cross-platform inference |
| TensorRT | `engine` | NVIDIA GPU optimization |
| CoreML | `coreml` | Apple devices (iOS/macOS) |
| TFLite | `tflite` | Mobile and embedded devices |
| OpenVINO | `openvino` | Intel hardware acceleration |
| NCNN | `ncnn` | Mobile deployment |

See the [Ultralytics Export documentation](https://docs.ultralytics.com/modes/export/) for more formats and options.

## Step 7: Run Inference on Your Trained Model

After training, you can run inference on images or videos using your trained model.

### Run Inference as a Training Job

FlexAI's managed inference endpoints currently support vLLM only. For YOLO models, use a training job to execute predictions directly:

```bash
flexai training run yolo11-predict \
  --accels 1 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint <CHECKPOINT_ID> \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect predict \
    model=/checkpoint/weights/best.pt \
    source=https://ultralytics.com/images/bus.jpg \
    conf=0.25 \
    iou=0.45 \
    save=True \
    project=/output-checkpoint/predictions
```

The predictions and annotated images will be saved in the job's output directory.

### Run Batch Predictions

To run inference on a directory of images, first upload them as a dataset:

```bash
flexai dataset push test-images --file path/to/images/
```

Then run prediction:

```bash
flexai training run yolo11-predict-batch \
  --accels 1 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint <CHECKPOINT_ID> \
  --dataset test-images \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect predict \
    model=/checkpoint/weights/best.pt \
    source=/input \
    conf=0.25 \
    save=True \
    project=/output-checkpoint/predictions
```

### Download Prediction Results

After the prediction job completes, download the results:

```bash
flexai training checkpoints yolo11-predict

flexai checkpoint fetch "<PREDICTION_CHECKPOINT_ID>" --destination ./predictions
```

### Quick Local Testing

> **💡 Tip**: Test your trained model locally before deploying to production.

You can test your model locally after downloading the checkpoint:

```bash
# Download the checkpoint
flexai checkpoint fetch "<CHECKPOINT_ID>" --destination ./yolo11-checkpoint

# Run inference locally
yolo detect predict \
  model=./yolo11-checkpoint/weights/best.pt \
  source=path/to/image.jpg \
  conf=0.25 \
  save=True
```

Results will be saved to `runs/detect/predict/`.

## Advanced Use Cases

### Object Tracking

YOLO11 supports multi-object tracking in videos. First upload your video as a dataset:

```bash
flexai dataset push test-video --file path/to/video.mp4
```

Then run tracking:

```bash
flexai training run yolo11-tracking \
  --accels 1 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint <CHECKPOINT_ID> \
  --dataset test-video \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo track \
    model=/checkpoint/weights/best.pt \
    source=/input/video.mp4 \
    conf=0.25 \
    iou=0.45 \
    save=True \
    project=/output-checkpoint/tracking
```

### Model Benchmarking

Compare model performance across different formats and hardware:

```bash
flexai training run yolo11-benchmark \
  --accels 1 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint <CHECKPOINT_ID> \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo benchmark \
    model=/checkpoint/weights/best.pt \
    data=coco8.yaml \
    imgsz=640
```

This will benchmark:
- PyTorch inference speed
- ONNX inference speed
- TensorRT inference speed (if available)
- Model accuracy metrics

### Hyperparameter Tuning

Use Ultralytics' built-in hyperparameter tuning:

```bash
flexai training run yolo11-tune \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect tune \
    data=coco8.yaml \
    model=yolo11n.pt \
    epochs=30 \
    iterations=300 \
    optimizer=AdamW
```

## Expected Results

### Detection Performance (YOLO11n on COCO8)

- **mAP50**: typically 0.45–0.55
- **mAP50-95**: typically 0.30–0.40
- **Inference Speed**: 1.5–2.5 ms/image (H100 TensorRT)
- **Model Size**: ~3 MB

### Segmentation Performance (YOLO11n-seg on COCO8-seg)

- **Box mAP50**: typically 0.45–0.55
- **Mask mAP50**: typically 0.40–0.50
- **Inference Speed**: 2–3 ms/image (H100 TensorRT)

### Training Time

- **YOLO11n on COCO8**: ~5–10 minutes (4 × H100, 100 epochs).
- **YOLO11s on COCO8**: ~10–15 minutes (4 × H100, 100 epochs).
- **YOLO11m on full COCO**: ~4–6 hours (4 × H100, 100 epochs).

## Technical Details

### Recommended Resource Configuration

| Model | GPUs | Batch Size | Memory | Training Time (100 epochs, COCO8) |
|-------|------|------------|--------|-----------------------------------|
| YOLO11n | 1-4 × H100 | 16-32 | 8GB+ | 5-10 min |
| YOLO11s | 2-4 × H100 | 8-16 | 12GB+ | 10-15 min |
| YOLO11m | 4 × H100 | 4-8 | 16GB+ | 15-20 min |
| YOLO11l | 4-8 × H100 | 2-4 | 24GB+ | 20-30 min |

### Key Training Parameters

**Image Size (`imgsz`)**:
- Standard: 640×640
- Small objects: 1280×1280 (slower but better detection)
- Real-time applications: 320×320 or 416×416 (faster inference)

**Batch Size (`batch`)**:
- Larger batches generally lead to better convergence
- Adjust based on GPU memory: YOLO11n (16-32), YOLO11s (8-16), YOLO11m (4-8)

**Epochs**:
- Small datasets (< 1000 images): 100-150 epochs
- Medium datasets (1000-10000): 50-100 epochs
- Large datasets (> 10000): 30-50 epochs

**Early Stopping (`patience`)**:
- Stops training if no improvement for N epochs
- Recommended: 50 epochs for COCO8, 30 epochs for larger datasets

**Data Augmentation**:
- Enabled by default with optimized settings
- Includes mosaic, mixup, HSV augmentation, and geometric transforms

### Multi-GPU Training

FlexAI automatically enables distributed training when multiple GPUs are requested:

```bash
flexai training run yolo11-multi-gpu \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect train \
    data=coco8.yaml \
    model=yolo11n.pt \
    epochs=100 \
    batch=32 \
    device=0,1,2,3,4,5,6,7 \
    project=/output-checkpoint
```

### Transfer Learning

Continue training from a previous checkpoint:

```bash
flexai training run yolo11-transfer-learning \
  --accels 4 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint <PREVIOUS_CHECKPOINT_ID> \
  --requirements-path code/ultralytics/requirements.txt \
  -- yolo detect train \
    data=coco8.yaml \
    model=/checkpoint/weights/best.pt \
    epochs=50 \
    batch=16 \
    project=/output-checkpoint
```

## Troubleshooting

### Common Issues

**Fixing Training Job Failures:**

```bash
# Check FlexAI authentication
flexai auth status

# Verify dataset upload
flexai dataset list
```

**Fixing Out-of-Memory Errors:**

- Reduce batch size: `batch=8` or `batch=4`
- Use smaller image size: `imgsz=416`
- Try a smaller model variant: YOLO11n instead of YOLO11s/m/l

**Fixing Low mAP / Poor Performance:**

- Train for more epochs: `epochs=200` or `epochs=300`
- Increase image size: `imgsz=1280`
- Use a larger model: YOLO11s/m instead of YOLO11n
- Check dataset quality and annotation accuracy
- Ensure balanced class distribution

**Fixing Dataset Format Errors:**

- Verify YOLO format: `<class_id> <x_center> <y_center> <width> <height>`
- Check that coordinates are normalized (0-1 range)
- Ensure `data.yaml` paths are correct
- Verify image-label pairs match (same filename, different extension)

**Fixing Export Failures:**

- Update Ultralytics: Add `ultralytics>=8.3.0` to requirements
- Check CUDA/TensorRT compatibility
- Try without optimization: `simplify=False` for ONNX
- Ensure model path is correct

**Fixing Prediction Job Failures:**

- Verify checkpoint ID is correct
- Check that model path `/checkpoint/weights/best.pt` exists
- Ensure source path is correct (use `/input` for datasets)
- Check logs: `flexai training logs <job-name>`

## Dataset Format Requirements

If you have a dataset in another format (COCO JSON, Pascal VOC, etc.), you'll need to convert it to YOLO format before using it with this blueprint.

### YOLO Format Specification

Each image should have a corresponding text file with the same name:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer class ID (0-indexed)
- `x_center`, `y_center`, `width`, `height`: Normalized coordinates (0-1 range)

Example label file (`image001.txt`):
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.15
```

### Converting Other Formats

For converting from other formats:
- **COCO JSON**: See [Ultralytics COCO format documentation](https://docs.ultralytics.com/datasets/detect/coco/)
- **Pascal VOC**: See [Ultralytics VOC format documentation](https://docs.ultralytics.com/datasets/detect/voc/)
- **Other formats**: Refer to [Ultralytics Data Formats guide](https://docs.ultralytics.com/datasets/detect/)

After converting your dataset to YOLO format, ensure your `data.yaml` has `path: /input` before uploading to FlexAI.

## Best Practices

### Dataset Preparation

- Use high-quality, diverse images
- Ensure balanced class distribution
- Include various lighting conditions, angles, and backgrounds
- Minimum 1500 images per class recommended
- Validate annotation accuracy before training

### Training

- Start with pre-trained weights (transfer learning)
- Use data augmentation (enabled by default)
- Monitor validation metrics, not just training loss
- Use early stopping to prevent overfitting
- Save checkpoints regularly

### Deployment

- Export to optimized formats (ONNX/TensorRT) for production
- Test on representative images before deployment
- Set appropriate confidence thresholds (0.25-0.5 typical range)
- Benchmark inference speed on target hardware

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Ultralytics CLI Guide](https://docs.ultralytics.com/usage/cli/)
- [YOLO11 Model Information](https://docs.ultralytics.com/models/yolo11/)
- [FlexAI Documentation](https://docs.flex.ai/)

---
