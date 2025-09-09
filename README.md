# Banknote Detector 💵🔍

A real-time banknote detection system using YOLOv11x for accurate identification of banknotes in live camera streams.

## Features

- **Real-time Detection**: Process live camera feeds with low latency
- **High Accuracy**: Achieves 98.37% mAP50 on validation dataset
- **Optimized Performance**: CUDA acceleration with mixed precision
- **Flexible Input**: Supports RTSP streams, webcam, and video files
- **Interactive Controls**: Adjustable confidence threshold and display options
- **Professional Model**: YOLO11x architecture fine-tuned for banknote detection

## Installation

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (recommended for optimal performance)
- uv package manager

### Setup
1. Clone the repository:
   ```bash
   git lfs install 
   git clone <repository-url>
   cd banknote_detector
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   uv run python --version
   ```

### Base Models

Before training your models, download the base YOLO weights:

1. Visit [Ultralytics GitHub Releases](https://github.com/ultralytics/assets/releases)
2. Download the required model files (e.g., `yolo11x.pt`, `yolo11l.pt`, `yolo12l.pt`)
3. Place them in the `model_training/base_models/` directory

**Note:** These pre-trained weights are used as starting points for fine-tuning on your banknote dataset.

## Usage

### Real-time Detection

Run the live camera stream detector:

```bash
uv run python live_camera_stream.py
```

**Controls:**
- `q` - Quit the application
- `c` - Toggle confidence threshold display
- `+` / `-` - Increase/decrease confidence threshold

### Configuration

The system uses the following default settings:
- **Model**: YOLO11x (best.pt)
- **Confidence Threshold**: 0.6
- **IoU Threshold**: 0.55
- **Input Resolution**: 640x640
- **Device**: CUDA with AMP

### Custom RTSP Stream

Edit the `rtsp_url` variable in `live_camera_stream.py`:

```python
rtsp_url = "rtsp://your-stream-url"
```

For webcam input, set `rtsp_url = 0`.

## Model Training

### Quick Start

Train a new model using the provided dataset:

```bash
uv run python model_training/model_training.py
```

## Dataset

### Download Sources

You can obtain banknote datasets from the following sources:

#### Public Datasets
- **[roboflow.com/fyp-leirw/banknote](https://universe.roboflow.com/fyp-leirw/banknote)**
- **[roboflow.com/anood-saad/classification-of-iraqi-money](https://universe.roboflow.com/anood-saad/classification-of-iraqi-money)**
- **[roboflow.com/mujahid-winslow-gj6uf/denomination-banknotes](https://universe.roboflow.com/mujahid-winslow-gj6uf/denomination-banknotes)**
- **[roboflow.com/esprit-l9mvj/currency-xkvsd](https://universe.roboflow.com/esprit-l9mvj/currency-xkvsd)**
- **[roboflow.com/new-0w8yu/currency-final](https://universe.roboflow.com/new-0w8yu/currency-final)**
- **[roboflow.com/the-bright-way-team/the-bright-way](https://universe.roboflow.com/the-bright-way-team/the-bright-way)**

**Note:** There are many public datasets available for banknote detection in Roboflow and other sources, just make sure to pick the one that best suits your needs.

#### Dataset Preparation

1. Download or collect banknote images
2. Annotate images in YOLO format (bounding boxes), if they are not already annotated
3. Organize in the expected directory structure
4. Create/Update `data.yaml` with correct paths

### Training Scripts

- `model_training.py` - Main training pipeline
- `hyperparameters_tuning.py` - Hyperparameter optimization
- `optimal_workers.py` - Worker optimization
- `create_val_split_from_test_split.py` - Dataset preparation

### Dataset Format

The project expects a YOLO-format dataset in `annotated_banknote_dataset/` or any other directory that should contain the `data.yaml` file and the `images` and `labels` directories, like this:

```
<dataset_directory>/data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Model Evaluation

### Performance Metrics

| Model | mAP50 | mAP50-95 | Precision | Recall | Status |
|-------|-------|----------|-----------|--------|--------|
| YOLO11x | 0.9838 | 0.7572 | 0.9872 | 0.9499 | ✅ **Recommended** |
| YOLO11l | 0.9869 | 0.8371 | 0.9981 | 0.9519 | Completed |
| YOLO12l | 0.9798 | 0.7420 | 0.9930 | 0.9535 | Completed |
| YOLO12l Fine-tuned | 0.9806 | 0.7432 | 0.9802 | 0.9516 | Fine-tuned |

### Inference Settings

All models evaluated with standardized parameters:
- Confidence Threshold: 0.6
- IoU Threshold: 0.55
- Device: CUDA
- Precision: AMP + Half Precision

## Project Structure

```
banknote-detector/
├── .git/                           # Git repository
├── .gitattributes                  # Git attributes
├── .gitignore                      # Git ignore rules
├── .python-version                 # Python version specification
├── .venv/                          # Virtual environment
├── LICENSE                         # Project license
├── README.md                       # Project documentation
├── pyproject.toml                  # Project configuration
├── uv.lock                         # Dependency lock file
├── live_camera_stream.py           # Main real-time detection script
├── banknote_detection_model_evaluation_report.html  # Evaluation report
├── yolo_export_formats_guide.md    # Export formats guide
├── annotated_banknote_dataset/     # Dataset (not included)
├── exported_models/                # Exported model formats
│   ├── yolo11x_banknote/
│   │   ├── onnx/
│   │   └── tensorrt/
├── helpers/                        # Helper scripts
│   ├── copy_dataset_files.py
│   ├── create_val_split_from_test_split.py
│   ├── delete_npy_files.py
│   └── edit_labels_to_one_label.py
├── model_outputs/                  # Trained model outputs
│   ├── banknote_detection_yolo11x_outputs/
│   │   ├── weights/
│   │   │   ├── best.pt            # Best model weights
│   │   │   └── last.pt            # Last epoch weights
│   │   └── results.png            # Training curves
├── model_training/                # Training utilities
│   ├── base_models/               # Pre-trained YOLO models
│   ├── model_training.py          # Main training script
│   ├── model_exportation.py       # Model export script
│   ├── hyperparameters_tuning.py  # Hyperparameter optimization
│   ├── optimal_workers.py         # Worker optimization
│   └── create_val_split_from_test_split.py
```

## Dependencies

- **ultralytics>=8.3.191**: YOLO implementation
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities
- **opencv-python**: Computer vision operations
- **numpy**: Numerical computing

## Development

### Code Quality

- Uses type hints and docstrings
- Follows PEP 8 style guidelines
- Modular architecture for maintainability

### Performance Optimization

- Mixed precision training (AMP)
- CUDA acceleration
- Optimized data loading with caching
- Efficient inference pipeline

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv11 implementation by Ultralytics
- PyTorch team for the deep learning framework
- OpenCV community for computer vision tools

---

*Built with ❤️ using modern AI and computer vision technologies*