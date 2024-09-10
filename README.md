# PVAN

A code repository for the segmentation of pulmonary vessels, airway trees, and nodules.

## Setup

1. **Install Environment**: Run `install.sh` to set up the required environment.
2. **Configure File Paths**: Ensure that file paths are properly configured before proceeding.

## Training

- **Run Training**: Execute `train.sh` to start the training process. Make sure to configure the file paths beforehand.

## Prediction

- **Run Prediction**: Use `predict.sh` for inference. Ensure that file paths are properly configured before running the script.

## Evaluation

- **Segmentation Performance Evaluation**: Utilize the `ct_seg_perf` function in `val.py` to evaluate the segmentation performance.
