# Retinal Vessel Segmentation

Deep learning project for retinal vessel segmentation on DRIVE and CHASEDB1.

## Project Scope

- Task: image segmentation for retinal blood vessels.
- CNN model: DeepLabV3+-ResNet50.
- Attention model: SegFormer-B0.
- Datasets: DRIVE and CHASEDB1.
- Training method: transfer learning.
- Demo application: Streamlit.

## Project Structure

```text
.
|-- app/
|   `-- streamlit_app.py
|-- dataset/
|   |-- CHASEDB1_processed_dataset.zip
|   |-- DRIVE_processed_dataset.zip
|   `-- drive_test_dataset.pt
|-- notebooks/
|   |-- 00_EDA_Data.ipynb
|   |-- 01_Pretraining.ipynb
|   |-- 02_SegFormer_B0_DRIVE.ipynb
|   |-- 03_DeepLabV3Plus_ResNet50_CHASE_to_DRIVE.ipynb
|   |-- 04_SegFormer_B0_CHASE_to_DRIVE.ipynb
|   `-- environment_setup.ipynb
|-- src/
|   |-- models/
|   |   |-- deeplabv3plus_resnet50.py
|   |   |-- segformer.py
|   |   |-- best_deeplabv3plus_resnet50.pth
|   |   `-- best_segformer_b0.pth
|   |-- utils/
|   |-- dataset.py
|   |-- evaluate.py
|   |-- predict.py
|   `-- train.py
|-- README.md
`-- requirements.txt
```

Generated files such as report figures, temporary notebook exports, and Markdown notes are kept outside the core runtime path or ignored by Git.

## Training Pipeline

The main training pipeline is implemented in notebooks:

- `notebooks/03_DeepLabV3Plus_ResNet50_CHASE_to_DRIVE.ipynb`
  - DeepLabV3+-ResNet50.
  - CHASEDB1 source training followed by DRIVE fine-tuning.
  - BCE + Tversky loss for class-imbalanced vessel segmentation.

- `notebooks/04_SegFormer_B0_CHASE_to_DRIVE.ipynb`
  - SegFormer-B0 with pretrained transformer encoder.
  - CHASEDB1 source training followed by DRIVE fine-tuning.
  - BCE + Dice loss for binary vessel segmentation.

Additional notebooks:

- `notebooks/00_EDA_Data.ipynb`: dataset exploration.
- `notebooks/01_Pretraining.ipynb`: dataset preprocessing/export workflow.
- `notebooks/02_SegFormer_B0_DRIVE.ipynb`: earlier DRIVE-only SegFormer experiment.
- `notebooks/environment_setup.ipynb`: environment check.

Supporting scripts:

- `src/evaluate.py`: evaluates checkpoints and tunes threshold.
- `app/streamlit_app.py`: Streamlit demo for uploaded fundus images.
- `src/models/deeplabv3plus_resnet50.py`: DeepLabV3+-ResNet50 wrapper using `segmentation-models-pytorch`.
- `src/models/segformer.py`: SegFormer-B0 implementation.
- `src/export_test_images.py`: exports sample DRIVE test images for app testing.
- `src/export_test_dataset.py`: builds a local test dataset file for evaluation.

## Evaluation Results

Evaluated on `dataset/drive_test_dataset.pt` with threshold tuning.

| Model | Best Threshold | Dice | IoU | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepLabV3+-ResNet50 | 0.85 | 0.7173 | 0.5596 | 0.9487 | 0.6960 | 0.7471 |
| SegFormer-B0 | 0.50 | 0.7160 | 0.5579 | 0.9495 | 0.7070 | 0.7308 |

## Run Evaluation

DeepLabV3+-ResNet50:

```powershell
python src\evaluate.py --model deeplabv3plus_resnet50 --checkpoint src\models\best_deeplabv3plus_resnet50.pth --data dataset\drive_test_dataset.pt --batch-size 1 --tune-threshold
```

SegFormer-B0:

```powershell
python src\evaluate.py --model segformer_b0 --checkpoint src\models\best_segformer_b0.pth --data dataset\drive_test_dataset.pt --batch-size 1 --tune-threshold
```

## Run App

```powershell
streamlit run app\streamlit_app.py
```

The app supports:

- model selection;
- fundus image upload;
- automatic fundus crop for composite uploaded images;
- probability map display;
- binary vessel mask prediction;
- overlay visualization;
- skeleton and vessel statistics.
