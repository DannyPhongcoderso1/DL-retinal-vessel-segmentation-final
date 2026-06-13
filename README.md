# Cross-Domain Retinal Vessel Intelligence Platform

Deep Learning-based Retinal Vessel Segmentation and Vascular Analysis using DeepLabV3+-ResNet50, SegFormer-B0, and Transfer Learning.

The project implements and compares two modern semantic segmentation approaches:

- **CNN-based model:** DeepLabV3+-ResNet50, using a ResNet50 backbone pretrained on ImageNet.
- **Attention/Transformer-based model:** SegFormer-B0, using a Transformer encoder with self-attention.
- **Training strategy:** Cross-domain transfer learning from **CHASEDB1** to **DRIVE**.
- **Demo application:** A Streamlit app that lets users upload retinal images, predict masks, display overlays, and analyze vessel morphology.

![Project pipeline](docs/images/project_pipeline.png)

## Project Summary

Retinal vessel segmentation is an important step in computer-aided medical image analysis systems, especially for problems related to diabetic retinopathy, hypertensive retinopathy, and vascular abnormalities. The system input is an RGB fundus image, and the output is a binary mask representing the vessel region.

The goal of this project is to build an end-to-end pipeline covering data preprocessing, data augmentation, transfer learning, model evaluation, and a demo application. The two selected models satisfy the project requirements by including one CNN architecture and one architecture with an attention mechanism.

## Meeting The Project Requirements

| Requirement | Implementation in the repo |
| --- | --- |
| The problem belongs to Image Segmentation | Retinal vessel segmentation from fundus images |
| CNN architecture | DeepLabV3+-ResNet50 in `src/models/deeplabv3plus_resnet50.py` |
| Attention/Transformer architecture | SegFormer-B0 in `src/models/segformer.py` |
| Suitable datasets | DRIVE and CHASEDB1 |
| Preprocessing, data augmentation | Resize, ImageNet normalization, binary masks, flips, rotations, brightness/contrast |
| Training two models | Notebook `03` for DeepLabV3+-ResNet50, notebook `04` for SegFormer-B0 |
| Training method | Transfer learning with CHASEDB1 source training -> DRIVE fine-tuning |
| Evaluation metrics | Dice/F1, IoU, Accuracy, Precision, Recall, validation/test loss |
| Simple application | Streamlit app in `app/streamlit_app.py` |

## Data

The project uses two benchmark datasets for retinal vessel segmentation:

| Dataset | Role | Characteristics |
| --- | --- | --- |
| **CHASEDB1** | Source domain | RGB retinal images, each image manually annotated by experts, used to learn general vessel features |
| **DRIVE** | Target domain | Retinal images for the vessel extraction task, used for fine-tuning and final evaluation |

In the main experiment notebooks, images and masks are resized to 512 x 512 for training and evaluation. Masks are converted into single-channel binary masks with shape `[B, 1, H, W]`.

![Dataset examples](docs/images/dataset_examples.png)

## Preprocessing And Augmentation

The preprocessing pipeline consists of the following main steps:

1. Read fundus images in RGB format and masks in grayscale.
2. Resize images and masks to the same spatial resolution.
3. Normalize images using the ImageNet mean and standard deviation to match the pretrained backbone/encoder.
4. Convert masks to binary 0/1 values and add a channel dimension.
5. Apply data augmentation on the training set using horizontal flip, vertical flip, rotation, and brightness/contrast adjustments.

![Preprocessing and augmentation](docs/images/preprocessing_augmentation_examples.png)

## Model Architecture

### DeepLabV3+-ResNet50

DeepLabV3+-ResNet50 is used as the representative CNN model. The model is built with `segmentation_models_pytorch.DeepLabV3Plus` using:

- `encoder_name="resnet50"`.
- `encoder_weights="imagenet"` during transfer learning.
- `classes=1` to produce a single-channel logit map for binary segmentation.
- `activation=None` to return raw logits, which is suitable for `BCEWithLogitsLoss` and composite losses.

The main code is located at:

```text
src/models/deeplabv3plus_resnet50.py
```

The model validates the input and output shapes in `forward`:

```text
Input : [B, 3, H, W]
Output: [B, 1, H, W]
```

### SegFormer-B0

SegFormer-B0 is the attention-based model in this project. It uses `SegformerForSemanticSegmentation` from Hugging Face Transformers. The attention mechanism is implemented in the MiT Transformer encoder of SegFormer, enabling the model to learn global contextual relationships across image regions.

The main code is located at:

```text
src/models/segformer.py
```

Key implementation details:

- Default pretrained checkpoint: `nvidia/segformer-b0-finetuned-ade-512-512`.
- Set `num_labels=1` for binary segmentation.
- Use `ignore_mismatched_sizes=True` when replacing a multi-class segmentation head with a single-channel head.
- Interpolate logits back to the original input resolution with `F.interpolate`.
- Ensure the final output shape is `[B, 1, H, W]`.

## CHASEDB1 -> DRIVE Transfer Learning

Both models are trained using the same high-level strategy:

1. **Source training on CHASEDB1:** learn vessel morphology features from the source domain.
2. **Fine-tuning on DRIVE:** load the best CHASEDB1 checkpoint and continue fine-tuning on DRIVE with a smaller learning rate.
3. **Validation monitoring:** track validation Dice/IoU/loss after each epoch.
4. **Early stopping:** stop early if the validation metric does not improve after the patience window.
5. **Best checkpoint:** save the best checkpoint according to validation Dice or validation IoU, depending on the notebook.
6. **Final evaluation:** evaluate on the DRIVE test set and save results, history, and prediction figures.

Main notebooks:

| Notebook | Role |
| --- | --- |
| `notebooks/00_EDA_Data.ipynb` | Data exploration, image and mask inspection |
| `notebooks/01_Pretraining.ipynb` | Preprocessing, data export, DataLoader shape checks |
| `notebooks/02_SegFormer_B0_DRIVE.ipynb` | SegFormer-B0 experiments directly on DRIVE |
| `notebooks/03_DeepLabV3Plus_ResNet50_CHASE_to_DRIVE.ipynb` | Transfer learning DeepLabV3+-ResNet50 from CHASEDB1 to DRIVE |
| `notebooks/04_SegFormer_B0_CHASE_to_DRIVE.ipynb` | Transfer learning SegFormer-B0 from CHASEDB1 to DRIVE |
| `notebooks/environment_setup.ipynb` | Runtime environment checks on Kaggle |

## Training Configuration

| Component | DeepLabV3+-ResNet50 | SegFormer-B0 |
| --- | --- | --- |
| Input size | 512 x 512 | 512 x 512 |
| Optimizer | AdamW | AdamW |
| Weight decay | 1e-4 | 1e-4 |
| Source domain | CHASEDB1 | CHASEDB1 |
| Target domain | DRIVE | DRIVE |
| Source learning rate | Backbone 1e-5, Head 1e-4 | Encoder 1e-5, Head 1e-4 |
| Target learning rate | Backbone 5e-6, Head 5e-5 | Encoder 5e-6, Head 5e-5 |
| Loss | BCEWithLogits + Tversky | BCEWithLogits |
| Early stopping | Yes | Yes |
| Main validation metric | Dice/IoU | Dice/IoU |

## Training Curves

DeepLabV3+-ResNet50:

![DeepLab training curves](docs/images/deeplab_training_curves.png)

SegFormer-B0:

![SegFormer training curves](docs/images/segformer_training_curves.png)

## Quantitative Results

The table below summarizes the final results on the DRIVE test set after transfer learning. Logits are passed through sigmoid and binarized using the threshold specific to each model.

| Model | Threshold | Dice/F1 | IoU | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepLabV3+-ResNet50 | 0.85 | 0.7173 | 0.5596 | 0.9487 | 0.6960 | 0.7471 |
| SegFormer-B0 | 0.50 | 0.7160 | 0.5579 | 0.9495 | 0.7070 | 0.7308 |

![Metric comparison](docs/images/metric_comparison.png)

### Analysis

DeepLabV3+-ResNet50 achieves slightly higher Dice and IoU, and also a higher Recall. This indicates that the model tends to detect more vessel regions, especially small branches, but may also produce more false positives.

SegFormer-B0 has higher Accuracy and Precision. Its predicted masks are usually cleaner and contain less background noise, but some small vessel branches may be missed because Recall is lower.

## Visual Results

DeepLabV3+-ResNet50 prediction example:

![DeepLab prediction example](docs/images/deeplab_prediction_example.png)

SegFormer-B0 prediction example:

![SegFormer prediction example](docs/images/segformer_prediction_example.png)

The figures above are extracted from notebook outputs generated during experiments on Kaggle. Each figure contains the original image, ground truth, and prediction mask.

## Streamlit Demo App

The demo application is located at:

```text
app/streamlit_app.py
```

Main features:

- Upload retinal images in PNG/JPG/JPEG format.
- Select either SegFormer-B0 or DeepLabV3+-ResNet50.
- Adjust the prediction threshold.
- Generate a probability map and binary vessel mask.
- Display an overlay between the original image and the mask.
- Perform skeletonization and morphology analysis.
- Export the prediction mask, overlay image, and technical summary.

![App workflow](docs/images/app_workflow.png)

Note: the app is intended for learning and research purposes, not as a medical diagnosis tool.

## Repository Structure

```text
.
|-- app/
|   `-- streamlit_app.py
|-- dataset/
|   |-- CHASEDB1_processed_dataset.zip
|   `-- DRIVE_processed_dataset.zip
|-- docs/
|   `-- images/
|       |-- app_workflow.png
|       |-- dataset_examples.png
|       |-- deeplab_prediction_example.png
|       |-- deeplab_training_curves.png
|       |-- metric_comparison.png
|       |-- preprocessing_augmentation_examples.png
|       |-- project_pipeline.png
|       |-- segformer_prediction_example.png
|       `-- segformer_training_curves.png
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
|   |   `-- best_segformer_b0.pth
|   |-- utils/
|   |   |-- losses.py
|   |   `-- metrics.py
|   |-- evaluate.py
|   |-- export_test_dataset.py
|   |-- export_test_images.py
|   |-- predict.py
|   |-- train.py
|   `-- visualize_predictions.py
|-- [Deep Learning] REPORT.pdf
|-- README.md
`-- requirements.txt
```

Some artifacts are created after running notebooks or scripts:

- `dataset/drive_test_dataset.pt`: local evaluation test tensor file.
- `src/models/best_deeplabv3plus_resnet50.pth`: DeepLabV3+-ResNet50 checkpoint after downloading from Kaggle outputs.
- `outputs/predictions/`: prediction images generated by the visualization script.

## Local Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you are running on Kaggle, install the libraries in `requirements.txt` or run `notebooks/environment_setup.ipynb` first.

## Running Notebooks On Kaggle

Recommended workflow:

1. Upload or mount `DRIVE_processed_dataset.zip` and `CHASEDB1_processed_dataset.zip`.
2. Run `notebooks/environment_setup.ipynb` to verify the libraries.
3. Run `notebooks/00_EDA_Data.ipynb` to inspect the data.
4. Run `notebooks/01_Pretraining.ipynb` to validate preprocessing and the DataLoader.
5. Run `notebooks/03_DeepLabV3Plus_ResNet50_CHASE_to_DRIVE.ipynb` to train DeepLabV3+-ResNet50.
6. Run `notebooks/04_SegFormer_B0_CHASE_to_DRIVE.ipynb` to train SegFormer-B0.
7. Download the checkpoints and result files from `/kaggle/working`.

Important Kaggle artifacts:

```text
/kaggle/working/best_segformer_b0.pth
/kaggle/working/deeplabv3plus_resnet50_outputs/checkpoints/best_deeplabv3plus_resnet50_drive.pth
/kaggle/working/segformer_b0_chase_to_drive/results/segformer_b0_drive_results.csv
/kaggle/working/deeplabv3plus_resnet50_outputs/results/deeplabv3plus_resnet50_drive_test_results.csv
```

When bringing the files back to the local repo, rename the checkpoints as follows:

```text
src/models/best_segformer_b0.pth
src/models/best_deeplabv3plus_resnet50.pth
```

## Preparing The Local Test Dataset

The `.pt` file used by `src/evaluate.py` can be exported from the processed DRIVE zip file:

```powershell
python src\export_test_dataset.py `
  --zip dataset\DRIVE_processed_dataset.zip `
  --output dataset\drive_test_dataset.pt `
  --image-size 512
```

Check the exported test images:

```powershell
python src\export_test_images.py `
  --data dataset\drive_test_dataset.pt `
  --output outputs\app_test_images_full
```

## Evaluating Local Models

SegFormer-B0:

```powershell
python src\evaluate.py `
  --model segformer_b0 `
  --checkpoint src\models\best_segformer_b0.pth `
  --data dataset\drive_test_dataset.pt `
  --batch-size 4 `
  --threshold 0.50
```

DeepLabV3+-ResNet50:

```powershell
python src\evaluate.py `
  --model deeplabv3plus_resnet50 `
  --checkpoint src\models\best_deeplabv3plus_resnet50.pth `
  --data dataset\drive_test_dataset.pt `
  --batch-size 2 `
  --threshold 0.85
```

Evaluate both models if both checkpoints are available:

```powershell
python src\evaluate.py `
  --model all `
  --data dataset\drive_test_dataset.pt `
  --batch-size 2
```

Find the best threshold by Dice:

```powershell
python src\evaluate.py `
  --model segformer_b0 `
  --checkpoint src\models\best_segformer_b0.pth `
  --data dataset\drive_test_dataset.pt `
  --batch-size 4 `
  --tune-threshold `
  --threshold-min 0.30 `
  --threshold-max 0.85 `
  --threshold-step 0.05
```

## Generating Local Prediction Images

```powershell
python src\visualize_predictions.py `
  --model segformer_b0 `
  --checkpoint src\models\best_segformer_b0.pth `
  --data dataset\drive_test_dataset.pt `
  --threshold 0.50 `
  --num-samples 5 `
  --output-dir outputs\predictions
```

## Running The Application

```powershell
streamlit run app\streamlit_app.py
```

By default, the application reads checkpoints from `src/models/`. The repo currently includes the SegFormer-B0 checkpoint. If you want to run DeepLabV3+-ResNet50 in the app, download the DeepLab checkpoint from the Kaggle output and place it under the exact name `src/models/best_deeplabv3plus_resnet50.pth`.

## Key Code Locations

| Component | File |
| --- | --- |
| SegFormer-B0 attention model | `src/models/segformer.py` |
| DeepLabV3+-ResNet50 CNN model | `src/models/deeplabv3plus_resnet50.py` |
| Metric functions | `src/utils/metrics.py` |
| Loss functions | `src/utils/losses.py` |
| Evaluation CLI | `src/evaluate.py` |
| Prediction visualization | `src/visualize_predictions.py` |
| DRIVE test export | `src/export_test_dataset.py` |
| Streamlit app | `app/streamlit_app.py` |

## Limitations And Future Work

The main limitations of the project are the relatively small dataset size, the domain shift between CHASEDB1 and DRIVE that still affects small vessel branches, and the risk score in the app, which is currently only a rule-based illustrative indicator and not clinically validated.

Future work includes:

- Expand the dataset with additional retinal vessel segmentation benchmarks.
- Experiment with post-processing to reconnect broken vessel branches.
- Tune the threshold separately for each model and validation set.
- Add calibration and uncertainty estimation.
- Turn the app into a dashboard for comparing multiple models on the same image.

## References

1. Fraz, M. M., et al. (2012). An ensemble classification-based approach applied to retinal blood vessel segmentation. *IEEE Transactions on Biomedical Engineering*.
2. Staal, J., et al. (2004). Ridge-based vessel segmentation in color images of the retina. *IEEE Transactions on Medical Imaging*.
3. Chen, L.-C., et al. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *ECCV*.
4. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
5. Xie, E., et al. (2021). SegFormer: Simple and efficient design for semantic segmentation with Transformers. *NeurIPS*.
