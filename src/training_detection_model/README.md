## Training and Evaluation Using `train_val-jupyter_nb.ipynb`

The file `train_val-jupyter_nb.ipynb` has been used extensively in the project for training and evaluating different versions of the YOLOv8 models. Below is a detailed overview of its use:

### Training of YOLOv8 Models

- **Purpose:** The notebook was utilized to train various versions of the YOLOv8 framework using Cross-Validation. This approach ensures that each model is trained and validated across different subsets of the dataset, which helps in assessing its performance more robustly.
- **Process:** The notebook includes code for:
  - Loading the dataset and preparing it for training.
  - Configuring different YOLOv8 model versions (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8) for training.
  - Executing training sessions with Cross-Validation to ensure each model's performance is evaluated across different data splits.

### Performance Metrics Extraction

- **Purpose:** Besides training, the notebook was also used to extract and analyze performance metrics for each model version.
- **Metrics:** The following metrics were evaluated:
  - Precision
  - Recall
  - F1-Score
  - mAP@50
- **Process:** The notebook provides scripts to:
  - Extract relevant performance metrics after training each model.
  - Compare these metrics across different YOLOv8 model versions.

### Model Selection

- **Objective:** Based on the extracted metrics, the best-performing model was selected for our specific application, which involves detecting olives both on trees and off trees. This step is explained in detail in the relevant “Experimentla Results” section of the paper containing the details of our study found in the “docs” folder
- **Process:** The notebook facilitated:
  - Comparing the performance of various YOLOv8 models.
  - Selecting the model with the highest accuracy and best performance metrics for the task of Olive On-Tree and Off-Tree detection.

This notebook is crucial for understanding how each YOLOv8 model version performs and aids in selecting the most suitable model for our detection goals.

