
# Olive-SmartScan_AI: An Object Detection-Based Counting System for both *On-Tree* and *Off-Tree* Olives

Intelligent Systems Project at University of Pisa - MSc Computer Engineering

## Overview Description

**Olive-SmartScan_AI System** uses an artificial intelligence model based on Convolutional Neural Networks to analyze the amount of olive present both On-Tree and Off-Tree through object detection and counting into a single deep network. The detection module is based on a YOLOv8 architecture, while for counting, the approach is based on detecting olives and counting them within the image downstream of the detection step through a specific algorithm. This automated solution, to give an example, will give farmers the ability to make a tree-by-tree assessment of the orchard, and thus be able to estimate with some accuracy which areas are the most productive. Overall, Olive-SmartScan AI system presents a promising solution for automated counting in the agricultural domain, offering significant potential for reducing manual labor and increasing productivity.

The main objective was to develop a system capable of using machine learning to estimate yield by imitating the procedure followed by experts in the field—based on observing and interpreting visible olives
directly on site—the results obtained are promising. It is reasonable to think that in the future with appropriate improvements it could be applied to give an estimate of the agricultural yield (e.g., per hectare) to improve harvesting operational efficiency.


## Required Software and Frameworks

To use and develop the proposed system, you need to have the following software and frameworks installed:

### Visual Studio Code (VSCode)

Visual Studio Code is a lightweight and powerful source code editor, widely used for writing and managing Python code. It allows you to run and debug code efficiently. You can download and install VSCode from the [official website](https://code.visualstudio.com/Download).

**Installation:**

1. Visit the [VSCode download page](https://code.visualstudio.com/Download).
2. Follow the instructions for your operating system.
3. Once installed, you can add extensions for Python and Jupyter to enhance your development experience.

### PyTorch

PyTorch is an open-source library for machine learning and deep learning, useful for working with tensors and building machine learning models. It is essential for training and evaluating models in our project.

**Installation:**

1. Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) to get the installation command specific to your operating system and hardware configuration.
2. Run the provided command to install PyTorch using pip or conda, depending on your setup.

## Training and Testing

### Detection-Model Training and Validation
To train the models in Cross-Validation, refer to the Jupyter Notebook named *train_val-jupyter_nb* located in the `src/training_detection_model` folder. This notebook contains the instructions and code needed to train the models with the provided data.

**Note:** During our project, we performed the training on a specific kernel associated with a virtual machine provided by the university. This virtual machine was equipped with high-performance remote resources, which helped reduce training times and improve overall efficiency. Make sure to configure your execution environment to use adequate resources to achieve optimal results.

### System Testing
To run and test the system in overall, the main modules to refer to are:

- **`evaluate_counting_on_tree`**: Located in the `src/inference` folder, this module is responsible for evaluating the count of olives on trees using the trained models, providing further analysis of the results.
- **`evaluate_counting_out_tree`**: Also in the `src/inference` folder, this module evaluates the count of olives off trees using the trained models, providing further analysis of the results.

These modules are designed to provide an accurate assessment of the model's performance and to test the system in realistic scenarios.

## Package Structure
```
├───datasets
│   ├───cross-validation
│   │   ├───folders
│   │   │   ├───fold_0
│   │   │   ├───fold_1
│   │   │   ├───fold_2
│   │   │   ├───fold_3
│   │   │   └───fold_4
│   │   │
│   │   └───rounds
│   │       ├───ROUND_0
│   │       │   ├───test
│   │       │   ├───train
│   │       │   └───val
│   │       ├───ROUND_1
│   │       │   ├───test
│   │       │   ├───train
│   │       │   └───val
│   │       ├───ROUND_2
│   │       │   ├───test
│   │       │   ├───train
│   │       │   └───val
│   │       ├───ROUND_3
│   │       │   ├───test
│   │       │   ├───train
│   │       │   └───val
│   │       └───ROUND_4
│   │           ├───test
│   │           ├───train
│   │           └───val
│   │
│   ├───evaluation_datasets
│   │   ├───oliveOff-Tree
│   │   └───oliveOn-Tree
│   │
│   └───full_dataset
│       ├───images
│       └───labels
│
├───docs
│   └───Paper-Olive-SmartScan_AI-(ENG).pdf
│
├───final_models
│   └───checkpoints
│       ├───1_YOLOv8 Nano
│       ├───2_YOLOv8 Small
│       ├───3_YOLOv8 Medium
│       ├───4_YOLOv8 Large
│       └───5_YOLOv8 XLarge
└───src
    ├───data_management
    ├───inference
    └───training_detection_model
```
## Documentation Report

You can find more details about our work related to this project by reading the **Paper** we appropriately wrote to summarize it by clicking on this link [Olive-SmartScan_AI: An Object Detection-Based Counting System for both On-Tree and Off-Tree Olives](/docs/Paper-Olive-SmartScan_AI-(ENG).pdf).

## Authors

* Gaetano Sferrazza - *e-mail*: g.sferrazza@studenti.unipi.it - *GitHub Profile*: [ [@g-sferr](https://github.com/g-sferr) ]
* Francesco Bruno - *e-mail*: f.bruno10@studenti.unipi.it - *GitHub Profile*: [ [@francescoB1997](https://github.com/francescoB1997) ]

