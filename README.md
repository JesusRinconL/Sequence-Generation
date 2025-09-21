# Sequence Generation of Microscopy Cell Images

This repository contains the complete implementation of a dual-model generative system for producing temporally coherent sequences of realistic cell microscopy images and their corresponding sketch representations. The system builds upon the Pix2Pix architecture and includes recursive inference to simulate cell migration processes from limited initial inputs.

---

## Project Overview

This project aims to generate synthetic sequences of microscopic cell images (or others), which can be used to supplement real biomedical datasets in scenarios where data acquisition is expensive, limited, or invasive.

Specifically, it targets the temporal modeling of cell migration phenomena using partial input data such as sketches or static frames.

---

## Key Features

- **Dual Pix2Pix Models**: 
  - Model 1 generates realistic images from sketch–image pairs at time steps *t* and *t+1*.
  - Model 2 transforms the output back into sketch space for evaluation and cycle consistency.

- **Recursive Inference**:
  - Reuses generated images as input for the next step, enabling continuous sequence generation.

- **Image Quality Control**:
  - Preprocessing with morphological operations (e.g., erosion, dilation).
  - Comparison of multiple input configurations (resolutions, segmentation layers, etc.).

- **Evaluation Tools**:
  - Includes scripts and notebooks for qualitative and quantitative analysis of generated sequences.

---

## Repository Structure

 - checkpoints/ # To store model weights
 - data/ # To configure input data (adapted to doble-input)
 - datasets/basic/ # Dataset samples used for both training and testing
 - models/ # pix2pix model definitions and custom modifications
 - options/ # Training and inference configurations
 - results/.../test_latest/ # Output image sequences from inference example
 - scripts/ # Data preprocessing utilities
 - util/ # Support functions
 - Metricas_Evaluacion_Imagenes.ipynb # Notebook for metric image analysis
 - pix2pix.ipynb # Interactive version of the training pipeline
 - Sequences_Metrics_Evaluation.ipynb # Notebook for metric sequence analysis
 - train.py # Main training script
 - test.py # Standard inference script
 - test_recur.py # Recursive inference script

---

## How to Use

### 1. Setup

Install required dependencies of libraries.

Also a potent GPU is recomended.

### 2. Prepare the data

Place your dataset in datasets/ using the expected format:

A/ = Sketch images

B/ = Real images

Example: 
<img width="1024" height="512" alt="151" src="https://github.com/user-attachments/assets/16a745e0-7820-4d15-9760-ae9d70db5236" />


Organize them in time-paired folders if generating sequences.

### 3. Train the model
python train.py --dataroot ./datasets/basic --name dualseq-basic-v1 --model dual_gan --direction AtoB --n_epochs 100

Expected training performance:

<img width="651" height="652" alt="image" src="https://github.com/user-attachments/assets/8aaecaa8-c400-4f2d-b89c-f0e3d6d58ae8" />

### 4. Test the model
python test.py --dataroot ./datasets/basic --name dualseq-basic-v1 --model dual_gan --direction AtoB

### 5. Generate sequences (recursive)
python test_recur.py --dataroot ./datasets/basic --name dualseq-basic-v1 --model dual_gan --recursive True


## Evaluation

Use the provided notebooks:

Metricas_Evaluacion_Imagenes.ipynb & Sequences_Metrics_Evaluation.ipynb

To visualize and compute all the metrics.

## Research Context

This work was developed as part of an academic research project focused on synthetic data generation for biomedical image analysis, especially cell migration modeling in cancer research.

Future directions include:

 - Exploring transformer-based or autoregressive sequence generation.

 - Developing advanced metrics for temporal evaluation.

 - Simulate trajectories between two frames, with variable number of steps.


## Authors

Developed by Jesus Rincon Laguarta in collaboration with José Luis Blanco Murillo as part of a research thesis at Universidad Politécnica de Madrid.
