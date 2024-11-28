[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)


# Dynamic_knowledge_distillation

This repository contains a Python script demonstrating an adaptive knowledge distillation approach for compressing convolutional neural networks (CNNs). The script uses TensorFlow and Keras to implement various model compression techniques.
## Features

Knowledge distillation from a teacher model to a smaller student model
Curriculum learning by sorting training data from easy to hard examples
Adaptive temperature and alpha scheduling during distillation
Model pruning to reduce parameter count
Quantization-aware training for further model compression

## Key Components

Teacher model: A larger CNN trained on CIFAR-10
Student model: A smaller CNN architecture
Distiller: Custom Keras model for knowledge distillation
Pruning: TensorFlow Model Optimization toolkit for weight pruning
Quantization: TFLite conversion with quantization

## Usage

Install required dependencies (TensorFlow, Keras, TensorFlow Model Optimization, Larq)
Run the script to train the teacher model, perform knowledge distillation, apply pruning, and quantize the final model
The compressed model is saved as a TFLite file for deployment on resource-constrained devices

## Results
The script compares the performance of:

The original teacher model
The distilled and compressed student model
A student model trained from scratch

Evaluation metrics include categorical accuracy and model size.
