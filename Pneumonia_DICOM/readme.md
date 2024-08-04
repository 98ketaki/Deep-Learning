# Pnuemonia Detection 

## Project Overview
This project involves the use of PyTorch, PyTorch Lightning, and torchvision to detect pneumonia from chest X-ray DICOM images from the RSNA Pneumonia Detection Challenge

## Model
Model
The model is based on a modified ResNet18 architecture, where the first convolutional layer and final fully connected layer are adapted to the specifics of the problem—processing single-channel X-ray images and binary classification.

## Training
Training involves using a binary cross-entropy loss function with logits, optimized using Adam. Metrics such as accuracy, precision, and recall are logged for both training and validation phases.

## Evaluation
The model's performance is evaluated on the validation dataset using metrics like accuracy, precision, recall, and confusion matrices.


I completed this project as part of the coursework for the "Deep Learning with PyTorch for Medical Image Analysis" on Udemy.
