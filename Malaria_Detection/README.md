## Malaria Detection using Transfer Learning with ResNet50 Architecture

This project aims to explore the potential of deep learning models for malaria diagnosis by detecting malaria parasites in blood smear images. The project uses transfer learning with the ResNet50 architecture to achieve high accuracy in detecting malaria parasites.

Data

The training, testing, and hidden image data, as well as the labels in CSV format for training and testing samples, are provided. The labels for the hidden data are not given.

Model

The project uses the 'timm' library in PyTorch to load the pre-trained ResNet50 model and modifies the last layer to classify the images into two classes: malaria infected and normal. The torchvision.transforms module is used to implement data augmentation techniques, such as resizing and normalization. The DataLoader is used to create batches of images and labels for training and testing.

Training

The model is trained for 10 epochs using a batch size of 32. The cross-entropy loss function and the Adam optimizer are used to train the model. The learning rate is set to 0.001, and a learning rate scheduler is used to adjust the learning rate during training. The model is evaluated on the training and testing datasets to monitor the training progress, and the accuracy and loss are reported.

The complete code and report is available in the repository as "code.py" and "Report.pdf"
