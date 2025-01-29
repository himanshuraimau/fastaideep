# Digit Classifier Project

A deep learning project that classifies digits (specifically 3s and 7s) using both simple linear models and CNNs implemented with PyTorch and FastAI.

## Project Structure

- `main.py` - Entry point of the application, orchestrates the entire training pipeline
- `data_preparation.py` - Handles data loading and preprocessing
- `data_visualization.py` - Contains utilities for visualizing images and plotting functions
- `model_training.py` - Implements training and validation loops
- `advanced_models.py` - Defines CNN and other neural network architectures
- `optimization.py` - Custom optimizer implementation
- `utils.py` - Helper functions for various operations

## Features

- Basic linear classifier for digit recognition
- Convolutional Neural Network (CNN) implementation
- Custom loss functions and optimization
- Data visualization utilities
- Modular code structure

## Requirements

- PyTorch
- FastAI
- matplotlib
- Python 3.x

## Usage

1. Import the required modules:
```python
from data_preparation import prepare_data
from model_training import train_model
from advanced_models import create_cnn_learner
```

2. Prepare your data:
```python
path = prepare_data()  # Downloads and extracts MNIST sample data
```

3. Train a model:
```python
# For linear model
train_model(linear_model, epochs=20, dl=dl, valid_dl=valid_dl, lr=1.0)

# For CNN
learn = create_cnn_learner(dls)
train_cnn(learn, epochs=1, lr=0.1)
```

## Model Architecture

- Linear Model: Simple binary classifier
- CNN Model: Based on ResNet18 architecture

## Performance

The model achieves binary classification of digits 3 and 7 with validation accuracy tracked during training.

## License

This project is open-source and available for educational purposes.