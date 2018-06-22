# Digit Recognition 
This repository provides an implementation of a convolutional neural network based model that recognizes handwritten digits. 
The designed model was trained and tested on the standard MNIST dataset.

This repository also provides a Python script that serves the following purposes:

1. Processes an image to make it compatible with the MNIST dataset format

2. Restores the saved model and predicts the digit in the processed image

You may modify the script to process multiple images and generate corresponding predictions.

# Credits 
Cheers to Hvass Laboratories for great tutorials on TensorFlow.

# Dependencies
Languages - Python

Frameworks - Matplotlib, Scikit-learn, Numpy, PIL, TensorFlow

Additional environments - Jupyter Notebook

# Train, test and save the model
Run 'digit_recognition.ipynb' with Jupyter Notebook. Files related to the model are saved in 'saved_models'(saved_models/) folder.

# Using the saved model for predictions
Go to 'saved_models'(saved_models/) folder. Place your image of .png extension in the 'img'(saved_models/img/) folder and rename it to '1'.
Run 'predict.py'. The processed image compatible with MNIST dataset format is generated in 'saved_models'(saved_models/) folder as 'sample'
with .png format. 
Predicted digit is displayed on the console window. I have provided a sample image for reference.

# Result
Results of training and testing the model on MNIST dataset are as follows:

Accuracy on Training-Set: 100 %

Accuracy on Test-Set: 98.2 % 
