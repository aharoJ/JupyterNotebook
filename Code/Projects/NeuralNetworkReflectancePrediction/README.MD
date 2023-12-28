# Project Structure Overview

This project is organized into several directories, each containing different types of files that contribute to the neural network for predicting material reflectance. Here's a quick guide:

### BRDF_Database/
Contains color data from various materials. For our project, we focused specifically on the color red in fabrics.
- `red-fabric.txt`: Data file for red fabric color information.

### NeuralNetworkReflectancePrediction/
- `JupyterNotebook`: Interactive notebook with all the code and documentation for the neural network model.

### CPP/
- Source code in C++ used to process data from its original format into a text format we can use.

### weights/ 
- `Final weights.txt`: This file contains the neural network's learned weights after training, which are crucial for making predictions.

### etc/
A miscellaneous section that includes extra resources, source files, and outputs of the neural network in image format (png).

For a detailed look into the project, each folder is named clearly to guide you through the different components that make up the neural network system. Additionally, the pdf gives guidance onto the task at hand.
