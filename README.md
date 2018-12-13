# NCCNN
Neighboring Collaborative Convolutional Neural Networks

Skip to [How to Run](#How-to-Run) Guide

See our proposal for this project [here](docs/NCCNNproposal.pdf)

## About this project

Combines K-Nearest Neighbors and Convolutional Neural Network (CNN) techniques with the use of capsules to create a collaborative network of neurons. This project was originially for the project requirement of the Machine Learning graduate level course at Worcester Polytechnic Institute (WPI). Our team decided to pursue an innovative way to compete in the [Google AI Open Images - Object Detection Track competition](https://www.kaggle.com/c/google-ai-open-images-object-detection-track) on Kaggle when it runs again next year. The data sets used in this project are linked in the [Data section](#Data) below.

### Methodology

Explains our methods and **what** we did

### Results

Explains our results and **how** we did.

## Data

### Original Data Sets used:

[Data and Validation sets](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/) in csv files can be found under the Data tab

[Class Names Data](https://storage.googleapis.com/openimages/web/download.html) is located under Meta Data on this page

### Actual Data Sets used:

[Berkeley DeepDrive](http://bdd-data.berkeley.edu/) : must create a log in to be able to download the data set

## How to Run

Follow what's below here for installation and training examples.

### Machine Requirements

In order to run our code your machine must have the following:
- 6GB CPU
- (Optional) 6GB GPU if running with GPU for faster processing

For TensorFlow to work:
- Ubuntu 16.04 or later
- Windows 7 or later
- macOS 10.12.6 (Sierra) or later (no GPU support)
- Raspbian 9.0 or later

If using GPU, also need:
- NVIDIA Graphics card with GPU drivers using Cuda 9.0
- cuDNN SDK 7.4
(Can update the above when installing TensorFlow below)
- NVIDIA Developer Account (Can be obtained for free at: https://developer.nvidia.com/developer-program)

### Installation

1. Clone the repository to your computer by using HTTPS or SSH (or download it) using the commands in the green drop down menu at the top of the repository. 

2. Update your system with:
   ```
   sudo apt-get update
   ```

3. Check for Python 3.6 (Not Python 3.7! This is not supported by TensorFlow yet.)

   To check the version you're running: 
   ```
   python -V
   ```

   To install: 
   ```
   sudo apt-get install python3.6
   ```

4. Install pip3:
   ```
   sudo apt install python3-pip
   ```

5. Install TensorFlow/TensorFlow-GPU if using GPU by following the steps on their website here:
   https://www.tensorflow.org/install/

   __Follow this guide in TensorFlow for GPU version:__ https://www.tensorflow.org/install/gpu

### Training

See [Tutorial Notebook](train_notebook.ipynb) for a tutorial.

See [Training Notebook](train_notebook.ipynb) for examples on how to train.
