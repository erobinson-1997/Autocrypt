# Autoencrypt
Author: Ethan Robinson
Started: November 16, 2023

## Overview
Autoencrypt is a machine-learning-based image reconstruction tool that reconstructs specific images based on a unique input sequence, which functions similarly to a "password." Rather than encryption or decryption, this project focuses on using trained neural networks to produce an accurate reconstruction of an image only when provided with the correct input sequence.

## Table of Contents
Overview
Features
Installation
Usage
Project Structure
Implementation Details

## Features
Input-Specific Image Reconstruction: The neural network reconstructs an image only when the correct input sequence is provided.
Zero-Padded Centered Images: Trains models to handle smaller images, reconstructing them as centered on a black background to accommodate dimension constraints.
Noise Generation for Incorrect Inputs: The model is trained to output random noise for all input sequences that do not match the specific image-generating input.
Customizable Neural Network Architecture: Flexible and modifiable for experimentation with different model structures.
Planned Enhancements
Zero-Padded Reconstruction: For smaller images, the model will reconstruct them as centered on black backgrounds using zero-padding. This adjustment accommodates the model’s output dimension requirements for generating consistently large images.
Noise Generation for Incorrect Inputs: The model will be trained to generate random noise for any input sequences other than the specific “password” input, adding a layer of security and unpredictability.

## Installation

#### Clone the Repository:

    git clone https://github.com/erobinson-1997/Autoencrypt.git
    cd Autoencrypt

#### Install Dependencies: 
Make sure you have Python installed, then install required packages:

    pip install -r requirements.txt

## Usage: 
To run the main script, provide a password and an image file path to initiate the reconstruction process.

#### Workflow
1. Training the Model:

    - The model is trained to reconstruct an image based on a specific input sequence (password).
    - After training, the model is saved to the file system for future use.
    - The user can confirm the reconstruction quality by comparing the reconstructed image with the original.

2. Reconstructing an Image:

    - The user selects a saved model from the file system and provides the correct input sequence (password).
    - The model attempts to reconstruct the image based on the input sequence.

3. Secure Workflow:

    - Once satisfied with the reconstruction quality, the user can delete the original image.
    - The model and input sequence together can be used to regenerate the image later.

#### Command-Line Usage

##### Train the Model
To train the model and save it to the file system:

    python main.py train <password> <path/to/image_file.jpg> <model_name>
>\<password>: The sequence required to train the model.
<path/to/image_file.jpg>: Path to the image to be reconstructed.
<model_name>: Name of the model file to save (e.g., model_1.h5).

Example:

    python main.py train mySecretPassword image.jpg my_model

###### Reconstruct an Image:
To reconstruct an image using a saved model:


    python main.py reconstruct <password> <model_name>

>\<password>: The correct input sequence for the model.
<model_name>: Name of the saved model file.

###### Example

python main.py reconstruct mySecretPassword my_model

## Project Structure
main.py: Entry point for running the reconstruction process.
services/password_processor.py: Converts the password string into a format suitable for model input.
services/image_processor.py: Preprocesses images to prepare them for training or reconstruction.
models/ImageDecoder.py: Defines the neural network architecture for reconstructing images based on input sequences.
README.md: Documentation and instructions for the project.
requirements.txt: Lists the necessary packages for the project.

## Implementation Details

- Image Reconstruction: The ImageDecoder model uses layers such as LSTMs and Conv2D Transpose to upscale and reshape encoded data back into image form.

- Zero-Padding: Smaller images are padded with zeros during training to ensure consistent output dimensions, allowing the model to generate centered reconstructions on a black background.

- Noise for Incorrect Inputs: For any incorrect "password" sequence, the model produces a random noise image, reducing unintended access and enhancing security.
