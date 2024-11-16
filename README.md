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

#### Usage: 
To run the main script, provide a password and an image file path to initiate the reconstruction process.

##### Command-Line Example:

    python main.py <password> <path/to/image_file.jpg>

>\<password\>: The sequence required to accurately reconstruct the image.
\<path/to/image_file.jpg\>: Path to the image file you wish to reconstruct or evaluate.

#### Example:

    python main.py mySecretPassword sample_image.jpg

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
