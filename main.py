# Created by Ethan Robinson
# Started November 17th, 2023

import sys
import os
from services import password_processor
from services import image_processor
from models import ImageDecoder

from tensorflow.keras import losses

if __name__ == "__main__":
    # Main entry point of the script for decrypting an image using a 'password' input.
    # TODO: Add error checking to handle missing or incorrect arguments.

    # Command-line arguments: password and image file path
    password = sys.argv[1]  # Password sequence used for decryption
    image_file_path = sys.argv[2]  # Path to the encrypted image file

    # Display input password and image file for verification
    print("pwd:", password)
    print("img:", os.path.basename(image_file_path))

    # Process the password to generate a numerical sequence required by the ImageDecoder
    preprocessed_pwd = password_processor.password_processor(password).__call__()
    print("pre-processed:", preprocessed_pwd.shape)  # Output the shape of the processed password

    # Process the image to get it in a format suitable for the model to reconstruct
    preprocessed_img = image_processor.image_processor(image_file_path).__call__()
    print("pre-processed img:", preprocessed_img.shape)  # Output the shape of the processed image

    # Retrieve image dimensions from the preprocessed image; these can be used in model configurations if needed
    img_width = preprocessed_img.shape[0]
    img_height = preprocessed_img.shape[1]

    # Initialize the ImageDecoder model with the preprocessed password sequence
    model_instance = ImageDecoder.ImageDecoder(preprocessed_pwd)
    
    # Compile the model with mean squared error as the loss function
    model_instance.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # Train the model on the processed password and image data
    # This fine-tunes the model to effectively 'reconstruct' the image using the provided password sequence
    history = model_instance.fit(
        preprocessed_pwd, preprocessed_img, 
        epochs=10,  # Number of training epochs
        batch_size=1,  # Batch size for training
        shuffle=True  # Shuffle data at each epoch
    )
