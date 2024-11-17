# Created by Ethan Robinson
# Started November 17th, 2023

import sys
import os
from services import password_processor
from services import image_processor
from models import ImageDecoder

from tensorflow.keras import losses
from tensorflow.keras.models import load_model

def train_model(password, image_file_path, model_name):
    """
    Train the model to reconstruct an image based on a specific input sequence.
    The trained model is saved to the file system for future use.
    
    Args:
        password (str): The password sequence used for training.
        image_file_path (str): Path to the image file to be reconstructed.
        model_name (str): Name of the model file to save.
    """
    print("Training Mode")
    print("pwd:", password)
    print("img:", os.path.basename(image_file_path))

    # Process the password
    preprocessed_pwd = password_processor.password_processor(password).__call__()
    print("pre-processed password shape:", preprocessed_pwd.shape)

    # Process the image
    preprocessed_img = image_processor.image_processor(image_file_path).__call__()
    print("pre-processed image shape:", preprocessed_img.shape)

    # Initialize and compile the model
    model_instance = ImageDecoder.ImageDecoder(preprocessed_pwd)
    model_instance.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # Train the model
    history = model_instance.fit(
        preprocessed_pwd, preprocessed_img,
        epochs=10,
        batch_size=1,
        shuffle=True
    )

    # Save the trained model
    model_instance.save(f"{model_name}.h5")
    print(f"Model saved as {model_name}.h5")

def reconstruct_image(password, model_name):
    """
    Reconstruct an image using a saved model and a specific input sequence.
    
    Args:
        password (str): The password sequence used for reconstruction.
        model_name (str): Name of the saved model file to load.
    """
    print("Reconstruction Mode")
    print("pwd:", password)
    print("model:", model_name)

    # Load the saved model
    model_instance = load_model(f"{model_name}.h5", custom_objects={"ImageDecoder": ImageDecoder.ImageDecoder})

    # Process the password
    preprocessed_pwd = password_processor.password_processor(password).__call__()
    print("pre-processed password shape:", preprocessed_pwd.shape)

    # Reconstruct the image
    reconstructed_img = model_instance(preprocessed_pwd)
    print("Reconstruction complete. Image shape:", reconstructed_img.shape)

    # Save the reconstructed image
    output_image_path = f"reconstructed_image.png"
    image_processor.save_image(reconstructed_img, output_image_path)
    print(f"Reconstructed image saved as {output_image_path}")

if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [<args>]")
        print("Commands:")
        print("  train <password> <image_file_path> <model_name>")
        print("  reconstruct <password> <model_name>")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        if len(sys.argv) != 5:
            print("Usage: python main.py train <password> <image_file_path> <model_name>")
            sys.exit(1)
        password = sys.argv[2]
        image_file_path = sys.argv[3]
        model_name = sys.argv[4]
        train_model(password, image_file_path, model_name)

    elif command == "reconstruct":
        if len(sys.argv) != 4:
            print("Usage: python main.py reconstruct <password> <model_name>")
            sys.exit(1)
        password = sys.argv[2]
        model_name = sys.argv[3]
        reconstruct_image(password, model_name)

    else:
        print(f"Unknown command: {command}")
        print("Commands:")
        print("  train <password> <image_file_path> <model_name>")
        print("  reconstruct <password> <model_name>")
        sys.exit(1)
