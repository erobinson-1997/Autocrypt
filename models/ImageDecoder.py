# Created by Ethan Robinson
# Started: November 16th, 2023

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class ImageDecoder(Model):
    """
    ImageDecoder is a custom model that takes encoded image data and 
    decodes it into a recognizable image format using a series of 
    convolutional transpose and upsampling layers.
    """

    def __init__(self, in_val):
        """
        Initialize the ImageDecoder model.
        
        Args:
            in_val (tf.Tensor): Input tensor containing encoded data to be decoded.
        """
        super(ImageDecoder, self).__init__()
        self._input_value = in_val 
        self._input_length = in_val.shape[1]  # Stores the length of the input shape

    def _dencoder(self):
        """
        Builds the sequential decoding model.
        
        Returns:
            tf.keras.Sequential: A sequential model that decodes the input tensor
            into an image format through LSTM, reshape, and Conv2DTranspose layers.
        """
        input_layer = layers.Input(shape=(None, self._input_length, 3))  # Define input layer with dynamic length
        
        # Sequential decoder architecture
        decoder = tf.keras.Sequential([
            input_layer,
            layers.LSTM(300),  # LSTM layer for sequential data processing
            layers.Reshape((10, 10, 3)),  # Reshape to 10x10x3 to approximate image-like shape
            # Multiple Conv2DTranspose and UpSampling2D layers for upscaling to the original image size
            layers.Conv2DTranspose(32, (2, 2), activation='relu', padding='same'),
            layers.UpSampling2D((4, 4)),
            layers.Conv2DTranspose(64, (2, 2), activation='relu', padding='same'),
            layers.UpSampling2D((4, 4)),
            layers.Conv2DTranspose(128, (2, 2), activation='relu', padding='same'),
            layers.UpSampling2D((4, 4)),
            layers.Conv2DTranspose(256, (2, 2), activation='relu', padding='same'),
            layers.UpSampling2D((4, 4)),
            layers.Conv2DTranspose(256, (2, 2), activation='relu', padding='same'),
            layers.UpSampling2D((4, 4)),
            layers.Conv2DTranspose(256, (2, 2), activation='relu', padding='same'),
            layers.UpSampling2D((4, 4)),
            layers.Conv2DTranspose(256, (2, 2), activation='relu', padding='same'),
            layers.UpSampling2D((4, 4)),
            # Final Conv2DTranspose layer to output a single-channel image
            layers.Conv2DTranspose(1, kernel_size=(2, 2), activation='relu', padding='same'),
        ])

        return decoder

    def call(self, x):
        """
        Invokes the decoding model on input data.
        
        Args:
            x (tf.Tensor): Input tensor to be decoded.
        
        Returns:
            tf.Tensor: Decoded tensor representing the upscaled image.
        """
        decoded = self._dencoder()(x)  # Passes input through the decoder model
        print("decoded:", decoded.shape)  # Prints the shape of the decoded output
        return decoded
