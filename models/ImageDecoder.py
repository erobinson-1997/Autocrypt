# Created by Ethan Robinson
# Started: November 16th, 2023

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

class ImageDecoder(Model):
    """
    ImageDecoder is a custom model designed to decode encrypted image data.
    It uses a specific input sequence, acting like a 'password,' to reconstruct
    the original image format through a series of decryption layers.
    """

    def __init__(self, in_val):
        """
        Initialize the ImageDecoder model with an input 'key' sequence.
        
        Args:
            in_val (tf.Tensor): Input tensor containing the encoded 'password' data 
            used to unlock and decode the image.
        """
        super(ImageDecoder, self).__init__()
        self._input_value = in_val
        self._input_length = in_val.shape[1]  # Stores the length of the input shape

    def _dencoder(self):
        """
        Builds the sequential decoding model that interprets the input sequence.
        
        Returns:
            tf.keras.Sequential: A sequential model that decodes the input tensor
            into an image format by reshaping, upsampling, and applying decryption layers.
        """
        decoder = tf.keras.Sequential([
            layers.LSTM(300, input_shape=(self._input_length, 3)),  # LSTM layer for sequential data processing
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
        Invokes the decoding model with the provided input sequence.
        
        Args:
            x (tf.Tensor): Input tensor that serves as the 'password' to be decoded 
            into the original image.
        
        Returns:
            tf.Tensor: Decoded tensor representing the reconstructed image.
        """
        decoded = self._dencoder()(x)
        print("decoded:", decoded.shape)
        return decoded
