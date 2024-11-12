# Created by Ethan Robinson
# Started: November 16th, 2023

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class ImageDecoder(Model):
    def __init__(self, in_val):
        super(ImageDecoder, self).__init__()
        self._input_value = in_val 
        self._input_length = in_val.shape[1]

    def _dencoder(self):
        input_layer = layers.Input(shape=(None, self._input_length, 3))
        
        decoder = tf.keras.Sequential([
            input_layer,
            layers.LSTM(300),
            layers.Reshape((10, 10, 3)),  # Reshape to a more 'image-like' shape
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
            layers.Conv2DTranspose(1, kernel_size=(2, 2), activation='relu', padding='same'),
        ])

        return decoder
    
    
    def call(self, x):
        decoded = self._dencoder()(x)
        print("decoded:", decoded.shape)
        return decoded
