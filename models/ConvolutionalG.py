# Created by Ethan Robinson
# Started: November 16th, 2023

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class decoder(Model):
    def __init__(self, in_val):
        self._input_value = in_val 

    def _dencode(self):
        input_layer = layers.Input(shape=(None, 1, 1))
        
        decoder = tf.keras.Sequential([
            input_layer,
            layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same'),
        ])

        return decoder
    
    
    def call(self, x):
        decoded = self._dencode(x)
        return decoded