import cv2
import numpy as np

"""
    Created by Ethan Robinson

    The image_processor class is responsible for managing and processing the image file that gets encrypted. The
    image_processor class is initalized with the file path to the image, and it converts the data into a numpy 
    array that is normalized to values between 0 and 1.
"""

class image_processor():
    def __init__(self, img_path: str):
        self._image_path = img_path
        
    def __call__(self):
        return self._process_data()

    def _process_data(self):
        # Image to array
        array_data = self._img_to_array(self._image_path)

        # Normalize each pixel
        norm = self._normalize_data(array_data)

        # Expand the data out a dimension to be presented to the machine learning algorithm
        expanded = np.expand_dims(norm, axis=0) # (1, 4, 1)

        return expanded

    # Assuming each pixel is a set of 3 numeric values between 0 and 255, divide by 255.
    def _normalize_data(self, array_data):
        normalized_arr = array_data / 255.0
        return normalized_arr

    def _img_to_array(self, image_path):
        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to a NumPy array
        numpy_array = np.array(image)
        print(numpy_array.shape)
        return numpy_array


