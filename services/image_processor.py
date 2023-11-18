import cv2
import numpy as np

class image_processor():
    def __init__(self, img_path: str):
        self._image_path = img_path
        
    def __call__(self):
        return self._process_data()

    def _process_data(self):
        array_data = self._img_to_array(self._image_path)
        norm = self._normalize_data(array_data)
        expanded = np.expand_dims(norm, axis=0) # (1, 4, 1)
        return expanded


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


