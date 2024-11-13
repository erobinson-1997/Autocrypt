"""
Created by Ethan Robinson

The password_processor class converts a password string into a normalized numpy array 
that can be used as input for a TensorFlow machine-learning model. This transformation 
involves encoding each character as an ASCII value, normalizing it, and reshaping it 
to fit the expected input shape for the model.
"""

import numpy as np

class password_processor():
    def __init__(self, pass_str: str):
        """
        Initializes the password_processor with the input password string.
        
        Args:
            pass_str (str): The password string to be processed.
        """
        self._password_string = pass_str

    def __call__(self):
        """
        Callable method that processes the password string by converting it to a 
        formatted numpy array for model input.
        
        Returns:
            np.ndarray: A formatted array derived from the password string, 
                        ready for model input.
        """
        return self._process_data()

    def _process_data(self):
        """
        Processes the password string through three steps: 
        ASCII conversion, normalization, and reshaping.
        
        Returns:
            np.ndarray: The final processed array in the shape (1, 4, 3).
                        This format aligns with the model’s expected input.
        """
        # Convert the password string to an array of ASCII values
        array_data = self._string_to_ascii_array(self._password_string)
        
        # Normalize the ASCII values to fall between 0 and 1
        norm = self._normalize_data(array_data)  # Shape: (4,)

        # Expand the dimensions to prepare for model input
        expanded = np.expand_dims(norm, axis=-1)  # Shape: (4, 1)
        expanded = np.expand_dims(expanded, axis=0)  # Shape: (1, 4, 1)

        # Repeat across the third dimension to achieve final shape (1, 4, 3)
        expanded = np.repeat(expanded, 3, axis=2)

        return expanded

    def _normalize_data(self, array_data):
        """
        Normalizes ASCII values to a range between 0 and 1.
        
        Args:
            array_data (np.ndarray): Array of ASCII values.

        Returns:
            np.ndarray: Array of normalized values between 0 and 1.
        """
        normalized_arr = array_data / 255.0
        return normalized_arr

    def _string_to_ascii_array(self, input_string):
        """
        Converts each character of the input string to its ASCII decimal value.
        
        Args:
            input_string (str): The string to be converted.

        Returns:
            np.ndarray: Array containing ASCII values of each character in the input string.
        """
        ascii_values = [ord(char) for char in input_string]
        return np.array(ascii_values)
