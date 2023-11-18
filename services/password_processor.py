import numpy as np

def password_processor():
    def __init__(self, pass_str: str):
        self._password_string = pass_str
        self._array_data = None

    def _process_data(self):
        self._array_data = _string_to_ascii_array(self._password_string)
        return _normalize_data(self._array_data)

    def _normalize_data(array_data):
        """
        Normalizes each character as a floating point value between 0 and 1

        Returns:
        np.ndarray: A numpy array containing normalized values that can be de-normlized into ASCII values.
        """
        normalized_arr = array_data / 255.0
        return normalized_arr

    def _string_to_ascii_array(input_string):
        """
        Converts each character of the input string to its ASCII decimal value and stores it in a numpy array.

        Args:
        input_string (str): The string to be converted.

        Returns:
        np.ndarray: A numpy array containing the ASCII decimal values of the characters in the input string.
        """
        ascii_values = [ord(char) for char in input_string]
        return np.array(ascii_values)
