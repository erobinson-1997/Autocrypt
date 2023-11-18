# Created by Ethan Robinson
# Started November 17th, 2023

import sys
from services import password_processor

if __name__ == "__main__":
    # TODO: add error checking 
    
    password = sys.argv[1]
    image_file_path = sys.argv[2]
    preprocessed_pwd = password_processor.password_processor(password)