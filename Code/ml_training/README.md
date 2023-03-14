# # TinyML Challenge 2022 - Machine Learning code
This folder contains everything concerning the training of the ML models and their conversion to tflite in the right format. It is organized as follow :  
- preprocess_data : contains the code to generate a valid dataset for a ML task from .wav files made from the MAX78000 acquisition part and the (ground truth) weather station information recordings. It also contains all the preprocessing tools
- tf : contains the models coded in tensorflow, the file to run the training part, and the files required to transform the resulted model into a tflite model suited for microcontrollers
- torch : contains the models coded in pytorch, the file to run the training part, and the files required to transform the resulted model into a tflite model. Nevertheless, the conversion can induce some operators that are not supported in the tflite micro library.
