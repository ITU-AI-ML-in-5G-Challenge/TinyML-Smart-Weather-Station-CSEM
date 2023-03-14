# TinyML Challenge 2022 - Tensorflow
This folder contains all the code related to the machine learning part implemented in Tensorflow, as well the conversion of the model to Tensorflow Lite. it is organised as follow :
- constants_ml.py : contains the constants (folder location, parameter to preprocess dataset, name model)
- dataloader*.py : contains function to load the corresponding dataset (without suffix = real dataset, vent = ventilator dataset)
- models.py : all the ML models used : implementation of Wei architectures
- train*.py : handles the training of the model:
  - optuna : will train multiples instance of the model by variating the hyperparameters and recording the best model
  - quantize_aware : will train the model with quantize aware techniques (if want to quantize the model after the training for Tflite)
- utils.py : contains all the utilitarities functions used in the training
- tf_to_tflite.py : transform a trained tf model into a tflite format