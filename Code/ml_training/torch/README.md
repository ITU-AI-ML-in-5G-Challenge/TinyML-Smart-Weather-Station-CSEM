# Torch folder
This folder contains all the code related to the machine learning part implemented in PyTorch, as well the conversion of the model to Tensorflow Lite. it is organised as follow :
- dataset*.py : all the PyTorch datasets for loading the different kind of data
- models.py : all the ML models used : implementation of Wei architectures, and also ACDNet
- train.py : train a PyTorch model
- train_transfer_lrn : retrain only a small amount of layer of a pre-trained Pytorch model
- onnx_to_tflite : transform onnx model to tflite (deprecated)
- torch_to_tf_graph : from a trained Torch model, create a TensorFlow graph for inference
- tf_graph_to_tflite : from a Tensorflow graph, create a tflite model
- utils.py : utils functions used in training