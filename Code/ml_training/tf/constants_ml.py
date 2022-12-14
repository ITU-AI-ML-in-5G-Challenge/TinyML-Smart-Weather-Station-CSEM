import os
import tensorflow as tf
import numpy as np
import sys
from utils import Labels


DATE = 'test'
NAME_MODEL = 'final_model'
OUTPUT_FOLDER = 'results/outside/' + DATE + '/' + NAME_MODEL + '/'
USE_TENSORBOARD = False

TYPE_LABELS = Labels.RAIN_WIND

###########################################
#### Dataset transformation parameters ####
###########################################
SPLIT_FACTOR = 2
LOWPASS_FREQ = None
MERGE_BUCKET_SIZE = 2
RESAMPLE_FREQUENCY = None #SAMPLING_FREQUENCY / 2
TO_MFCC = False
# INDEX_HOURS = range(85, 89)

def compress_labels_wind(x):
    if x < 6:
        return 0
    elif x < 12:
        return 1
    else :
        return 2

def compress_labels_rain(x):
    if x >= 3 :
        return 2
    elif x >= 1 :
        return 1
    else :
        return 0


def compress_labels(labels):
    if TYPE_LABELS == Labels.RAIN:
        return np.vectorize(compress_labels_rain)(labels)
    elif TYPE_LABELS == Labels.WIND:
        return np.vectorize(compress_labels_wind)(labels)
    else: 
        return np.vectorize(compress_labels_rain)(labels[0]), np.vectorize(compress_labels_wind)(labels[1])



MODULE_TFLITE_PATH = 'u111/Ports/Targets/FTHR_Max78000/Variant_TFLite_vent/Tools/TinyML'

#RAM of 128K, but need space for other stuff
MAX_RAM_MEMORY = 80000
#FLASH of 518K
MAX_FLASH_MEMORY = 260000


# For ventilator dataset
STORE_ML_FOLDER_VENT = 'results/ventilator/last/'
PATH_MODEL_VENT = STORE_ML_FOLDER_VENT + NAME_MODEL
PATH_DATA_VENT = "data/office/ventilator"
AUDIO_LENGTH_VENT = 8000



#####################################
######### Do not modify #############
TF_FOLDER = OUTPUT_FOLDER + 'tf/'
MODEL_FILEPATH = TF_FOLDER + NAME_MODEL + '.pb'
CHECKPOINT_FILEPATH = TF_FOLDER + 'checkpoint/' + NAME_MODEL + '_checkpoint'
TFLITE_FILEPATH = OUTPUT_FOLDER + 'tflite/' + NAME_MODEL + ".tflite"





def print_total_trainable_parameters_count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)



def check_memory(model, with_assertion=False):
    
    max_shape = 0
    max_layer = None
    for idx, layer in enumerate(model.layers):
        output_shape = layer.output_shape
        # Encapsulated into list if first layer is Input
        if len(output_shape) == 1 and idx == 0:
            output_shape = output_shape[0]
        num_elems =  np.prod([s for s in output_shape if s is not None])
        if num_elems > max_shape :
            max_shape = num_elems
            max_layer = layer
    
    num_params = model.count_params()
    print('-------------------------------------------')
    print('Max layer shape :', max_shape, ', layer :', max_layer.name, ', shape', max_layer.output_shape)
    print("RAM memory used:", max_shape * 2, ', Max RAM memory :', MAX_RAM_MEMORY)
    print("FLASH memory used:", num_params * 2,', Max FLASH memory :', MAX_FLASH_MEMORY)
    print('-------------------------------------------')
    
    if with_assertion :
        # *2 because represented as 2 bytes
        assert(max_shape * 2) <= MAX_RAM_MEMORY, f"Exceeding RAM memory of {MAX_RAM_MEMORY}, got {max_shape * 2} in layer {idx}: {layer.name}"
        assert(num_params * 2) <= MAX_FLASH_MEMORY, f"Exceeding FLASH memory of {MAX_FLASH_MEMORY}, got {num_params * 2} for the parameters"
    
    
    return (max_shape * 2) <= MAX_RAM_MEMORY and (num_params * 2) <= MAX_FLASH_MEMORY

    