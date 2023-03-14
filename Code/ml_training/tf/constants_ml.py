"""""
 *  \brief     constants_ml.py
 *  \author    Jonathan Reymond
 *  \version   1.0
 *  \date      2023-02-14
 *  \pre       None
 *  \copyright (c) 2022 CSEM
 *
 *   CSEM S.A.
 *   Jaquet-Droz 1
 *   CH-2000 Neuchâtel
 *   http://www.csem.ch
 *
 *
 *   THIS PROGRAM IS CONFIDENTIAL AND CANNOT BE DISTRIBUTED
 *   WITHOUT THE CSEM PRIOR WRITTEN AGREEMENT.
 *
 *   CSEM is the owner of this source code and is authorised to use, to modify
 *   and to keep confidential all new modifications of this code.
 *
 """

import os
import tensorflow as tf
import numpy as np
import sys
from utils import *
from sklearn.metrics import balanced_accuracy_score
import keras


DATE = 'test'
NAME_AUDIO_MODEL = 'm5' #m_rec
NAME_SENSOR_MODEL = 'm3'
OUTPUT_FOLDER = 'results/outside/' + DATE + '/' + NAME_AUDIO_MODEL + '_new_package/'
USE_TENSORBOARD = True


TYPE_LABELS = Labels.WIND

TYPE_INPUTS = Inputs.AUDIO




NAME_MODEL = 'wind_only'




###########################################
#### Dataset transformation parameters ####
###########################################
SPLIT_FACTOR = 2
LOWPASS_FREQ = None
MERGE_BUCKET_SIZE = 24
RESAMPLE_FREQUENCY = None #SAMPLING_FREQUENCY / 2
TO_MFCC = False

BEST_RAIN_IDX = [2, 4]
BEST_WIND_IDX = [2, 18] 





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
QUANT_MODEL_FILEPATH = TF_FOLDER + NAME_MODEL + '_quant.pb'
CHECKPOINT_FILEPATH = TF_FOLDER + 'checkpoint/' + NAME_MODEL + '_checkpoint'
QUANT_CHECKPOINT_FILEPATH = TF_FOLDER + 'checkpoint/' + NAME_MODEL + '_quant_checkpoint'
TFLITE_FILEPATH = OUTPUT_FOLDER + 'tflite/' + NAME_MODEL + ".tflite"



def print_total_trainable_parameters_count():
    '''Prints the number of trainable parameters of the model after having compiled it
    '''
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


def get_results(output, labels, name_label, outfile=None):
    '''get the accuracy/confusion matrix for a given label

    Args:
        output (2D array): output of model in one-hot encoding
        labels (2D array): corresponding labels in one-hot encoding
        name_label (str): name of label evaluated
        outfile (str, optional): if not None, store the results into the corresponding output file. Defaults to None.

    Returns:
        (float, float): balanced accuracy for original problem, balanced accuracy for 2-classes reduction
    '''
    print(name_label)
    target = labels[name_label]
    pred = output.argmax(axis=1)
    confusion_3 = confusion_matrix(target.argmax(axis=1), pred)
    acc_3 = np.round(balanced_accuracy_score(target.argmax(axis=1), pred) * 100, 3)
    print(confusion_3, '3-class:', acc_3)
    
    target, output = to_binary_labels(target, output, 1)
    acc_2 = np.round(balanced_accuracy_score(target, output) * 100, 3)
    confusion_2 = confusion_matrix_2(target, output)
    print(confusion_2, '2-class:', acc_2)
    
    if outfile :    
        with open(outfile, 'a') as f:
            print(name_label, file=f)
            print(repr(confusion_3), '3-class:', acc_3, file=f)
            print(repr(confusion_2), '2-class:', acc_2, file=f)
            print(file=f)
    return acc_3, acc_2
            
    


    


def check_memory(model, with_assertion=False):
    '''Check the flash and ram memory consumption of the model

    Args:
        model (keras model): evaluated model
        with_assertion (bool, optional): if trigger an assertion if model memory bigger than allowed. Defaults to False.

    Returns:
        boolean: if respects the flash and ram memory consumption limits
    '''
    
    max_shape = 0
    max_layer = None
    for idx, layer in enumerate(model.layers):
        output_shape = layer.output_shape
        # Encapsulated into list if first layer is Input
        if isinstance(layer, keras.layers.InputLayer):
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

    