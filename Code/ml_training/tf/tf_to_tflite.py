"""""
 *  \brief     tf_to_tflite.py
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
import shutil
import pathlib
import numpy as np
import tensorflow as tf
import itertools
from constants_ml import *
sys.path.insert(0, 'ml_training/')
sys.path.insert(0, 'ml_training/preprocess_data')
# from preprocess_data.prepare_data import get_rain_dataset
import dataloader

#Note : currently using the ventilator dataset
from dataloader_vent import extract_dataset

from imblearn.under_sampling import RandomUnderSampler
import copy

with_optim = True
with_quant = True



# def representative_data_gen(X, num_samples=-1, seed=1):
#     '''Creates a representative dataset generator to be used in the tflite interpreter creation

#     Args:
#         X (numpy array): dataset that was used for the machine learning part
#         num_samples (int, optional): number of samples returned. Defaults to -1: all the dataset
#         seed (int, optional): seed used for shuffling the dataset. Defaults to 1.

#     Yields:
#         _type_: _description_
#     '''
#     size = len(X)
#     arr = np.arange(size)
#     np.random.seed(seed)
#     np.random.shuffle(arr)
#     for idx in arr[:num_samples]:
#         # need reshape (batch=1, size=32000, channel=1)
#         yield [np.array(X[idx], dtype=np.float32).reshape((1,-1, 1))]
  
  
def representative_data_gen2(X, labels, num_samples=5000, seed=1):
    '''Creates a balanced representative dataset generator to be used in the tflite interpreter creation.
    Do not support multi-labels

    Args:
        X (np array): dataset that was used for the machine learning part
        labels (np array): labels array
        num_samples (int, optional): number of samples in the dataset taken. Defaults to 5000.
        seed (int, optional): seed for random sampling. Defaults to 1.

    Returns:
        generator: generator of the representative dataset
    '''
    labels = np.argmax(labels, axis=1)
    counter_labels = Counter(labels)
    labels_keys = counter_labels.keys()
    num_classes = len(labels_keys)
    minor_class, minor_count = counter_labels.most_common(None)[-1]
    num_per_class = int(num_samples / num_classes)
    if minor_count < num_per_class:
        print('not enough samples for class', minor_class)
        print('taking', minor_count, 'per class instead of', num_per_class)
        num_per_class = minor_count
    

    sampling_dict = {}
    for class_label in labels_keys:
        sampling_dict.update({class_label : num_per_class}) 

    under_sampler = RandomUnderSampler(random_state=seed, sampling_strategy=sampling_dict)

    # transform the dataset
    original_shape = X[0].shape
    print(original_shape)
    X, labels  = under_sampler.fit_resample([x.flatten() for x in X], labels)

    X = np.asarray([x.reshape(original_shape) for x in np.asarray(X)])
    print('Resampled dataset shape %s' % Counter(labels)) 
    
    return ([np.array(x, dtype=np.float32).reshape((1, 1, -1, 1))] for x in X)


def create_tflite_interpreter(export_dir, with_optimization, with_quantization, store_path=None, X=None, labels=None):
    '''Creates the tflite interpreter from a tensorflow model

    Args:
        export_dir (str): location of the trained tensorflow model
        with_optimization (bool): if use optimization.default when creating the model
        with_quantization (bool): when optimization set to true, will quantize w.r.t. the representative dataset
        store_path (str, optional): Path where to store the .tflite model, if None will not store it. Defaults to None.
        X (numpy array, optional): dataset used for training, to get the representative dataset. Defaults to None.

    Returns:
        dict: the tflite interpreter, as well as the input, output pointers
    '''
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    debugger = None
    if with_optimization:
        print('with weights compression')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    if with_quantization:
        print('with only unit8 computation')
        
        converter.representative_dataset = lambda : representative_data_gen2(X, labels)
        

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 
        
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
        
    tflite_model = converter.convert()
    
    debugger = tf.lite.experimental.QuantizationDebugger(
                    converter=converter, debug_dataset=(lambda : representative_data_gen2(X, labels)))
    if store_path is not None:
        tflite_model_file = pathlib.Path(store_path)
        tflite_model_file.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    #multiple inputs
    input_details = []
    for input in interpreter.get_input_details():  
        input_details.append(input)
    output_details = []
    for output in interpreter.get_output_details():
        output_details.append(output)

    return {'interpreter': interpreter, 'input':input_details, 'output':output_details}, debugger


def get_tflite_output(interpreter_dict, test_sample):
    '''From the tfliter interpreter dict generated from :py:func:create_tflite_interpreter, 
       evaluates the test sample's output of the tflite model

    Args:
        interpreter_dict (dict): the tflite interpreter, as well as the input, output pointers
        test_sample (numpy array): test sample to be evaluated

    Returns:
        numpy array: output of the tflite model
    '''
    interpreter = interpreter_dict['interpreter']
    interpreter.set_tensor(interpreter_dict['input'], test_sample)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter_dict['output'])


def tflite_to_cc(to_module=False):
    '''Transfrom a .tflite file to a .cc + .h file. It assumes that the .tflite is stored in TFLITE_FILEPATH.
        it will store the result in the same folder where the .tflite is located
        
    Args:
        to_module : will store the result also in the u111 folder, and replace it if already exist. The module path
                    is defined by the MODULE_TFLITE_PATH
    '''
    path_cc = TFLITE_FILEPATH.replace(".tflite", "_temp.cc")
    os.system("xxd -i " + TFLITE_FILEPATH + " > " + path_cc)

    old_name_model = TFLITE_FILEPATH.replace('/', '_').replace('.', '_')

    outfile = path_cc.replace("_temp.cc", ".cc")

    #write final .cc file
    with open(path_cc) as fin, open(outfile, "w+") as fout:
        fout.write( '''#include "'''+ NAME_MODEL + '.h' + '''"'''+ '\n\n')
        for line in fin:
            new_line = line
            new_line = new_line.replace(old_name_model, NAME_MODEL)
            new_line = new_line.replace('unsigned char', 'alignas(16) const unsigned char')
            new_line = new_line.replace('unsigned int', 'const unsigned int')
            fout.write(new_line)

    #write final .h file
    outfile_h = outfile.replace(".cc", ".h")
    with open(outfile_h, "w+") as fout:
        fout.write("#include <cstdint>" '\n\n')
        fout.write("extern const unsigned char " + NAME_MODEL + "[];\n")
        fout.write("extern const unsigned int " + NAME_MODEL + "_len;\n")

    #remove temporary file
    os.remove(path_cc)

    #replace files in tflite module
    if to_module: 
        old_model_file = MODULE_TFLITE_PATH + '/' + NAME_MODEL + '.h'
        if os.path.exists(old_model_file):
            os.remove(old_model_file) 
            os.remove(old_model_file.replace(".h", ".cc")) 
        shutil.copy2(outfile_h, MODULE_TFLITE_PATH)
        shutil.copy2(outfile, MODULE_TFLITE_PATH)




if __name__ == '__main__':

    tflite_folder = os.path.dirname(TFLITE_FILEPATH)
    if not os.path.exists(tflite_folder):
        os.makedirs(tflite_folder)
    

    # X, labels = get_rain_dataset(MERGE_BUCKET_SIZE, LOWPASS_FREQ, RESAMPLE_FREQUENCY, TO_MFCC, INDEX_HOURS, compress_labels, SPLIT_FACTOR)
    # X, labels = dataloader.get_dataset(split_factor=SPLIT_FACTOR)
    num_classes, num_splits, inputs, labels, audio_length, sensor_shape = dataloader.prepare_dataset(TYPE_LABELS, TYPE_INPUTS, SPLIT_FACTOR,
                                                                                              timesteps=None, step_size=None, test_size=0.01, validation_size=0.01)
    X = inputs[0]['train']['audio']
    # print(X)
    labels = labels[0]['train']['rain']

    tflite_interp_quant = create_tflite_interpreter(MODEL_FILEPATH, with_optimization=with_optim, with_quantization=with_quant, 
                                                X=X, labels=labels, store_path= TFLITE_FILEPATH)

    tflite_to_cc(to_module=False)

    print('done')
