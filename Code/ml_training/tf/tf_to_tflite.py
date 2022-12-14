
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



with_optim = True
with_quant = True



def representative_data_gen(X, num_samples=-1, seed=1):
    '''Creates a representative dataset generator to be used in the tflite interpreter creation

    Args:
        X (numpy array): dataset that was used for the machine learning part
        num_samples (int, optional): number of samples returned. Defaults to -1: all the dataset
        seed (int, optional): seed used for shuffling the dataset. Defaults to 1.

    Yields:
        _type_: _description_
    '''
    size = len(X)
    arr = np.arange(size)
    np.random.seed(seed)
    np.random.shuffle(arr)
    for idx in arr[:num_samples]:
        # need reshape (batch=1, size=32000, channel=1)
        yield [np.array(X[idx], dtype=np.float32).reshape((1,-1, 1))]
       

def create_tflite_interpreter(export_dir, with_optimization, with_quantization, store_path=None, X=None):
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
    if with_optimization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if with_quantization:
        converter.representative_dataset = lambda : representative_data_gen(X)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 
    tflite_model = converter.convert()
    if store_path is not None:
        tflite_model_file = pathlib.Path(store_path)
        tflite_model_file.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]['index']
    output_details = interpreter.get_output_details()[0]['index']
    return {'interpreter': interpreter, 'input':input_details, 'output':output_details}


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
    X, labels = dataloader.get_dataset(split_factor=SPLIT_FACTOR, type_labels=TYPE_LABELS)

    tflite_interp_quant = create_tflite_interpreter(MODEL_FILEPATH, with_optimization=with_optim, with_quantization=with_quant, 
                                                X=X, store_path= TFLITE_FILEPATH)

    tflite_to_cc(to_module=False)

    print('done')
