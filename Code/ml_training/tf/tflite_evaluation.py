"""""
 *  \brief     tflite_evaluation.py
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
from tf_to_tflite import get_tflite_output, create_tflite_interpreter, tflite_to_cc
from train_time_serie import load_pretrained_model, get_class_weight
sys.path.insert(0, 'ml_training/')
sys.path.insert(0, 'ml_training/preprocess_data')
# from preprocess_data.prepare_data import get_rain_dataset
import dataloader
import pickle as pkl

already_computed = False

temp_file = 'myfile_non_ver_quantized_.pkl'


def is_correct(labels, outputs):
    pred = outputs.argmax(axis=2)
    return np.equal(labels, pred).astype(int)


# only work for single input
def tflite_eval(tflite_model, X):
    interpreter = tflite_model['interpreter']
    res = []
    for i in range(len(X)):
        if (i + 1)% 100 == 0 :
            print('tflite evaluation :', i + 1, len(X), end="\r")
        
        #TODO : correct with sensor + audio : different inputs, use name of input
        for input in tflite_model['input']:
            interpreter.set_tensor(input['index'], [X[i]])
        interpreter.invoke()
        # iterate over outputs : rain + wind output
        output_sample = []
        for tflite_output in tflite_model['output']:
            output_sample.append(interpreter.get_tensor(tflite_output['index']))
        res.append(output_sample)

    return np.asarray(res)






if __name__ == '__main__':
    # load dataset
    # print(tf.__version__)
    # sys.exit()
    expand_dims = True
    dataset = dataloader.prepare_dataset(TYPE_LABELS, TYPE_INPUTS, SPLIT_FACTOR,
                                        timesteps=None, step_size=None, test_size=0.01, validation_size=0.01,
                                        rain_split_idx=BEST_RAIN_IDX,  wind_split_idx=BEST_WIND_IDX, expand_dims=expand_dims)
    
    num_classes, num_splits, inputs, labels, audio_length, sensor_shape = dataset
    X = inputs[0]['train']['audio']
    X = np.array(X, dtype=np.float32).reshape((-1, 1, audio_length, 1))
    labels = labels[0]['train']

    
    
    # Load TFLite model and allocate tensors.
    # interpreter = tf.lite.Interpreter(model_path=TFLITE_FILEPATH)
    # interpreter.allocate_tensors()
    # # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # tflite_model = {'interpreter': interpreter, 'input':input_details, 'output':output_details}
    # for input in input_details:
    #     print(input['name'])
    
    # for output in output_details:
    #     print(output)
        

    
    with_optim = True
    with_quant = True
    type_label = TYPE_LABELS.value[0]
    tflite_model, debugger = create_tflite_interpreter(QUANT_MODEL_FILEPATH, with_optimization=with_optim, with_quantization=with_quant, 
                                                X=X, labels=labels[type_label], store_path= TFLITE_FILEPATH)
    
    # ip_statistics_dump(f)
    tflite_to_cc(to_module=False)

    num_test_samples = 10000
    X = X[:num_test_samples]
    labels[type_label] = labels[type_label][:num_test_samples]
    # labels['wind'] = labels['wind'][:num_test_samples]
    
    print('evaluating the inputs')
    res = tflite_eval(tflite_model, X)
    # if not already_computed:
    #     res = tflite_eval(tflite_model, X)
    #     with open(temp_file, 'wb') as f:
    #         pkl.dump(res, file=f)
    
    # with open(temp_file, 'rb') as f:
    #     res = pkl.load(file=f)

    res = res.reshape((-1, 3))
    # wind_outputs, rain_outputs  = zip(*list(res))
    
    outputs_dict = {type_label : np.array(res)}#, wind=np.array(wind_outputs))
    for name, output in outputs_dict.items():
        y_pred = output_to_pred(output)
        acc_3, acc_2 = get_results(y_pred, labels, name, "delete_me_please.txt")


    #tf model
    print('========================================')
    print('Comparing with tf model')
    model = load_pretrained_model(dataset)
    inputs = inputs[0]['train']

    inputs['audio'] = inputs['audio'][:num_test_samples]
    outputs = model.predict(inputs, verbose=0)
    if len(model.output_names) == 1:
        outputs_dict = {model.output_names[0] : outputs}
    else :
        outputs_dict = {name: pred for name, pred in zip(model.output_names, outputs)}
        
    for name, output in outputs_dict.items():
        y_pred = output_to_pred(output)
        acc_3, acc_2 = get_results(y_pred, labels, name, "delete_me_im_waiting.txt")
    
    print('done')
    
