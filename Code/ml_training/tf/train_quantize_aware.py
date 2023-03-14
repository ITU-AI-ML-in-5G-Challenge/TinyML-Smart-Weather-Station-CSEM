"""""
 *  \brief     train_quantize_aware.py
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

import tensorflow_model_optimization as tfmot
import sys
sys.path.insert(0, '/local/user/jrn/tinyml-challenge-2022/ml_training')
sys.path.insert(0, '/local/user/jrn/tinyml-challenge-2022/ml_training/preprocess_data')
from preprocess_data.prepare_data import get_final_dataframe
import numpy as np
import pandas as pd

from tf_to_tflite import *
from tensorflow import keras
from sklearn.model_selection import train_test_split
from constants_ml import *
from sklearn.metrics import classification_report, confusion_matrix
from models import models_dict
from utils import *
from train_time_serie import load_pretrained_model

from keras.callbacks import ReduceLROnPlateau

                  

epochs = 30
batch_size = 16 * 2 * 2
test_size = 0.15
validation_size = 0.15
seed = 24



if __name__ == '__main__':
    num_split = 0
    expand_dims = True
    dataset = dataloader.prepare_dataset(TYPE_LABELS, TYPE_INPUTS, SPLIT_FACTOR,
                                        None, None, test_size, validation_size,
                                        BEST_RAIN_IDX, BEST_WIND_IDX, expand_dims)
    
    num_classes, num_splits, inputs, labels, audio_length, sensor_shape = dataset
    
    type_label = TYPE_LABELS.value[0]
    for type_ in ['val', 'train', 'test']:
        labels[num_split][type_]['quant_' + type_label] = labels[num_split][type_][type_label]
        labels[num_split][type_].pop(type_label, None)
    
    model, compile_param = load_pretrained_model(dataset, return_compile_param=True, quant_name=True, learning_rate=1e-04)
    compile_param['optimizer'] = tf.keras.optimizers.Adam(1e-04)
    model = tfmot.quantization.keras.quantize_model(model)
    
    
    model.compile(**compile_param)
 
    


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=6e-08, verbose=1)

    c_matrix_callback = ConfusionMatrixCallback(inputs[num_split]['val'], labels[num_split]['val'], model.output_names)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=QUANT_CHECKPOINT_FILEPATH,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

    callbacks = [reduce_lr, c_matrix_callback, model_checkpoint_callback]
    if USE_TENSORBOARD:
        tb_callback = tf.keras.callbacks.TensorBoard(TF_FOLDER + 'logs', histogram_freq = 1, update_freq='epoch')
        callbacks.append(tb_callback)


    

    model.fit(x=inputs[num_split]['train'],
            y=labels[num_split]['train'],
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            shuffle=True,
            validation_data=(inputs[num_split]['val'], labels[num_split]['val']),
            use_multiprocessing = True
            , callbacks = callbacks)


    print("=============================")
    print('Finished training, evaluation')
    print("=============================")
    model.load_weights(QUANT_CHECKPOINT_FILEPATH).expect_partial()
    print()
    outputs = model.predict(inputs[num_split]['test'], verbose=0)
    # outputs_dict = {name: pred for name, pred in zip(model.output_names, outputs)}
    if len(model.output_names) == 1:
        outputs_dict = {model.output_names[0] : outputs}
    else :
        outputs_dict = {name: pred for name, pred in zip(model.output_names, outputs)}
        
    for name, output in outputs_dict.items():
        y_pred = output_to_pred(output)
        acc_3, acc_2 = get_results(y_pred, labels[num_split]['test'], name, outfile="bobby_dooby.txt")
        
    tf.saved_model.save(model, QUANT_MODEL_FILEPATH)

