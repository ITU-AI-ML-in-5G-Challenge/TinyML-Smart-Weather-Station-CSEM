"""""
 *  \brief     transfer_lr_train.py
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

import sys
import os

os.system("module load cuda/11.2")
os.system("module load cudnn/8.1")

from models import *
from utils import *
sys.path.insert(0, 'ml_training/')
sys.path.insert(0, 'ml_training/preprocess_data')
from preprocess_data.prepare_data import get_rain_dataset
import dataloader

import numpy as np
import pandas as pd
from collections import Counter

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score



l2_factor = 0.001
batch_size = 16 * 2
test_size = 0.16
validation_size = 0.16
seed = 24
smote_resampling = False
class_reweighting = True
epochs = 22


if __name__ == '__main__':
    sys.path.insert(1, 'ml_training/')
    tf.config.run_functions_eagerly(False)
    print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    
    args = sys.argv
    if len(args) == 2:
        NAME_MODEL = args[1].lower()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    
    X, labels = get_rain_dataset(compress_labels_rain, SPLIT_FACTOR)


    # sys.exit()

    print('Raw dataset label distribution : %s' % Counter(labels))
    
    if smote_resampling:
        # define pipeline
        X, labels = smote_resampler(X, labels, 1.5, 1, seed)
        
    X, labels, num_classes = reformat_dataset(X, labels) 

    x_tr, x_te, y_tr, y_te = train_test_split(X, labels, test_size=test_size + validation_size, random_state=seed)
    x_val, x_te, y_val, y_te = train_test_split(x_te, y_te, test_size=test_size/(test_size + validation_size), random_state=seed) 

    print('==================================================')
    print(X[0].shape[:-1][0])
    print('x train shape :', x_tr.shape, ', y train shape :', y_tr.shape)
    print('x test shape :', x_te.shape, ', y test shape :', y_te.shape)
    print('x val shape :', x_val.shape, ', y val shape :', y_val.shape)
    print('==================================================')

    d_class_weights = None
    if class_reweighting:
     # for class reweighting
        y_integers = np.argmax(labels, axis=1)
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        d_class_weights = dict(enumerate(class_weights))
    
    print('Model selected:', NAME_MODEL)
    
    if NAME_MODEL != 'mfcc_model':
        model = models_dict['m5'](X[0].shape[:-1][0], num_classes, l2_factor)
    else :
        model = models_dict[NAME_MODEL](X[0].shape, num_classes)

    optimizer = tf.keras.optimizers.Adam(
                            learning_rate=1e-03)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.load_weights(CHECKPOINT_FILEPATH)
    
    # num_layers = len(model.layers)
    # for idx, layer in enumerate(model.layers): 
    #     if idx < num_layers - 40:
    #         layer.trainable = False
            
    
    checkpoint_path_rain = CHECKPOINT_FILEPATH + '_rain'
    
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=4, min_lr=6e-08, verbose=1)

    c_matrix_callback = ConfusionMatrixCallback(x_val, y_val)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_rain,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    callbacks = [reduce_lr, c_matrix_callback, model_checkpoint_callback]
    if USE_TENSORBOARD:
        tb_callback = tf.keras.callbacks.TensorBoard(TF_FOLDER + 'logs', update_freq='epoch')
        callbacks.append(tb_callback)


    model.fit(x=x_tr,
              y=y_tr,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle=True,
              validation_data=(x_val, y_val),
              use_multiprocessing = True
              , callbacks = callbacks
              ,class_weight=d_class_weights)

    model.load_weights(checkpoint_path_rain)


    # Evaluate test set over best model encountered
    loss, acc = model.evaluate(x=x_te, y=y_te, verbose=0)
    output = model.predict(x_te, verbose=0)
    y_pred = output_to_pred(output)
    print(confusion_matrix_2(y_te, y_pred))
    print('Test accuracy: ', accuracy_score(y_te, y_pred))

    # Evaluate test set over best model encountered for binary class
    y_te, y_pred = one_hot_labels_to_arr(y_te, y_pred)
    y_pred = (y_pred > 0).astype(np.int16)
    y_te = (y_te > 0).astype(np.int16)
    print(confusion_matrix_2(y_te, y_pred))
    print('Test accuracy: ', accuracy_score(y_te, y_pred))
    print('Balanced test accuracy: ', balanced_accuracy_score(y_te, y_pred))

    
    
    
    