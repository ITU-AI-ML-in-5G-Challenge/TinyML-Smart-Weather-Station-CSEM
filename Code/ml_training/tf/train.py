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




smote_resampling = False
class_reweighting = False


batch_size = 16 * 2
test_size = 0.16
validation_size = 0.16
seed = 24
epochs = 35

def get_class_weight(labels):
    y_integers = np.argmax(labels, axis=1)
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
    return dict(enumerate(class_weights))
    



if __name__ == '__main__':
    sys.path.insert(1, 'ml_training/')
    tf.config.run_functions_eagerly(False)
    print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    
    args = sys.argv
    if len(args) == 2:
        NAME_MODEL = args[1].lower()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    
    X, labels = dataloader.get_dataset(split_factor=SPLIT_FACTOR, type_labels=TYPE_LABELS)
    labels = compress_labels(labels)
  
    
    if smote_resampling:
        # define pipeline
        X, labels = smote_resampler(X, labels, 1.5, 1, seed)
        
    X, labels, num_classes = reformat_dataset(X, labels, type_labels=TYPE_LABELS) 
    
    x_tr, x_te, y_tr, y_te = train_test_split(X, labels, test_size=test_size + validation_size, random_state=seed)
    x_val, x_te, y_val, y_te = train_test_split(x_te, y_te, test_size=test_size/(test_size + validation_size), random_state=seed) 

    print('==================================================')
    print(X[0].shape[:-1][0])
    print('x train shape :', x_tr.shape, ', y train shape :', y_tr.shape)
    print('x test shape :', x_te.shape, ', y test shape :', y_te.shape)
    print('x val shape :', x_val.shape, ', y val shape :', y_val.shape)
    print('==================================================')


    y_tr_rain, y_tr_wind = separe_labels(y_tr, num_classes)
    y_te_rain, y_te_wind = separe_labels(y_te, num_classes)
    y_val_rain, y_val_wind = separe_labels(y_val, num_classes)
    
    class_weights = None
    if class_reweighting:
        class_weights_rain = get_class_weight(y_tr_rain)
        class_weights_wind = get_class_weight(y_tr_wind)
        class_weights = {'rain' : class_weights_rain,
                         'wind' : class_weights_wind}
        



    print('Model selected:', NAME_MODEL)
    
    model = models_dict['final'](X[0].shape[:-1][0], *num_classes)

    optimizer = tf.keras.optimizers.Adam(
                            learning_rate=1e-03)

    model.compile(optimizer=optimizer,
                  loss={'rain':'categorical_crossentropy', 'wind':'categorical_crossentropy'},
                  metrics={'rain':'categorical_accuracy', 'wind':'categorical_accuracy'})

    print(model.summary())

    check_memory(model, with_assertion=True)

    # Define callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=6e-08, verbose=1)

    c_matrix_callback = ConfusionMatrixCallback(x_val, y_val_rain, y_val_wind)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

    callbacks = [reduce_lr, c_matrix_callback, model_checkpoint_callback]
    if USE_TENSORBOARD:
        tb_callback = tf.keras.callbacks.TensorBoard(TF_FOLDER + 'logs', update_freq='epoch')
        callbacks.append(tb_callback)


    model.fit(x=x_tr,
              y=(y_tr_rain, y_tr_wind),
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              shuffle=True,
              validation_data=(x_val, (y_val_rain, y_val_wind)),
              use_multiprocessing = True
              , callbacks = callbacks
              ,class_weight=class_weights)


    print("=============================")
    print('Finished training, evaluation')
    print("=============================")
    model.load_weights(CHECKPOINT_FILEPATH)


    print()
    output = model.predict(x_te, verbose=0)

    y_pred_rain = output_to_pred(output[0])
    
    print("Rain confusion matrix")
    print(confusion_matrix(y_te_rain.argmax(axis=1), y_pred_rain.argmax(axis=1)))
    print('Balanced rain test accuracy: ', balanced_accuracy_score(y_te_rain.argmax(axis=1), y_pred_rain.argmax(axis=1)))
        
    y_pred_wind = output_to_pred(output[1])
    print("Wind confusion matrix")
    print(confusion_matrix(y_te_wind.argmax(axis=1), y_pred_wind.argmax(axis=1)))
    print('Balanced wind test accuracy: ', balanced_accuracy_score(y_te_wind.argmax(axis=1), y_pred_wind.argmax(axis=1)))
    
    
    print()
    
    y_test_rain, y_pred_rain = to_binary_labels(y_te_rain, y_pred_rain, 1)
    print('Rain binary confusion matrix')
    print(confusion_matrix_2(y_test_rain, y_pred_rain))
    print('Balanced rain test accuracy: ', balanced_accuracy_score(y_test_rain, y_pred_rain))
    
    y_test_wind, y_pred_wind = to_binary_labels(y_te_wind, y_pred_wind, 1)
    print('Wind binary confusion matrix')
    print(confusion_matrix_2(y_test_wind, y_pred_wind))
    print('Balanced wind test accuracy: ', balanced_accuracy_score(y_test_wind, y_pred_wind))

    print()
    tf.saved_model.save(model, MODEL_FILEPATH)


