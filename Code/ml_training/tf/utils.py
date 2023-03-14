"""""
 *  \brief     utils.py
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

from keras.callbacks import Callback
import numpy as np
import tensorflow as tf

from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from enum import Enum

import keras.backend as K
from itertools import product

from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

import sys
import optuna


class Labels(Enum):
    RAIN = ['rain']
    WIND = ['wind']
    RAIN_WIND = ['rain', 'wind']
    
class Inputs(Enum):
    AUDIO = ['audio']
    SENSOR = ['sensor']
    AUDIO_SENSOR = ['audio', 'sensor']

def one_hot_labels_to_arr(y_te, y_pred):
    return y_te.argmax(axis=1), y_pred.argmax(axis=1)


def confusion_matrix_2(y_te, y_pred):
    if len(y_te.shape) == 2 :
        y_te, y_pred = one_hot_labels_to_arr(y_te, y_pred)

    elif len(y_te.shape) > 2 :
        raise Exception
    return confusion_matrix(y_te, y_pred)


def smote_resampler(X, labels, minor_upsampling=1, major_downsampling=1, seed=1):
    '''Resample dataset with the SMOTE resampler

    Args:
        X (numpy array): dataset
        labels (numpy 1D array): labels
        minor_upsampling (int, optional): the upsampling factor to resample the minor classes (len * factor). Defaults to 1.
        major_downsampling (int, optional): the downsampling factor of the majority class. Defaults to 1.

    Returns:
        X, labels: dataset resampled
    '''
    counter_labels = Counter(labels)

    major_class, major_count = counter_labels.most_common(1)[0]
    sampling_dict = {major_class : major_count}

    for class_label in counter_labels.keys() :
        if class_label != major_class:
            sampling_dict.update({class_label : min(major_count, int(counter_labels[class_label] * minor_upsampling))})
        
    over = SMOTE(random_state=seed, sampling_strategy=sampling_dict, n_jobs=4)
    under_sampling = sampling_dict.copy()
    under_sampling.update({major_class : int(major_count * major_downsampling)})
    under = RandomUnderSampler(random_state=seed, sampling_strategy=under_sampling)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    original_shape = X[0].shape
    X, labels  = pipeline.fit_resample([x.flatten() for x in X], labels)

    X = np.asarray([x.reshape(original_shape) for x in np.asarray(X)])
    print('Resampled dataset shape %s' % Counter(labels))

    return X, labels


def to_one_hot(labels):
    '''transform 1D vector into matrix of one-hot vectors

    Args:
        labels (numpy array): labels

    Returns:
        numpy array, int : matrix of one-hot vectors, total number of classes
    '''
    labels = np.asarray(labels).reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(labels)
    labels = enc.transform(labels).toarray()
    num_classes = len(enc.categories_[0])
    return labels, num_classes
    


def reformat_dataset(X, labels):
    '''Reorganize dataset to be in right shape for the ML model:
        1. labels in one-hot encoding
        2. X add a dimension 

    Args:
        X (numpy array): dataset
        labels (numpy 1D array): labels
    Returns:
        _type_: X, labels formatted and the number of classes
    '''

    rain_labels, wind_labels = labels
    rain_labels, num_rain_classes = to_one_hot(rain_labels)
    wind_labels, num_wind_classes = to_one_hot(wind_labels)
    labels = np.concatenate((rain_labels, wind_labels), axis=1)
    num_classes = dict(rain=num_rain_classes, wind=num_wind_classes)

    X = np.expand_dims(X, axis=-1)

    return X, labels, num_classes


def separe_labels(labels, num_classes):
    rain_labels = labels[:, :num_classes['rain']]
    wind_labels = labels[:, num_classes['rain']:]
    return rain_labels, wind_labels

def separe_labels_time_series(labels, num_classes):
    rain_labels = []
    wind_labels = []
    for num_splits in range(len(labels)):
        rain_labels.append(labels[num_splits][:, :num_classes['rain']])
        wind_labels.append(labels[num_splits][:, num_classes['rain']:])
    return rain_labels, wind_labels
    
    

def output_to_pred(output):
    output = np.asarray(output)
    return np.asarray(output == output.max(axis=1)[:, None]).astype(int)


def to_binary_labels(y_test, y_pred, threshold):
    y_test, y_pred = one_hot_labels_to_arr(y_test, y_pred)
    y_pred = (y_pred > threshold).astype(np.int16)
    y_test = (y_test > threshold).astype(np.int16)
    
    return y_test, y_pred
    
        
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, labels, output_names):
        self.x_test = x_test
        self.labels = labels
        self.output_names = output_names

    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model.predict(self.x_test, verbose=0)

        if len(self.output_names) == 1:
            outputs_dict = {self.output_names[0] : outputs}
        else :
            outputs_dict = {name: pred for name, pred in zip(self.output_names, outputs)}
        
        for name, y_test in self.labels.items() : 
            print(name, 'confusion matrix')
            print(confusion_matrix(y_test.argmax(axis=1), outputs_dict[name].argmax(axis=1)))
            
        
       

def weighted_categorical_crossentropy(target, output, weights_table):
    weights_vect = weights_table.lookup(K.argmax(target, axis=1))
    return K.categorical_crossentropy(target, output) * weights_vect



def to_lookup_table(dictionnary):
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            list(dictionnary.keys()),
            list(dictionnary.values()),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        ),
        default_value=-1)
    
    
def get_flops(model):
    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        for inp in model.inputs
    ]
    
    graph_info = profile(tf.function(model).get_concrete_function(inputs).graph,
                        options=ProfileOptionBuilder.float_operation())
    flops = graph_info.total_float_ops
    return flops / 1e9


class PrintCallback:
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        print()
        print("Trial number: ",trial.number)
        
        if trial.state == optuna.trial.TrialState.PRUNED:      
            print('Pruned')
        else :
            print("  Value: ", trial.value)
            print("  Params: ", trial.params)

        trial_best = study.best_trial
        print("Best trial:", trial_best.number)
        print("  Value: ", trial_best.value)
        print("  Params: ", trial_best.params)
        print('---------------------------------')
        print()

# def get_flops(model) -> float:
#     """
#     Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
#     in inference mode. It uses tf.compat.v1.profiler under the hood.
#     """

#     from tensorflow.python.framework.convert_to_constants import (
#         convert_variables_to_constants_v2_as_graph,
#     )

#     # Compute FLOPs for one sample
#     batch_size = 1
#     inputs = [
#         tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
#         for inp in model.inputs
#     ]

#     # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
#     real_model = tf.function(model).get_concrete_function(inputs)
#     frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

#     # Calculate FLOPs with tf.profiler
#     run_meta = tf.compat.v1.RunMetadata()
#     opts = (
#         tf.compat.v1.profiler.ProfileOptionBuilder(
#             tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
#         )
#         .with_empty_output()
#         .build()
#     )

#     flops = tf.compat.v1.profiler.profile(
#         graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
#     )
#     # convert to GFLOPs
#     return flops.total_float_ops / 1e9

