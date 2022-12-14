
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


class Labels(Enum):
    RAIN = 0
    WIND = 1
    RAIN_WIND = 2

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
    


def reformat_dataset(X, labels, type_labels):
    '''Reorganize dataset to be in right shape for the ML model:
        1. labels in one-hot encoding
        2. X add a dimension 

    Args:
        X (numpy array): dataset
        labels (numpy 1D array): labels
    Returns:
        _type_: X, labels formatted and the number of classes
    '''
    if type_labels == Labels.RAIN_WIND:
        rain_labels, wind_labels = labels
        rain_labels, num_rain_classes = to_one_hot(rain_labels)
        wind_labels, num_wind_classes = to_one_hot(wind_labels)
        labels = np.concatenate((rain_labels, wind_labels), axis=1)
        num_classes = [num_rain_classes, num_wind_classes]
    else :
        labels, num_classes = to_one_hot(labels)
        num_classes = [num_classes]

    X = np.expand_dims(X, axis=-1)

    return X, labels, num_classes


def separe_labels(labels, num_classes):
    rain_labels = labels[:, :num_classes[0]]
    wind_labels = labels[:, num_classes[0]:]
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
    def __init__(self, x_test, y_test_rain, y_test_wind):
        self.x_test = x_test
        self.y_test_rain = y_test_rain
        self.y_test_wind = y_test_wind

    def on_epoch_end(self, epoch, logs=None):
        # only for sequential model
        output = self.model.predict(self.x_test, verbose=0)
        
        y_pred_rain = output_to_pred(output[0])
        print("Rain confusion matrix")
        print(confusion_matrix(self.y_test_rain.argmax(axis=1), y_pred_rain.argmax(axis=1)))
        
        y_pred_wind = output_to_pred(output[1])
        print("Wind confusion matrix")
        print(confusion_matrix(self.y_test_wind.argmax(axis=1), y_pred_wind.argmax(axis=1)))