"""""
 *  \brief     dataloader.py
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
import tensorflow as tf
from scipy.signal import butter, lfilter
import torch
from torchaudio.transforms import Resample, MFCC
import pandas as pd
import numpy as np
import itertools
from utils import Labels

sys.path.insert(0, 'ml_training/')
sys.path.insert(0, 'ml_training/preprocess_data')
from preprocess_data.constants import SAMPLING_FREQUENCY, INDEX_HOURS, get_result_filename
sys.path.insert(0, 'ml_training/tools')
from tools import print_progress_bar
from constants_ml import *
from functools import partial

from sklearn.model_selection import train_test_split





def audio_filter(Wn, btype, sampling_freq, order=5):
    '''Compute an audio filter function

    Args:
        Wn (float or [float, float]): cutoff frequency/ies, see :py:func:butter
        btype (str): type of filter ('lowpass', 'highpass', 'bandpass'), see :py:func:butter
        sampling_freq (float): sampling frequency of the signal
        order (int, optional): order of the filter. Defaults to 5, see :py:func:butter

    Returns:
        funct array -> array : audio filter function
    '''
    if sampling_freq is None:
        sampling_freq = SAMPLING_FREQUENCY
    b, a = butter(order, Wn, fs=sampling_freq, btype=btype, analog=False)
    return lambda x : lfilter(b, a, x)



def create_audio_pipeline(resampling_freq=None, lowcut_freq=None, highcut_freq=None, mfcc_kwargs=None):
    '''Creates an audio pipeline where the audio in form of an array can pass through

    Args:
        resampling_freq (float, optional): resampling frequency. Defaults to None.
        lowcut_freq (float, optional): lowcut frequency. Defaults to None.
        highcut_freq (float, optional): highcut frequency. Defaults to None.
        mfcc_kwargs (dict, optional): mfcc arguments. Defaults to None.

    Returns:
        list[func]: pipeline of functions to apply to the incoming audio signal
    '''
    audio_pipeline = []

    if resampling_freq:
        resampler = Resample(SAMPLING_FREQUENCY, resampling_freq, resampling_method="kaiser_window", lowpass_filter_width=8)
        audio_pipeline.append(lambda x : resampler(torch.FloatTensor(x)).numpy())
    else :
        resampling_freq = SAMPLING_FREQUENCY

    if lowcut_freq and highcut_freq:
        bandpass_filter = audio_filter([lowcut_freq, highcut_freq], 'bandpass', resampling_freq)
        audio_pipeline.append(bandpass_filter)
    elif lowcut_freq:
        audio_pipeline.append(audio_filter(lowcut_freq, 'highpass', resampling_freq))
    elif highcut_freq:
        audio_pipeline.append(audio_filter(highcut_freq, 'lowpass', resampling_freq))
    
    if mfcc_kwargs:
        # TODO : correct, maybe after frame by frame
        mfcc_transform = MFCC(sample_rate = resampling_freq, 
                                n_mfcc = mfcc_kwargs.get('n_mfcc', 40),
                                melkwargs = mfcc_kwargs.get('melkwargs', None))

        audio_pipeline.append(lambda x : mfcc_transform(torch.FloatTensor(x)).numpy())

    return audio_pipeline


   
 
def get_time_series(df, timesteps, step_size):
    '''From the dataset, construct the time-serie samples : each sample has the past defined by timesteps and the step size

    Args:
        df (dataframe): loaded dataset
        timesteps (int): _description_
        step_size (int): _description_

    Returns:
        list: dataset with time-serie
    '''
    cols, names = list(), list()
    # input sequence (t-n*step_size, ... t- step_size, t)
    range_past_sequence = range(timesteps * step_size, -1, - step_size)
    for i in range_past_sequence:
        cols.append(df.shift(i))
        names += [(df.columns[j] + '(t-%d)' % i) for j in range(len(df.columns))]
        
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    agg.dropna(inplace=True)
    
    for col_name in df.columns:
        names = [col_name+'(t-%d)' % i for i in range_past_sequence]
        agg[col_name] = agg[names].values.tolist()
        agg.drop(names, axis=1, inplace=True)

    return agg.apply(lambda x : np.asarray(x.values.tolist()).T,axis=1).to_list()
  


# assume split_idx already sorted
def compress_labels(x, split_indexes):
    '''reduce the original number of classes into the list of splits defining the new range for each class

    Args:
        x (int): original label value
        split_indexes (list[int]): list of splits to define the different levels

    Returns:
        int: new label value
    '''
    idx = 0
    while idx < len(split_indexes):
        if x < split_indexes[idx]:
            return idx
        idx += 1
    return len(split_indexes)
        
    
    
  
def get_dataset(split_factor, timesteps=None, step_size=None):
    '''load dataset, split each sample, and build each sample for the given timesteps and step size

    Args:
        split_factor (int): number of time have to split each sample of originally 1 second
        timesteps (int, optional): number of past sample for each entry for time serie. Defaults to None.
        step_size (int, optional): step size for time series : t, t - step_size, t - 2*step_size, ... . Defaults to None.

    Returns:
        (X, X_series, (label(rain), label()): dataset for audio, time-serie sensor and the labels
    '''
    print('extracting dataset')
    df_list = []
    for idx, hour in enumerate(INDEX_HOURS):
        print_progress_bar(idx, len(INDEX_HOURS))
        df = pd.read_pickle(get_result_filename(hour))
        df_list.append(df)
    df = pd.concat(df_list) 

    X = df['audio'].to_numpy()
    X_list = [np.split(xi, split_factor) for xi in X]
    X = list(itertools.chain.from_iterable(X_list))
    
    X_series = get_time_series(df[['humidity','pressure','temperature']], timesteps, step_size)
    #TODO : implement linear interpolation to not get same values
    X_series = X_series * split_factor
    # remove NaN entries 
    X = X[timesteps * split_factor*step_size:]
    
    def get_labels(label_name, timesteps):
        labels = df[label_name].to_numpy()
        return np.repeat(labels, split_factor)[timesteps*split_factor*step_size:]
    
    return X, X_series, (get_labels('rain_level', timesteps), get_labels('wind_count', timesteps))
   
   
 
def split_chunks(X, X_series, labels, n_splits, gap, test_size, validation_size):
    '''if using time-serie data, split the dataset into multiple non-overlapping chunks of datasets.
    From that, it separe the dataset into test, train and validation sets (should be not overlapping for

    Args:
        X (np array): audio dataset
        X_series (np array): sensor dataset
        labels (array): label
        n_splits (int): number of split to do
        gap (int): _description_
        test_size (float): percentage used for test set
        validation_size (float): percentage used for validation set

    Returns:
        x_tr   : audio train dataset
        x_s_tr  : sensor train dataset
        y_tr    : labels train
        x_te    : audio test dataset
        x_s_te  : sensor test dataset
        y_te    : labels test
        x_val   : audio validation dataset
        x_s_val : sensor validation dataset 
        y_val   : labels validation
    '''
    idx = range(len(labels))
    idx_chunks = np.array_split(idx, n_splits)
    train_idxs = []
    val_idxs = []
    test_idxs = []
    for interval in idx_chunks:
        l = len(interval) - 3 * gap
        tr_split_idx = int(l * (1 - (test_size + validation_size)))
        val_split_idx = int(tr_split_idx + l * validation_size) + gap
        
        train_idxs += list(interval[:tr_split_idx])
        val_idxs += list(interval[tr_split_idx + gap: val_split_idx])
        test_idxs += list(interval[val_split_idx + gap: len(interval) - gap])
        
    x_tr, x_te, x_val       = X[train_idxs], X[test_idxs], X[val_idxs]
    x_s_tr, x_s_te, x_s_val = X_series[train_idxs], X_series[test_idxs], X_series[val_idxs]
    y_tr, y_te, y_val       = labels[train_idxs], labels[test_idxs], labels[val_idxs]
    return x_tr, x_s_tr, y_tr, x_te, x_s_te, y_te, x_val, x_s_val, y_val   
   
 
   
   
def prepare_dataset(type_labels, type_inputs, split_factor, timesteps, step_size, test_size, validation_size,
                    rain_split_idx=None, wind_split_idx=None, expand_dims=False):
    '''load and prepare the dataset and split into test, validation and train sets

    Args:
        type_labels (enum Label): type of labels (rain, wind, ...)
        type_inputs (enum Input): type of inputs (audio, sensor, ...)
        split_factor (int): how much we divide the each 1s sample : if =2, split each sample by two : 0.5s samples
        timesteps (int): timesteps to be done for time series: how much samples from past for each entry.
        step_size (int): step size for time series : t, t - step_size, t - 2*step_size, ... . Defaults to None.
        test_size (float): percentage used for test set
        validation_size (float): percentage used for validation set
        rain_split_idx (list sorted integer, optional): list defining the different classes splits rain. Defaults to None.
        wind_split_idx (list sorted integer, optional): list defining the different classes splits wind. Defaults to None.
        expand_dims (bool, optional): if need to expand the dimension of the audio samples, for QAT. Defaults to False.

    Returns:
        num_classes=dict   : number of different classes for each label
        num_splits=int   : Not Implemented = 0
        inputs_list=list   : list containing num_splits elements where each element is a dict of inputs, where the keys are test, train, val
        labels_list=list   : list containing num_splits elements where each element is a dict of labels, where the keys are test, train, val
        audio_length=int   : audio length
        sensor_shape=tuple   : shape of sensor sample
    '''
    if timesteps is None:
        timesteps = 60
    if step_size is None:
        step_size = 4
    X, X_series, labels = get_dataset(split_factor=split_factor, timesteps=timesteps, step_size=step_size)
    
    X_series = np.asarray(X_series)
    num_total_samples = X_series.shape[0] 
    use_sensor = 'sensor' in type_inputs.value
    num_splits = int(1 /(test_size + validation_size)) if use_sensor else 1
    if wind_split_idx is None:
        wind_split_idx = [6, 12]
        
    if rain_split_idx is None:
        rain_split_idx = [1, 2, 4]
    if wind_split_idx is None:
        wind_split_idx = [2, 16, 24]

    labels_rain = np.vectorize(partial(compress_labels, split_indexes=rain_split_idx))(labels[0])
    labels_wind = np.vectorize(partial(compress_labels, split_indexes=wind_split_idx))(labels[1])
    labels = labels_rain, labels_wind
    
    # if smote_resampling:
    #     # define pipeline
    #     X, labels = smote_resampler(X, labels, 1.5, 1, seed)
        
    X, labels, num_classes = reformat_dataset(X, labels) 
    
    
    if use_sensor:
        print('using sensor input ...')
        x_tr, x_s_tr, y_tr, x_te, x_s_te, y_te, x_val, x_s_val, y_val = split_chunks(X, X_series, labels, n_splits=40, gap=timesteps*step_size,
                                                                                     test_size=test_size, validation_size=validation_size)   
    else: 
        seed = 1
        x_tr,  x_te, x_s_tr,  x_s_te, y_tr,  y_te = train_test_split(X, X_series, labels, test_size=test_size + validation_size, random_state=seed)
        x_val, x_te, x_s_val, x_s_te, y_val, y_te = train_test_split(x_te, x_s_te,  y_te, test_size=test_size/(test_size + validation_size), random_state=seed)
    
    
    if expand_dims:
        print(x_tr.shape)
        x_tr = np.expand_dims(x_tr, axis=1)
        x_te = np.expand_dims(x_te, axis=1)
        x_val = np.expand_dims(x_val, axis=1)
        print(x_tr.shape)
        # sys.exit()
        
    # TODO : to be compatible with sensor that trains multiple times, not implemented
    x_tr, x_s_tr, y_tr = [x_tr], [x_s_tr], [y_tr]
    x_te, x_s_te, y_te = [x_te], [x_s_te], [y_te]
    x_val, x_s_val, y_val = [x_val], [x_s_val], [y_val]

    y_tr_rain, y_tr_wind = separe_labels_time_series(y_tr, num_classes)
    y_te_rain, y_te_wind = separe_labels_time_series(y_te, num_classes)
    y_val_rain, y_val_wind = separe_labels_time_series(y_val, num_classes)
    
    audio_length = X[0].shape[:-1][0]

    sensor_shape = X_series.shape[1:]
    
    audio = dict(train=x_tr, val=x_val, test=x_te)
    sensor = dict(train=x_s_tr, val=x_s_val, test=x_s_te)
    rain = dict(train=y_tr_rain, val=y_val_rain, test=y_te_rain)
    wind = dict(train=y_tr_wind, val=y_val_wind, test=y_te_wind)

    inputs_list, labels_list = [], []
    # depends on time-serie method
    for num_split in range(1) : 
        labels = dict()
        inputs = dict()
        for k in ['train', 'val', 'test']:
            # labels
            labels[k] = dict()
            for name_label in type_labels.value:
                if name_label == 'rain':
                    labels[k]['rain'] = rain[k][num_split]
                else :
                    labels[k]['wind'] = wind[k][num_split]
            # inputs
            inputs[k] = dict()
            for name_inputs in type_inputs.value:
                if name_inputs == 'audio':
                    inputs[k]['audio'] = audio[k][num_split]
                else:
                    inputs[k]['sensor'] = sensor[k][num_split]
        inputs_list.append(inputs)
        labels_list.append(labels)
           
    if type_labels is Labels.RAIN:
        num_classes = dict(rain=num_classes['rain'])
    elif type_labels is Labels.WIND:
        num_classes = dict(wind=num_classes['wind'])    
    
    return num_classes, num_splits, inputs_list, labels_list, audio_length, sensor_shape    
      
   

     
def explode_dataset():
    '''store the dataset into a file for each sample

    Raises:
        NotImplementedError: not implemented
    '''
    print('extracting dataset')

    for idx, hour in enumerate(INDEX_HOURS):
        print_progress_bar(idx, len(INDEX_HOURS))
        df = pd.read_pickle(get_result_filename(hour))
        # TODO
        raise NotImplementedError  
    
    