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




# TODO : temporary, remove after store each entry separately 
def get_dataset(split_factor, type_labels):
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
    
    def get_labels(label_name):
        labels = df[label_name].to_numpy()
        return np.repeat(labels, split_factor)
    
    if type_labels == Labels.RAIN:
        return X, get_labels('rain_level')
    elif type_labels == Labels.WIND:
        return X, get_labels('wind_count')
    else :
        return X, (get_labels('rain_level'), get_labels('wind_count'))
   
    

        
    