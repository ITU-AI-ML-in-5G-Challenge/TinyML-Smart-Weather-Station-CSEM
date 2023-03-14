"""""
 *  \brief     preprocess.py
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
import numpy as np
import pandas as pd
from constants import *
# for resampling only
import torch
from torchaudio.transforms import Resample, MFCC
from scipy.signal import butter, lfilter
from tools import print_progress_bar


def normalize(values, min_val=None, max_val=None):
    '''Normalize the data

    Args:
        values (numpy array): values to be normalized
        min_val (float, optional): min value for the normalization, if values is a subarray. 
                                   Defaults to None: compute min in values
        max_val (float, optional): max value for the normalization, if values is a subarray. 
                                    Defaults to None: compute max in values

    Returns:
        numpy array: normalized data
    '''
    if min_val is None :
        min_val = min(values)
    if max_val is None:
        max_val = max(values)
    return (values - min_val)/(max_val - min_val)


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

        


def preprocess_data(audio_preprocess_kwargs, suffix):
    '''Preprocess the whole data and store the results

    Args:
        audio_preprocess_kwargs (dict): arguments for the audio preprocessing (resampling_freq, lowcut_freq, highcut_freq, mfcc_kwargs)
                                        for more details, see :py:func:create_audio_pipeline
        suffix (str): suffix added to the name when stored, see :py:func:get_result_filename
    '''
    audio_pipeline = create_audio_pipeline(**audio_preprocess_kwargs)

    # FIRST PASS
    print('First pass')
    audio_stream_buffer = dict()
    df_data = []
    for idx, hour in enumerate(INDEX_HOURS):
        print_progress_bar(idx, len(INDEX_HOURS))
        df = pd.read_pickle(get_result_filename(hour))
        df_data.append(df.drop(['timestamp', 'audio'], axis=1))

        # process audio as a single array (the whole hour) 
        audio_stream = np.concatenate(df['audio'].to_numpy())
        for layer_process in audio_pipeline :
            audio_stream = layer_process(audio_stream)

        #TODO store audio_stream: where will loose speed, but after can reuse it, buffer ? Yes, minus : when mfcc
        audio_stream_buffer.update({hour:audio_stream})
        df = None


    data_stats = pd.concat(df_data, ignore_index=True).agg(['min', 'max', 'mean', 'var'])

    # SECOND PASS
    print('Second pass')
    for idx, hour in enumerate(INDEX_HOURS):
        print_progress_bar(idx, len(INDEX_HOURS))
        df = pd.read_pickle(get_result_filename(hour)).drop(['audio'], axis=1)
        # normalize data
        for col in [c for c in df.columns if c not in ['timestamp', 'audio', 'wind_count', 'rain_count']]:
            df[col] = normalize(df[col].to_numpy(), data_stats[col]['min'], data_stats[col]['max'])

        df['audio'] = np.split(audio_stream_buffer[hour], len(df))

        df.to_pickle(get_result_filename(hour, suffix=suffix))
    print('done')




if __name__ == '__main__':

    # true win_length = wind_length * (length numpy // resampling)
    resampling_freq = 8000
    resampling_freq = None
    lowcut_freq = None
    highcut_freq = None
    # win_length = None

    mfcc_kwargs = dict(n_mfcc=128
                        ,melkwargs =  dict(
                            n_fft= 2048 * 4,
                            mel_scale= "htk"),
                            win_length = 256,
                            hop_length = 128,
                            n_mels = 512
                        )
    mfcc_kwargs = None
        

    audio_preprocess_kwargs = dict(resampling_freq=resampling_freq, 
                                    lowcut_freq=lowcut_freq, 
                                    highcut_freq=highcut_freq, mfcc_kwargs=mfcc_kwargs)
    
    preprocess_data(audio_preprocess_kwargs=audio_preprocess_kwargs, suffix="ERASE")




