"""""
 *  \brief     Python script used for ML training
 *  \author    Jonathan Reymond, Robin Berguerand, Jona Beysens
 *  \version   1.0
 *  \date      2022-11-14
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
import numpy as np
import pickle
import wave
import torch
import matplotlib.pyplot as plt
import librosa


def print_progress_bar(idx, length):
    perc = idx / length
    num_completed = int(perc * 35)
    sys.stdout.write('\r')
    s = '[' + '='*num_completed + '>' +' '*(35 - num_completed - 1) + ']  ' 
    s += str(round((perc + 1/length) * 100.0, 2)) + '%' + '   ' + str(idx+1) + '/' + str(length) 
    if idx == length -1:
        s += '\n'
    sys.stdout.write(s)
    sys.stdout.flush()


def index_to_time_array(index, fs):
    '''Get the time in seconds of index in the numpy audio recording at a given sampling frequency

    Args:
        index (int): targeted index
        fs (int): sampling frequency

    Returns:
        float: time in seconds of index
    '''
    return 1/fs * index


def time_to_index_array(t, fs):
    """from time, find index in the numpy audio recording (round nearest left)

    Args:
        t (float): time in sec
        fs (float): sampling frequency

    Returns:
        int: index of sample
    """
    return int(t * fs)



def index_to_time_entry(index):
    '''Get the minutes and the second time from the index of the entry approx

    Args:
        index (int): index of entry

    Returns:
        (int, int): (minutes, seconds)
    '''

    res =  index
    res_min = res // 60
    res_sec = res % 60
    return res_min, res_sec

def time_to_index_entry(min, sec):
    '''Get the index of the entry given the minutes and seconds

    Args:
        min (int): minutes
        sec (int): seconds

    Returns:
        int: index
    '''
    if min == -1 or sec == -1 :
        return -1
    else :
        return min * 60 + sec


# From micro:bit function, last factor to convert mph -> kmh
def wind_count_to_kmh(wind_count):
    '''get the speed of the wind given the number of counts of the anemometer in a time slot of 1 second
        (time slot length defined by the micro:bit program).
        Example :
            wind_count :    np.array([0, 2,           4,          6,          8,          12,          14 ])
            output in kmh : np.array([0, 2.15729759,  4.31459517, 6.47189276, 8.62919035, 12.94378552, 15.10108311])

    Args:
        wind_count (int or array of ints): number of interruptions of the anemometer during the given timeslot

    Returns:
        float or array of float: speed in kmh of the given timeslot
    '''
    return wind_count / 2 * 1000 / 1492 * 2 * 1.609344


# From micro:bit function, last factor to convert inches -> mm
def rain_count_to_mm(rain_count):
    '''get the amount of rain fallen in mm given the number of interrupts of the rain sensor
        Example :
            rain_count : np.array([0, 2,        4,       6,      8 ])
            rain in mm : np.array([0, 0.5588,   1.1176,  1.6764, 2.2352])

    Args:
        rain_count (int of array of ints): number of interruptions of the rain sensor

    Returns:
        float or array of float t: mm of rain fallen
    '''
    return rain_count * 11 / 1000 * 25.4


def get_date_after(start_date, time_elapsed):
    start = np.datetime64(start_date)
    elapsed =  np.timedelta64(time_elapsed, 'ms')
    print(start + elapsed)

##############################################
######### Signal processing functions ########
##############################################

def spectrum(x, sf, fmax=None, dB=False):
    if fmax is None or fmax > sf / 2:
        fmax = sf / 2
    N = int(len(x) * fmax / sf)
    X = np.abs(np.fft.fft(x)[0:N])
    if dB:
        X = 20 * np.log10(X)
    return X, N, fmax

def moving_avg(x, N):
    x = np.r_[np.zeros(N), x]
    xc = np.cumsum(x)
    return (xc[N:] - xc[:-N]) / N

def leaky_integrator(x, lambda_):
    result = np.zeros(x.size)
    y = 0
    for i in range(x.size):
        result[i] = (1- lambda_)*x[i] + lambda_ * y
        y = result[i]
    return result

def rms(x, type_, param):
    if type_ == 'moving_avg':
        return np.sqrt(moving_avg(x**2, param))
    else:
        return np.sqrt(leaky_integrator(x**2, param))




##############################################
###### Signal processing plot functions ######
##############################################

def plot_time_domain(name, audio_data, audio_samplerate, ax):
    ax.plot(np.linspace(0, len(audio_data) / audio_samplerate, num=len(audio_data)), audio_data)


#For pytorch prin spectrogram : https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#mfcc
def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    specgram = torch.FloatTensor(specgram)
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_mel_spectrogram(name, audio_data_float, audio_samplerate, ax):
    S = librosa.feature.melspectrogram(y=audio_data_float, sr=audio_samplerate, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(log_S, sr=audio_samplerate, x_axis='time', y_axis='mel', ax=ax)


def plot_mfcc(name, audio_data_float, audio_samplerate, ax):
    mfcc = get_mfcc(audio_data_float,audio_samplerate)
    librosa.display.specshow(mfcc, sr=audio_samplerate, x_axis='time', ax=ax)


def plot_delta_mfcc(name, audio_data_float, audio_samplerate, ax):
    mfcc = get_mfcc(audio_data_float,audio_samplerate)
    delta_mfcc  = librosa.feature.delta(mfcc)
    librosa.display.specshow(delta_mfcc, sr=audio_samplerate, x_axis='time', ax=ax)


def plot_delta2_mfcc(name, audio_data_float, audio_samplerate, ax):
    mfcc = get_mfcc(audio_data_float,audio_samplerate)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    librosa.display.specshow(delta2_mfcc, sr=audio_samplerate, x_axis='time', ax=ax)


def get_mfcc(audio_data_float,audio_samplerate):
    S = librosa.feature.melspectrogram(y=audio_data_float, sr=audio_samplerate, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    return librosa.feature.mfcc(S=log_S, n_mfcc=13)


#from epfl course https://github.com/prandoni/COM418/blob/main/AudioMetrics/AudioMetrics.ipynb
def plot_metrics(x, axis=None, title=""):
    n_max, n_min = np.argmax(x), np.argmin(x)
    n_peak = n_max if np.abs(x[n_max]) > np.abs(x[n_min]) else n_min
    dc, rms = np.mean(x), np.sqrt(np.mean(x ** 2))
    m = {
        "DC": dc, 
        "peak_location": n_peak,
        "peak": x[n_peak],
        "p-p": x[n_max] - x[n_min],
        "RMS": rms
    }
    if axis is not None:
        axis.plot(x, color='lightgray', linewidth=0.5)
        axis.axhline(y=dc, color='C2', label=f"DC: {dc:.3f}")    
        axis.axhline(y=rms, color='C3', label=f"RMS: {rms:.3f}")    
        axis.plot((n_peak, n_peak), (dc, x[n_peak]), color='C0', label=f"peak = {x[n_peak]:.3f}")
        axis.axhline(y=x[n_max], color='C4') 
        axis.axhline(y=x[n_min], color='C4')
        axis.plot((0, 0), (x[n_max], x[n_min]), color='C4', label=f"p-p = {(x[n_max] - x[n_min]):.3f}")
        axis.plot(0, 0, linewidth=0, label=f'Crest: {np.abs(x[n_peak])/rms:.2f}')
        axis.plot(0, 0, linewidth=0, label=f'PAPR: {20 * np.log10(np.abs(x[n_peak])/rms):.2f} dB')
        axis.grid()
        axis.legend(loc='lower right')
        axis.set_title(title)
        axis.set_xlabel('time')
        
    return m

