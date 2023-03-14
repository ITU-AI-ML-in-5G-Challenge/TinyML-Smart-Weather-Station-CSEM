"""""
 *  \brief     prepare_data.py
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
import sys
import os
import pickle
from constants import *
# for resampling only
import torch
import itertools

from torchaudio.transforms import Resample, MFCC
from statsmodels.stats import diagnostic
from scipy import stats
from scipy.signal import butter, lfilter
from itertools import chain




#could split each to smaller, need have index, how to have a dataloader in tensorflow?
#could do here the resampling and defining the AUDIO_LENGTH
#Merging :

def concat_all_dataframes(store=False):
    df_list = []
    for hour, df in dfs_iterator():
        df_list.append(df)
    df_result = pd.concat(df_list)
    if store:
        df_result.to_pickle(get_result_filename())
    return df_result


def merge_audio(x):
    return np.array(list(chain.from_iterable(x)))


def merge_samples(merged_df, bucket_size):
    if bucket_size == 1 :
        return merged_df
    print('Initial length :', len(merged_df))
    rest = len(merged_df) % bucket_size
    if rest != 0:
        print('dropping last samples to get exact number bucket size, rest:', rest)
        merged_df = merged_df.iloc[:-rest].copy()
        print('Final length after removing last:', len(merged_df))

    index_group = np.repeat(np.arange((len(merged_df)) // bucket_size), bucket_size)
    merged_df['group'] = index_group.astype(int)

    aggregate_dict = {name : 'mean' for name in list(merged_df.columns) if name not in ['wind_dir', 'audio']}
    aggregate_dict.update({'audio' : merge_audio})

    result_df = merged_df.drop('wind_dir', axis=1).groupby('group').agg(aggregate_dict)
    result_df.drop('group', axis=1, inplace=True)
    
    # recover wind direction
    angles_rad = np.arctan2(result_df['wind_y'], result_df['wind_x'])
    result_df['wind_dir'] = np.round(angles_rad / RAD_ANGLE) % len(WIND_DIRECTIONS)

    a = np.apply_along_axis(lambda x : len(x), 1, result_df['audio'].to_list())
    assert len(a[a != bucket_size * SAMPLING_FREQUENCY]) == 0, "Exists samples with length different from sampling frequency"

    result_df.index.names = [None]
    print('Final length after merging:', len(result_df))
    return result_df.sort_values(by=['timestamp'])


def resample(df, new_sf):
    if new_sf == SAMPLING_FREQUENCY:
        print('no resampling needed')
        return df

    resampler = Resample(SAMPLING_FREQUENCY, new_sf, resampling_method="kaiser_window", lowpass_filter_width=8)
    df['audio'] = df['audio'].map(lambda x : resampler(torch.FloatTensor(x)).numpy())

    a = np.apply_along_axis(lambda x : len(x), 1, df['audio'].to_list())
    # assert len(a[a != MERGE_BUCKET_SIZE * new_sf]) == 0
    return df


def transform_to_mfcc(df, resample_frequency=None):
    if resample_frequency is None :
        resample_frequency = SAMPLING_FREQUENCY
    n_fft = 2048 * 2
    # win_length = None
    hop_length = 256
    n_mels = 220
    n_mfcc = 220

    mfcc_transform = MFCC(
        sample_rate = resample_frequency,
        n_mfcc = n_mfcc,
        melkwargs = {
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },)
    df['audio'] = df['audio'].map(lambda x : mfcc_transform(torch.FloatTensor(x)).numpy())
    return df


def normalize(values, min_val=None, max_val=None):
    if min_val is None :
        min_val = min(values)
    if max_val is None:
        max_val = max(values)
    return (values - min_val)/(max_val - min_val)



# def outlier_bounds_IQR(arr, alpha):
#    df = pd.DataFrame(arr, columns=['audio'])
#    q1 = df.quantile(0.25)
#    q3 = df.quantile(0.75)
#    IQR = q3 - q1
#    low = (q1 - alpha * IQR).item()
#    high = (q3 + alpha * IQR).item()
#    num_under = len(arr[arr < low])
#    num_upper = len(arr[arr > high])
#    print("Number samples under the low bound: ", num_under, ", percentage :", num_under / len(arr) * 100)
#    print("Number samples upper the high bound: ", num_upper, ", percentage :", num_upper / len(arr) * 100)
#    return low, high


# def process_audio(audio, alpha=1.5):
#     audio_stream = np.array(list(chain.from_iterable(audio)))
#     # sanity check
#     a = np.apply_along_axis(lambda x : len(x), 1, audio)
#     assert len(a[a != SAMPLING_FREQUENCY]) == 0, "Exists samples with length different from sampling frequency"

#     low, high = outlier_bounds_IQR(audio_stream, alpha)
#     audio_stream = np.clip(audio_stream, low, high)
#     audio_stream = normalize(audio_stream)

#     return np.split(audio_stream, len(audio_stream) // SAMPLING_FREQUENCY)


def butter_lowpass(cutoff_freq, order=5):
    return butter(order, cutoff_freq, fs=SAMPLING_FREQUENCY, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff_freq, order=5):
    b, a = butter_lowpass(cutoff_freq, order=order)
    y = lfilter(b, a, data)
    return y

def process_audio(audio, lowpass_freq=None):
    audio_stream = np.array(list(chain.from_iterable(audio)))
    # sanity check
    a = np.apply_along_axis(lambda x : len(x), 1, audio)
    assert len(a[a != SAMPLING_FREQUENCY]) == 0, "Exists samples with length different from sampling frequency"

    if lowpass_freq is not None:
        audio_stream = butter_lowpass_filter(audio_stream, cutoff_freq=lowpass_freq, order=6)
    audio_stream = np.float32(audio_stream)
    return np.split(audio_stream, len(audio_stream) // SAMPLING_FREQUENCY)


def get_audio_generator():
    for hour in range(MIC_NUM_HOURS):
        yield merge_audio(pd.read_pickle(get_result_filename(hour))['audio'].values)


def get_dataframe_without_audio(max_num_hours=None):
    if max_num_hours is None:
        max_num_hours = MIC_NUM_HOURS
    dfs_list = []
    for hour in INDEX_HOURS:
        merged_df = pd.read_pickle(get_result_filename(hour))
        dfs_list.append(merged_df.drop(['timestamp', 'audio'], axis=1).copy())
        merged_df = None

    return pd.concat(dfs_list, ignore_index=True)


def print_distribution_test(name, values, dist):
    _, p_val = diagnostic.kstest_normal(values, dist = dist)   
    if p_val < 0.05:
        print(name,': p-value :',p_val,"reject null hypothesis : doesn't come from", dist, "distribution" )
    else :
        print(name,': p-value :',p_val,"not reject null hypothesis : could come from", dist, "distribution" )


def print_spearman_correlation(test_name, test_values,target_name, target_values):
    result = stats.spearmanr(test_values, target_values)
    p_val = result.pvalue
    if p_val < 0.05:
        print(test_name, '-', target_name, ':significant correlation: p-val =', p_val, ', spearman correlation =', result.correlation)
    else :
        print(test_name, '-', target_name, ':not so significant correlation: p-val =', p_val, ', spearman correlation =', result.correlation)
        


def get_stats(df_without_audio, need_print=False, cols_interest=['humidity', 'pressure', 'temperature']):
    if need_print:
        # test distribution for eventual normalization 
        print("-----------------------------------------------")
        print("test if given column follow normal distribution")
        print("-----------------------------------------------")
        for col in list(df_without_audio.columns):
            print_distribution_test(col, df_without_audio[col], 'norm')
        print("----------------------------------------------------")
        print("test if given column follow exponential distribution")
        print("----------------------------------------------------")
        for col in list(df_without_audio.columns):
            print_distribution_test(col, df_without_audio[col], 'exp')

        # rain/wind correlation measure
        print("------------------------------")
        print("rain/wind correlation measure")
        print("------------------------------")
        for col in cols_interest:
            print_spearman_correlation('wind_count', df_without_audio['wind_count'], col, df_without_audio[col])
        for col in cols_interest:
            print_spearman_correlation('rain_count', df_without_audio['rain_count'], col, df_without_audio[col])

    stats = df_without_audio.agg(['min', 'max', 'mean', 'var'])
    if need_print:
        print("----------------------")
        print("Basic statistics resum")
        print("----------------------")
        print(stats)
    return stats


def dfs_iterator():
     for hour in INDEX_HOURS:
        yield hour, pd.read_pickle(get_result_filename(hour))



def process_data(stats_res, lowpass_freq, df):
    df['audio'] = process_audio(df['audio'].tolist(), lowpass_freq)
    for col in [c for c in df.columns if c not in ['timestamp', 'audio', 'wind_count', 'rain_count']]:
        df[col] = normalize(df[col].to_numpy(), stats_res[col]['min'], stats_res[col]['max'])

    return df



def get_suffix_name(lowpass_freq, merge_bucket_size, resample_frequency, to_mfcc):
    suffix = '_lowfreq_' + str(lowpass_freq)
    suffix +='_b_size_' + str(merge_bucket_size)
    resample_frequency = resample_frequency if resample_frequency is not None else SAMPLING_FREQUENCY
    suffix += '_freq_' + str(resample_frequency)
    is_mfcc = '_mfcc' if to_mfcc else ''
    return suffix + is_mfcc



def get_final_dataframe(merge_bucket_size, lowpass_freq=None, resample_frequency=None, to_mfcc=False):
    suffix_name = get_suffix_name(lowpass_freq, merge_bucket_size, resample_frequency, to_mfcc) 

    if not os.path.isfile(get_result_filename(suffix=suffix_name)):
        print('Dataset not previously computed, computing...')
        
        df = None
        ### STEP 1 ###
        suffix = ''
        print('Step 1 : get concatenated dataframe')
        if not os.path.isfile(get_result_filename(suffix=suffix)):
            print('         not done, computing...')
            df_w = get_dataframe_without_audio()
            df = concat_all_dataframes()
            assert df['audio'].iloc[0].dtype == np.float32, 'wrong audio type '
            print(get_result_filename(suffix=suffix))
            df.to_pickle(get_result_filename(suffix=suffix))
        
        ### STEP 2 ###
        old_suffix = suffix
        suffix = old_suffix + '_lowfreq_' + str(lowpass_freq)
        print('Step 2 : audio preprocessing + lowpass')
        if not os.path.isfile(get_result_filename(suffix=suffix)):
            print('         not done, computing...')
            df = df if (df is not None) else pd.read_pickle(get_result_filename(suffix=old_suffix))
            df_w = get_dataframe_without_audio()
            stats_results = get_stats(df_w, need_print=False)
            df = process_data(stats_results, lowpass_freq, df)
            assert df['audio'].iloc[0].dtype == np.float32, 'wrong audio type '
            print(get_result_filename(suffix=suffix))
            df.to_pickle(get_result_filename(suffix=suffix))
        
         ### STEP 3 ###
        old_suffix = suffix
        suffix = old_suffix + '_b_size_' + str(merge_bucket_size)
        print('Step 3 : merging samples')
        if not os.path.isfile(get_result_filename(suffix=suffix)):
            print('         not done, computing...')
            df = df if (df is not None) else pd.read_pickle(get_result_filename(suffix=old_suffix))
            df = merge_samples(df, merge_bucket_size)
            assert df['audio'].iloc[0].dtype == np.float32, 'wrong audio type '
            print(get_result_filename(suffix=suffix))
            df.to_pickle(get_result_filename(suffix=suffix))

        ### STEP 4 ###
        old_suffix = suffix
        resample_frequency = resample_frequency if resample_frequency is not None else SAMPLING_FREQUENCY
        suffix = old_suffix + '_freq_' + str(resample_frequency)
        print('Step 4 : Resampling samples')
        if not os.path.isfile(get_result_filename(suffix=suffix)):
            print('         not done, computing...')
            df = df if (df is not None) else pd.read_pickle(get_result_filename(suffix=old_suffix))
            df = resample(df, resample_frequency)
            assert df['audio'].iloc[0].dtype == np.float32, 'wrong audio type '
            print(get_result_filename(suffix=suffix))
            df.to_pickle(get_result_filename(suffix=suffix))

        ### STEP 5 ###
        old_suffix = suffix
        suffix = old_suffix + '_mfcc' if to_mfcc else suffix
        print('Step 5 : To mfcc')
        if not os.path.isfile(get_result_filename(suffix=suffix)):
            print('         not done, computing...')
            print(get_result_filename(suffix=suffix))
            df = df if (df is not None) else pd.read_pickle(get_result_filename(suffix=old_suffix))
            df = transform_to_mfcc(df, resample_frequency)
            df.to_pickle(get_result_filename(suffix=suffix))          
            
        assert df is not None, 'df is None'
        return df
    else :
        print('Dataset already computed, loading from pickle')
        return pd.read_pickle(get_result_filename(suffix=suffix_name))


def get_rain_dataset(compress_labels, split_factor):
    df = pd.read_pickle('/local/user/jrn/tinyml-challenge-2022/final_dataset_86_87_88.pkl')
    X = df['audio'].to_list()
    labels = df['rain_count'].map(compress_labels).to_numpy()
    
    X_list = [np.split(xi, split_factor) for xi in X]
    X = list(itertools.chain.from_iterable(X_list))

    labels = np.repeat(labels, split_factor)
    return X, labels
    

if __name__ == '__main__':
    concat_all_dataframes(True)
    outlier_audio_control = 4
    merge_bucket_size = 3
    resample_frequency = None
    mfcc = False
    dataset = get_final_dataframe(outlier_audio_control, merge_bucket_size, resample_frequency, mfcc)



# if __name__ == '__main__':
#     df_w = get_dataframe_without_audio()

#     stats_results = get_stats(df_w, need_print=False)

#     df = concat_all_dataframes()
#     df = process_data(stats_results, OUTLIER_AUDIO_CONTROL, df)
#     df = merge_samples(df, MERGE_BUCKET_SIZE)

#     df = resample(df, NEW_SAMPLING_FREQUENCY)
#     df.to_pickle(get_result_filename(suffix='_time'))

#     df = to_mfcc(df)
#     df.to_pickle(get_result_filename(suffix='_mfcc'))
#     print(df.head())

#     sys.exit()


    # one file after the other : for RAM consumption
    # for hour, df in dfs_iterator():
    #     print(hour)
    #     df = process_data(stats_results, OUTLIER_AUDIO_CONTROL, df)
    #     df = merge_samples(df, MERGE_BUCKET_SIZE)

    #     df = resample(df, NEW_SAMPLING_FREQUENCY)
    #     df.to_pickle(get_result_filename(hour, '_time'))

    #     df = to_mfcc(df)
    #     df.to_pickle(get_result_filename(hour, '_mfcc'))


    # print('done')

    

    



    
       








    
