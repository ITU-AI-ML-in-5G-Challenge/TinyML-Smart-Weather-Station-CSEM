"""""
 *  \brief     dataset.py
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

import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
import sys
import itertools
from collections import Counter
sys.path.insert(0, 'ml_training/')
sys.path.insert(0, 'ml_training/preprocess_data')
from preprocess_data.constants import SAMPLING_FREQUENCY, INDEX_HOURS, get_result_filename
sys.path.insert(0, 'ml_training/tools')
from tools import print_progress_bar
from utils_torch import to_one_hot
from torchaudio.transforms import Resample
from functools import partial


# def compress_labels_wind(x):
#     if x < 6:
#         return 0
#     elif x < 12:
#         return 1
#     else :
#         return 2
    
# def compress_labels_wind(x):
#     if x < 6:
#         return 0
#     elif x < 14:
#         return 1
#     # elif x < 18:
#     #     return 2
#     else :
#         return 2
    
# def compress_labels_wind(x):
#     if x < 6:
#         return 0
#     elif x < 12:
#         return 1
#     elif x <  18:
#         return 2
#     elif x < 24 :
#         return 3
#     elif x < 30 :
#         return 4
#     else :
#         return 5

# def compress_labels_rain(x):
#     if x >= 3 :
#         return 2
#     elif x >= 1 :
#         return 1
#     else :
#         return 0
    
# def compress_labels(y, label):
#     if label=='rain':
#         return compress_labels_rain(y)
#     elif label=='wind':
#         return compress_labels_wind(y)
#     else :
#         y_rain, y_wind = y
#         return compress_labels_rain(y_rain), compress_labels_wind(y_wind)


def compress_labels(x, split_indexes):
    idx = 0
    while idx < len(split_indexes):
        if x < split_indexes[idx]:
            return idx
        idx += 1
    return len(split_indexes)


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
        labels = df[label_name].to_numpy().astype(int)
        return np.repeat(labels, split_factor)
    
    if type_labels == 'rain':
        return X, get_labels('rain_level')
    elif type_labels == 'wind':
        return X, get_labels('wind_count')
    else :
        return X, (get_labels('rain_level'), get_labels('wind_count'))
    
    


class TinyDataset(Dataset):
    def __init__(self, split_factor, type_labels, split_idx, device=None, new_sf=None):
        self.type_labels = type_labels
        self.x, self.y = get_dataset(split_factor, type_labels)
        # TODO : modify for wind+rain
        # self.y = np.vectorize(compress_labels)(self.y, type_labels)
        self.y = np.vectorize(partial(compress_labels, split_indexes=split_idx))(self.y)
        self.split_factor = split_factor
        
        # print(self.y)
        # sys.exit()
        
        self.num_classes = len(Counter(self.y).keys())
        print(self.num_classes)
        
        # keys = sorted(list(Counter(self.y).keys()))
        # d = dict(zip(keys, range(len(keys))))
        # self.y = np.vectorize(d.get)(self.y)
        
        # TODO : redo after
        resampler = Resample(SAMPLING_FREQUENCY, SAMPLING_FREQUENCY * 2, resampling_method="kaiser_window", lowpass_filter_width=8)
        self.resample = (lambda x : resampler(torch.FloatTensor(x)))
        # for i in range(len(self.x)):
        #     self.x[i] = self.resample(self.x[i])
        
        self.y = to_one_hot(self.y, self.num_classes)
        # if device:
        #     self.y = torch.Tensor(self.y).to(device)
        #     self.x = torch.Tensor(self.x).to(device)
            
        # print(self.y.shape)
        
 
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.type_labels in ['rain', 'wind']:
            return np.tile(self.x[idx],2 * self.split_factor), self.y[idx]
        else :
            return self.x[idx], (self.y[0][idx], self.y[1][idx])

        