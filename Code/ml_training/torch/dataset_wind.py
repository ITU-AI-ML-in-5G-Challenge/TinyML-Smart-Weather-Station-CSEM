"""""
 *  \brief     dataset_wind.py
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
from torchaudio.transforms import Resample
import numpy as np
import torch
import glob




class WindDatasetSimple(Dataset):
    def __init__(self, file_name, fs=16384, resample_rate=None, length=1, ignore_first_sec=0):
        df = pd.read_pickle(file_name)
         
        self.x = df['value'].apply(torch.FloatTensor)
        self.labels = df['wind_speed']
        self.origin_fs = fs
        self.length = length
        self.ignore_first_sec=ignore_first_sec

        if resample_rate is not None:
            resampler = Resample(fs, resample_rate)
            self.x = self.x.apply(resampler)
 

    def __len__(self):
        #TODO : modify to take into account ignore first seconds
        return int(len(self.labels) / self.length)

    def __getitem__(self, idx):
        begin_idx = int(idx + self.ignore_first_sec * self.fs)
        x = torch.cat(self.x[idx: idx + self.length].to_list())
        label = self.labels[idx: idx + self.length].mean()
        return x.unsqueeze(0), torch.FloatTensor([label])

    def shape(self):
        length = len(self)
        shape_input = self[0][0].shape
        return length, shape_input

    def get_num_outputs(self):
        return 1




class WindDatasetMinute(Dataset):
    def __init__(self, path_base, prefix='row_16s_', fs=16384, resample_rate=None, ignore_first_sec=0):
        self.fs = fs
        self.audio_paths = glob.glob(path_base + prefix + '*')
        

        self.resample_rate = resample_rate

        if resample_rate is not None:
            self.resampler = Resample(fs, resample_rate, lowpass_filter_width=128)
 

    def __len__(self):
        #TODO : modify to take into account ignore first seconds + if split each minute into smaller
        return len(self.audio_paths)

    def __getitem__(self, idx):
        row = pd.read_pickle(self.audio_paths[idx])
        #Only take for now the audio value
        x = row['value']
        if self.resample_rate is not None:
            x = self.resampler(torch.FloatTensor(x))
        #TODO : remove it when end test
        x = x[:32000]
        x = torch.FloatTensor(x).unsqueeze(0)
        #TODO : think of moving this step at end of ml analysis into preprocess
        
        
        label = torch.FloatTensor([row['wind_speed']])
        return x, label


    def shape(self):
        length = len(self)
        shape_input = self[0][0].shape
        return length, shape_input

    def get_num_outputs(self):
        return 1


# path_base_wind = "data/mic_outside/data_13_09_2022/processed/"
# resample_rate_wind = 8000
# w = WindDatasetMinute(path_base_wind, resample_rate=resample_rate_wind)
# w[0]
# print(w[1])