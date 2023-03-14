"""""
 *  \brief     train.py
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

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import models
import trainer
from trainer import train
from utils import Cross_validation_splitter
import pickle
from dataset_wind import WindDatasetSimple, WindDatasetMinute
from dataset_8K import AudioDataset_8K
from dataset_vent import AudioDatasetVent
import sys
import os
from utils import get_a_free_gpu

# "AudioDatasetVent", "AudioDataset_8K", "WindDatasetSimple"
choices = ['ventilator', '8K', 'wind', 'wind_minute']
choice_idx = 0
# ['M3', 'M5', 'M11', 'M18', 'M34', 'ACDNet']

network_names = ['M3']

#Overall arguments
epochs = 100
batch_size = 16
num_workers = 8
k_fold = 24
test = True # if run only for 1 fold to test
seed = 3
store = True
store_path =  "results/"


#Arguments for simpleWind
path_base_mic = "data/mic_outside/data_09_09_2022/"
name_mic = 'df_data_microphone_seq_7_run_52436_hour_'
file_names = [path_base_mic + name_mic + str(i) + '.pkl' for i in range(34)]
resample_rate = 8000
length = 4
ignore_first_sec = 0

#Arguments for WindMinute
path_base_wind = "data/mic_outside/data_13_09_2022/processed/"
resample_rate_wind = 8000


#Arguments for AudioDataset8k
file_path_8K = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
audio_path_8K = "data/UrbanSound8K/audio"

#Arguments for AudioDatasetVent
path_base = "data/office/ventilator/"
resample = None
normalize = False



def print_progress_bar(idx, length):
    sys.stdout.write('\r')
    s = '[' + '='*idx + ' '*(length - idx) + ']  ' + str(round(idx*100.0/length, 2)) + '%' 
    if idx == length :
        s += '\n'
    sys.stdout.write(s)
    sys.stdout.flush()


if __name__ == '__main__':

    g = torch.Generator()
    g.manual_seed(seed)

    store_res = {}
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.use_deterministic_algorithms(True)

    if(torch.cuda.is_available() ):
        device = get_a_free_gpu()
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    print("Device:",device)
    categorical_output = True
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    dataset_types = [AudioDatasetVent, AudioDataset_8K, WindDatasetSimple, WindDatasetMinute]
    dataset_type = dataset_types[choice_idx]
    print('loading dataset...')
    if dataset_type == WindDatasetSimple:
        categorical_output = False
        criterion = nn.MSELoss()
        datasets = []
        for idx, file_name in enumerate(file_names):
            dt = dataset_type(file_name, resample_rate=resample_rate, length=length, ignore_first_sec=ignore_first_sec)
            datasets.append(dt)
            print_progress_bar(idx + 1, len(file_names))
            break
           

    elif dataset_type == AudioDataset_8K:
        criterion = nn.CrossEntropyLoss()
        datasets = [dataset_type(file_path_8K, audio_path_8K)]

    elif dataset_type == AudioDatasetVent:
        criterion = nn.CrossEntropyLoss()
        datasets = [dataset_type(path_base, resample, normalize)]

    elif dataset_type == WindDatasetMinute:
        criterion = nn.MSELoss()
        datasets = [WindDatasetMinute(path_base_wind, resample_rate=resample_rate_wind)]
    else:
        raise NotImplementedError

    is_regression = True if type(criterion) is nn.MSELoss else False
    
    criterion = criterion.to(device)

    print("Dataset length:", sum([len(dt) for dt in datasets]))
    print("Dataset shape:", datasets[0].shape())


    for network_name in network_names:
        print("============================")
        print("===========", network_name, "=============")
        print("============================")
        

        model = models.get_model(network_name, datasets, seed)
        model.to(device) 

        dataset = torch.utils.data.ConcatDataset(datasets)

        for weight_decay in [0.0001]:
            optimizer = optim.Adam
            optimizer_args =  {'lr': 0.01, 'weight_decay': weight_decay}
            scheduler = optim.lr_scheduler.StepLR
            scheduler_args = {'step_size': 15, 'gamma': 0.1}

            model, loss_train, loss_test, acc_train, acc_test = train(model, criterion, optimizer, scheduler, dataset, device, epochs, 
                                                                    batch_size, k_fold, num_workers, test,optimizer_args, scheduler_args, seed)
        
            if store : 
                test_str = ""
                if test :
                    test_str = "_test"
                torch.save(model, store_path + choices[choice_idx] + '/' + network_name + '_' + str(weight_decay) + test_str + ".pt")
                if not test :
                    loss_train = np.mean(loss_train, axis=1)
                    loss_test = np.mean(loss_test, axis=1)
                    acc_train = np.mean(acc_train, axis=1)
                    acc_test  = np.mean(acc_test, axis=1)
                if test :
                    loss_train = loss_train[:, 0]
                    loss_test = loss_test[:, 0]
                    acc_train = acc_train[:, 0]
                    acc_test  = acc_test[:, 0]

                

                with open(store_path + choices[choice_idx] + '/' + network_name + '_results' + test_str + '_' + str(weight_decay) + '.pkl', 'wb') as handle:
                    if categorical_output: 
                        pickle.dump((loss_train, loss_test, acc_train, acc_test), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    else :
                        pickle.dump((loss_train, loss_test), handle, protocol=pickle.HIGHEST_PROTOCOL)

    

