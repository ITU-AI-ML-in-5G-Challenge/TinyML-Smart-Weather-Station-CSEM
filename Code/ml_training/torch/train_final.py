"""""
 *  \brief     train_final.py
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

from dataset import TinyDataset
from tf_learning_hts import  get_hts_model
import pickle
from sklearn.utils.class_weight import compute_class_weight
import sys
import os
from utils_torch import get_a_free_gpu


split_factor = 2
type_labels = 'rain'

#Overall arguments
epochs = 21
batch_size = 64
num_workers = 8
k_fold = 5
test = True # if run only for 1 fold to test
seed = 3
store = False
store_path =  "results/torch/"




def get_class_weights(labels):
    y_integers = np.argmax(labels, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
    return dict(enumerate(class_weights))


def main(num_trainable_outputs):

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

    rain_split_idx = (1, 2, 4)
    dataset = TinyDataset(split_factor, type_labels, rain_split_idx, device)
    num_classes = dataset.num_classes
    
    print('Length dataset:', len(dataset), ", num_classes:", num_classes)
    class_weights = get_class_weights(dataset.y)
    weights = torch.Tensor(list(dict(sorted(class_weights.items())).values())).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    model = get_hts_model(num_classes, num_trainable_outputs).to(device) 
    

    optimizer = optim.Adam
    optimizer_args =  {'lr': 1e-03}
    scheduler = optim.lr_scheduler.ReduceLROnPlateau
    scheduler_args = dict(factor=0.6, patience=2, min_lr=1e-07, threshold=0.02, verbose=True)
    

    model, loss_train, loss_test, acc_train, acc_test, conf_res = train(model, criterion, optimizer, scheduler, dataset, device, epochs, 
                                                            batch_size, k_fold, num_workers, test, optimizer_args, scheduler_args,
                                                            verbose=2, seed=seed)
    

    if store : 
        torch.save(model, store_path +  "htsat.pt")
        
    return loss_test, acc_test, conf_res


    
if __name__ == '__main__':
    train_layers = [2, 5, 8, 10, 12, 15, 18, 20]
    file = '/local/user/jrn/tinyml-challenge-2022/htsat_results.txt'
    for num_layers in train_layers:
        loss_test, acc_test, conf_res = main(num_layers)
        with open(file, 'a') as f:
            print(num_layers,  " acc: ", acc_test, file=f)
            print(repr(conf_res), file=f)
            
    
    

        