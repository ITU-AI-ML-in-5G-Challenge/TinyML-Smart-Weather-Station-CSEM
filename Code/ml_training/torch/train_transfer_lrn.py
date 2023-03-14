"""""
 *  \brief     train_transfer_lrn.py
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

import time
import copy
import torch
import models
import torch.nn as nn
from trainer import train
from dataset_vent import AudioDatasetVent
import torch.optim as optim

num_classes = 4
#assume no need to retrain with diff input length due to conv layer properties
new_input_length = 32000
old_input_length = 30225
new_fs = 16384
old_fs = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


epochs = 20
batch_size = 64
num_workers = 16
k_fold = 5
test = False # if run only for 1 fold to test
seed = 3
criterion = nn.CrossEntropyLoss()
path_base = "data/office/ventilator/"
resample = None
normalize = False
dataset = AudioDatasetVent(path_base, resample, normalize)
optimizer = optim.Adam
optimizer_args =  {'lr': 0.01, 'weight_decay': 0.0001}
scheduler = optim.lr_scheduler.StepLR
scheduler_args = {'step_size': 20, 'gamma': 0.1}


#ACDNet 

store_base = 'results/8K/acdnet_20_web.pt'
state = torch.load(store_base, map_location=device);
model = models.ACDNetV2(input_length=new_input_length, n_class=50, fs=new_fs)

model.load_state_dict(state['weight'], strict=False);
print('Model Loaded');



num_ftrs = model.fcn.in_features
model.fcn = nn.Linear(num_ftrs, num_classes)

for param in list(model.parameters())[:len(list(model.parameters())) - 3]:
    param.requires_grad = False

# model.fcn.weight.requires_grad_(True)
# model.fcn.bias.requires_grad_(True)
model = model.to(device)


model, loss_train, loss_test, acc_train, acc_test = train(model, criterion, optimizer, scheduler, dataset, device, epochs, 
                                                                batch_size, k_fold, num_workers, test,optimizer_args, scheduler_args, seed)







