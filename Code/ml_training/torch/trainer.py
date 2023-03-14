"""""
 *  \brief     trainer.py
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
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from utils_torch import Cross_validation_splitter, to_one_hot, print_progress_bar
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from models import init_weights
import time
import copy
import sys
from torchmetrics import ConfusionMatrix, Accuracy



# class Trainer:
#     def __init__(self, model, criterion, optimizer, scheduler, dataset, device, epochs, batch_size, k_fold, num_workers, test=False, optimizer_args={}, scheduler_args={}, seed=1):
#         g = torch.Generator()
#         g.manual_seed(seed)
        
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.dataset = dataset
#         self.device = device
#         self.epochs = epochs
#         self.batch_size = batch_size
        
        


def train(model, criterion, optimizer, scheduler, dataset, device, epochs, batch_size, k_fold, num_workers, test=False, optimizer_args={}, scheduler_args={},verbose=2, seed=1):
    criterion = criterion.to(device)
    model.to(device) 

    confmat = ConfusionMatrix(task="multiclass", num_classes=dataset.num_classes).to(device)
    accu = Accuracy(task='multiclass', num_classes=dataset.num_classes, top_k=1, average='macro').to(device)

    g = torch.Generator()
    g.manual_seed(seed)

    splitter = Cross_validation_splitter(k_fold, len(dataset), seed)

    loss_train = np.zeros((epochs, k_fold))
    loss_test = np.zeros((epochs, k_fold))
    acc_train = np.zeros((epochs, k_fold))
    acc_test  = np.zeros((epochs, k_fold))

    k_folds = [0] if test else range(k_fold)

    for fold_number in k_folds:
        optim = optimizer(model.parameters(), **optimizer_args)
        sched = scheduler(optim, **scheduler_args)
        since = time.time()
        best_loss = float('inf')
        best_acc = 0

        # model.apply(lambda a : init_weights(a, seed))
        best_model_wts = copy.deepcopy(model.state_dict())
        print("Fold", fold_number + 1)

        train_set, test_set = splitter.get_kfold_sets(dataset, fold_number)            
            
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if device == torch.device("cuda") else {}
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, generator=g, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, generator=g, **kwargs)

        for epoch in range(1, epochs + 1):
                
            start_time = time.time()

            acc, loss = train_step(model, optim, criterion, train_loader, verbose, device)
            loss_train[epoch-1, fold_number] = loss
            acc_train[epoch-1, fold_number] = acc
            acc_t,loss_t = test_step(model, criterion, test_loader, device, confmat, accu, epoch)
            loss_test[epoch-1, fold_number] = loss_t
            acc_test[epoch-1, fold_number] = acc_t
            sched.step(loss)
            if loss_t < best_loss:
                best_loss = loss_t
                best_acc = acc_t
                best_model_wts = copy.deepcopy(model.state_dict())


            epoch_str = 'Epoch {}, train_loss {:.4f}, test_loss {:.4f}'.format(epoch, loss, loss_t)
            epoch_str += ', train_acc {:.5f}, test_acc {:.5f}'.format(acc, acc_t)
            epoch_str += ',  {:.2f} seconds'.format(time.time() - start_time) 
            if verbose != 0:
                print(epoch_str)
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best loss : {best_loss:4f}, corresponding acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        print('Results')
        _, _, conf_res = test_step(model, criterion, test_loader, device, confmat, accu, epoch, True)
        

    
    return model, loss_train, loss_test, acc_train, acc_test, conf_res



def train_step(model, optimizer, criterion, train_loader, verbose, device):
    model.train()
    avg_loss = 0
    batch_step = 0
    num_batches = len(train_loader)
    for data, target in train_loader:
        if (verbose == 2) and (batch_step % 10 == 0 or (batch_step + 1) == num_batches):
            print_progress_bar(batch_step, num_batches)
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        
        if isinstance(output, dict):
            # case HTS transformer
            output = output['clipwise_output']

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        batch_step += 1
    
    return 0, avg_loss/ len(train_loader)


def test_step(model, criterion, test_loader, device, confmat, acc, epoch, conf_return=False):
    test_avg_loss = 0
    model.eval() #turns off dropout and batchnorm during test time
    
    targets = []
    preds = []

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        if isinstance(output, dict):
            output = output['clipwise_output']
        loss = criterion(output, target)
        test_avg_loss += loss.item()
        targets.append(target.argmax(axis=1).int())
        preds.append(output.argmax(axis=1).int())
        
    targets = torch.cat(targets)
    preds = torch.cat(preds)
    
    if conf_return:
        conf_res = confmat(preds, targets).cpu().numpy()
        return acc(preds, targets).item(), test_avg_loss/ len(test_loader), conf_res
    else :
        return acc(preds, targets).item(), test_avg_loss/ len(test_loader)



def count_correct(output, target):
    with torch.no_grad():
        output = F.softmax(output, dim=1)
        pred = torch.argmax(output, 1)
        target = torch.argmax(target, dim=1)
        
        return torch.sum(torch.eq(pred, target))


def count_correct_regression(output, target, precision=0.5):
    with torch.no_grad():
        count_correct = torch.sum(torch.abs(output - target) < precision)
    return count_correct
        



