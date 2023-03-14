"""""
 *  \brief     utils_torch.py
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

import subprocess
import numpy as np
import torch
from dotenv import load_dotenv
load_dotenv()
import sys


class Cross_validation_splitter():
    def __init__(self, k_fold, num_row, seed):
        self.k_fold = k_fold
        self.seed = seed
        self.num_row = num_row
        self.k_indices = self.build_k_indices()


    def build_k_indices(self):
        """build k indices for k-fold.
        Returns:
            A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

        >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
        array([[3, 2],
            [0, 1]])
        """
        interval = int(self.num_row / self.k_fold)
        np.random.seed(self.seed)
        indices = np.random.permutation(self.num_row)
        k_indices = [indices[k * interval: (k + 1) * interval] for k in range(self.k_fold)]
        return np.array(k_indices)    

    def get_kfold_sets(self, dataset, k):
        te_indice = self.k_indices[k]
        tr_indice = self.k_indices[~(np.arange(self.k_indices.shape[0]) == k)]
        tr_indice = tr_indice.reshape(-1)

        train_set = torch.utils.data.Subset(dataset, tr_indice)
        test_set =  torch.utils.data.Subset(dataset, te_indice)
        return train_set, test_set

        


def get_a_free_gpu():
    """Returns the available GPU torch device with the most free memory.
    Returns
    -------
    torch.device
            A torch device able to carry the data.

    """
    ### Caution: needs the right environment variable to be set. (CUDA_DEVICE_ORDER=PCI_BUS_ID)
    # This is done in the __init__.py.
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')

    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    device_name = 'cuda:'+str(list(gpu_memory_map.keys())[np.argmax(np.array(list(gpu_memory_map.values())))])

    return torch.device(device_name)


def to_one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


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