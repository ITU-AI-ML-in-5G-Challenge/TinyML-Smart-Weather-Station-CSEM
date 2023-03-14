"""""
 *  \brief     dataset_8K.py
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

from torch.utils.data import Dataset
import pandas as pd
import glob
import torchaudio
import torch.nn.functional as F

class AudioDataset_8K(Dataset):
    """
    A rapper class for the UrbanSound8K dataset.
    """

    def __init__(self, file_path, audio_path):
        """
        Args:data
            file_path(string): path to the audio csv file
            root_dir(string): directory with all the audio folds
        """
        self.audio_file = pd.read_csv(file_path)

        audio_paths = []
        for fold in range(1, 11):
            audio_paths += glob.glob(audio_path + '/*' + str(fold) + '/*')
        self.audio_paths = audio_paths

        _, self.fs = torchaudio.load(self.audio_paths[0])
        

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        
        audio_path = self.audio_paths[idx]
        audio, rate = torchaudio.load(audio_path)
        audio = audio.mean(0, keepdim=True)
        c, n = audio.shape
        zero_need = 160000 - n
        audio_new = F.pad(audio, (zero_need //2, zero_need //2), 'constant', 0)
        audio_new = audio_new[:,::5]
        
        #Getting the corresponding label
        audio_name = audio_path.split(sep='/')[-1]
        label = self.audio_file.loc[self.audio_file.slice_file_name == audio_name].iloc[0,-2]
        return audio_new, label

    def shape(self):
        length = len(self)
        shape_input = self[0][0].shape
        return length, shape_input, self[0][1].shape

    def get_num_outputs(self):
        return 10
