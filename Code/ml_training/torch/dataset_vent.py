"""""
 *  \brief     dataset_vent.py
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
import os
import numpy as np
import wave
import torchaudio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import Resample



date = "20220810"
device_rec = "mbp"
wind_levels = range(4)
run_ids = [0, 1]



def extract_dataset(path_base, resample_rate=None, normalize=False):
    if normalize:
        raise NotImplementedError
    x = []
    labels = []
    for run_id in run_ids:
        for wind_level in wind_levels:
            filename_wav = date + "_" + device_rec + "_ventilator_run_" + str(run_id) + "_level_" + str(wind_level) + ".wav"
            path_wav = os.path.join(path_base,filename_wav)
            wavefile = wave.open(path_wav, 'rb')
            audio_numframes = wavefile.getnframes()
            audio_samplerate = wavefile.getframerate()
            audio_data = np.frombuffer(wavefile.readframes(audio_numframes), dtype=np.int8) 
            audio_data_float = np.float32(audio_data)

            #short to be divided by 300
            audio_data_float = audio_data_float[:9598800]

            #remove dc bias
            audio_data_float -= audio_data_float.mean()

            audio_list = np.split(audio_data_float, 300) #to get approx 1s samples
            # mfcc_list = [get_mfcc(a, fs) for a in audio_list]
            audio_list = [np.concatenate((a, [0]*(32000 - len(a)))) for a in audio_list ]
            audio_list = [torch.Tensor(a).unsqueeze(0) for a in audio_list]
            #TODO : see if store the resampled data
            if resample_rate is not None:
                raise NotImplementedError
                # audio_list = Resample(16384, resample_rate, lowpass_filter_width=128)

            x += audio_list
            curr_labels = [wind_level] * len(audio_list) 

            labels += curr_labels
    
    return x, labels, audio_samplerate



class AudioDatasetVent(Dataset):
    """
    A rapper class for the ventilator dataset.
    """

    def __init__(self, path_base, resample_rate=None, normalize=False):
        """
        Args:data
            path_vase(string): path to the audio files
            resample(float): if resample .wav during process
            normalize(bool): if normalize data or not
        """

        self.x, self.labels, self.fs = extract_dataset(path_base, resample_rate, normalize)
    

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return self.x[idx], self.labels[idx]

    def shape(self):
        length = len(self)
        shape_input = self[0][0].shape
        return length, shape_input

    def get_num_outputs(self):
        return 4