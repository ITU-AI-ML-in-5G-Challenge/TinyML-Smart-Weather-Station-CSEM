"""""
 *  \brief     process_esc50.py
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

import pickle
from glob import iglob
from shutil import rmtree

import numpy as np

from constants_ml import *
from dataloader_esc50 import read_audio_from_filename


DATA_AUDIO_DIR = 'data/UrbanSound8K/audio'
TARGET_SR = 8000
OUTPUT_DIR = '/tmp/very-deep-conv-nets-raw-waveforms'
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_DIR_TEST = os.path.join(OUTPUT_DIR, 'test')
AUDIO_LENGTH = 32000



def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def del_folder(path):
    try:
        rmtree(path)
    except:
        pass


del_folder(OUTPUT_DIR_TRAIN)
del_folder(OUTPUT_DIR_TEST)
mkdir_p(OUTPUT_DIR_TRAIN)
mkdir_p(OUTPUT_DIR_TEST)


def extract_class_id(wav_filename):
    """
    The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav, where:
    [fsID] = the Freesound ID of the recording from which this excerpt (slice) is taken
    [classID] = a numeric identifier of the sound class (see description of classID below for further details)
    [occurrenceID] = a numeric identifier to distinguish different occurrences of the sound within the original recording
    [sliceID] = a numeric identifier to distinguish different slices taken from the same occurrence
    """
    return wav_filename.split('-')[1]


def convert_data():
    for i, wav_filename in enumerate(iglob(os.path.join(DATA_AUDIO_DIR, '**/**.wav'), recursive=True)):
        class_id = extract_class_id(wav_filename)
        audio_buf = read_audio_from_filename(wav_filename, target_sr=TARGET_SR)
        # normalize mean 0, variance 1
        audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
        original_length = len(audio_buf)
        print(i, wav_filename, original_length, np.round(np.mean(audio_buf), 4), np.std(audio_buf))
        if original_length < AUDIO_LENGTH:
            audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
            print('PAD New length =', len(audio_buf))
        elif original_length > AUDIO_LENGTH:
            audio_buf = audio_buf[0:AUDIO_LENGTH]
            print('CUT New length =', len(audio_buf))

        output_folder = OUTPUT_DIR_TRAIN
        if 'fold10' in wav_filename:
            output_folder = OUTPUT_DIR_TEST
        output_filename = os.path.join(output_folder, str(i) + '.pkl')

        out = {'class_id': class_id,
               'audio': audio_buf,
               'sr': TARGET_SR}
        with open(output_filename, 'wb') as w:
            pickle.dump(out, w)


if __name__ == '__main__':
    convert_data()
