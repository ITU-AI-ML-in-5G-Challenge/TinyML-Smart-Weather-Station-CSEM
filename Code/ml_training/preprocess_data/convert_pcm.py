"""""
 *  \brief     convert_pcm_wav.py
 *  \details   Python script to convert recorded PCM file by the MAX78000FTHR to a WAV file
 *  \author    Jona Beysens
 *  \version   1.0
 *  \date      2022-07-29
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

from cgitb import small
import os
import sys
import wave
import pickle
import sys
import array
import pandas as pd
from constants import *
import multiprocessing
import numpy as np
from scipy.io import wavfile


#### constants to adapt ####
# settings for the WAV file
num_channels = 1 # mono = 1, stereo = 2
sample_duration = 1 # in seconds (for every sample, we have a header with timestamp information)
sample_header_size = 4 # in bytes
sample_width = 2 # in bytes
num_frames = 0 # number of frames in the file (will be changed automatically when data is written to a wav file)
bme_sensor_size = 3 * 8 # temperature, pressure, temperature stored using 8 bytes

test = "test_"
### constants : do not change ####

def extract_bme_sensor_data(start_index, pcmdata):
    pcm_sensor_data = pcmdata[start_index: start_index + bme_sensor_size]
    double_sequence = list(array.array('d', pcm_sensor_data))
    return double_sequence

def process_pcm(idx, path_pcm):
    with open(path_pcm, 'rb') as pcmfile:
        pcmdata = pcmfile.read()
        data = []
        timestamps = []
        convert_data = []
        bme_sensor_data = []
        sample_size = sample_header_size + sample_duration * num_channels * SAMPLING_FREQUENCY * sample_width + bme_sensor_size
        num_samples = int(len(pcmdata) / sample_size)
        print(len(pcmdata))
        print(sample_size)
        print(len(pcmdata)/(sample_size * 1.0))
        for j in range(num_samples):
            #extract timestamp
            timestamp = pcmdata[j * sample_size     ] << 24
            timestamp += pcmdata[j * sample_size + 1] << 16
            timestamp += pcmdata[j * sample_size + 2] << 8
            timestamp += pcmdata[j * sample_size + 3]

            start_index = j * sample_size + sample_header_size
            if j % 100 == 0: 
                print(f'Timestamp of sample {j}: {timestamp}, start_index: {start_index}')
            start_index_arr =  int((start_index - 4 * (j + 1))/ 2)
            timestamps.append(timestamp)
            #Extract audio
            for i in range(0, int(SAMPLING_FREQUENCY * sample_duration)): # don't multiply by sample width, because in loop width of sample is handled
                value =     pcmdata[start_index + 2*i       ] << 8
                value +=    pcmdata[start_index + 2*i + 1  ]
                if(value>2**15 - 1):
                    value = -(2**16 - value)
                convert_data.append(value)

            #Extract BME sensor data
            sensor_start_index = start_index + int(SAMPLING_FREQUENCY * sample_duration) * 2
            row_bme_sensor = extract_bme_sensor_data(sensor_start_index, pcmdata)
            bme_sensor_data.append(row_bme_sensor)

        
        data = b''.join((wave.struct.pack('h', item) for item in convert_data if item ))
        print(f"length of data : {len(data)}")

 
    wave_filename = get_filename(idx, '.wav')
    print(wave_filename)
    with wave.open(WAVE_FOLDER + test + wave_filename , 'wb') as wavefile:
        wavefile.setparams((num_channels, sample_width, SAMPLING_FREQUENCY, num_frames, 'NONE', 'NONE'))
        wavefile.writeframes(bytes(data))
        
    # with scipy : equivalent + simpler
    # left_channel = convert_data[0::2]
    # right_channel = convert_data[1::2]
    # stereo_output=np.vstack((left_channel, right_channel)).T
    # stereo_output = np.array(convert_data)
    # wavfile.write('stereo_audio.wav', SAMPLING_FREQUENCY, stereo_output.astype(np.int16))


    #Store bme sensor data and timestamp as pandas dataframe
    df = pd.DataFrame(bme_sensor_data, columns=['humidity', 'pressure', 'temperature'])
    df['timestamp'] = timestamps
    print(df)
    pickle_filename = get_filename(idx, '.pkl')
    df.to_pickle(PICKLE_FOLDER + pickle_filename)


    



    
if __name__ == '__main__':
    if not os.path.exists(SENSOR_FOLDER):
        os.makedirs(SENSOR_FOLDER)
    if not os.path.exists(WAVE_FOLDER):
        #for audio recording
        os.makedirs(WAVE_FOLDER)
    if not os.path.exists(PICKLE_FOLDER):
        #for dataframe of [humidity, pressure, temperature, timestamp]
        os.makedirs(PICKLE_FOLDER)

    filenames_pcm = [get_filename(i, '.pcm', True)  for i in range(MIC_NUM_HOURS)]
    paths_pcm = [os.path.join(MIC_FOLDER_RAW, filename_pcm) for filename_pcm in filenames_pcm]

    print("=================================")
     #Run sequentially
    print("processing in sequentially...")
    for idx, path_pcm in enumerate(paths_pcm):
        process_pcm(idx, path_pcm)



