"""""
 *  \brief     Python script used for ML training
 *  \author    Jonathan Reymond, Robin Berguerand, Jona Beysens
 *  \version   1.0
 *  \date      2022-11-14
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
 
import glob
import numpy as np


############################
#### constants to adapt ####
############################
#End of recording, for creating the folder
DATE_RECORDING = '9_11_2022'
#recording of interest (to crop the original recording)
#format : "year-month-day hour:minutes:seconds"
REAL_START_DATE = None
REAL_END_DATE =  None

MIC_ID = None
MIC_NUM_HOURS = None
MIC_PREFIX = None
MIC_START_DATE = None

#format : "year-month-day hour:minutes:seconds"
STATION_START_DATE = None

SLOPE = 1#0.9967787669329223


def initialize_global_constants():
    global REAL_START_DATE
    global REAL_END_DATE
    global MIC_ID
    global MIC_NUM_HOURS
    global MIC_PREFIX
    global MIC_START_DATE
    global STATION_START_DATE

    if DATE_RECORDING == '3_11_2022':
        REAL_START_DATE = '2022-10-28 11:39:00'
        REAL_END_DATE = '2022-11-03 10:00:00'
        MIC_ID = 59318
        MIC_NUM_HOURS = 89
        MIC_PREFIX = 'data_seq_4_run_'
        MIC_START_DATE = '2022-10-28 11:31:47'
        STATION_START_DATE = '2022-10-28 11:33:54'

    elif DATE_RECORDING == '9_11_2022':
        REAL_START_DATE = '2022-11-03 10:40:00'
        REAL_END_DATE = '2022-11-08 13:50:00'
        MIC_ID = 32856
        MIC_NUM_HOURS = 124
        MIC_PREFIX = 'data_seq_2_run_'
        MIC_START_DATE = '2022-11-03 10:27:50'
        STATION_START_DATE = '2022-11-03 10:23:29'
        
    elif DATE_RECORDING == '11_11_2022':
        REAL_START_DATE = '2022-11-08 14:15:00'
        REAL_END_DATE = '2022-11-11 14:45:00'
        MIC_ID = 55191
        MIC_NUM_HOURS = 74
        MIC_PREFIX = 'data_seq_4_run_'
        MIC_START_DATE = '2022-11-08 14:01:15'
        STATION_START_DATE = '2022-11-08 13:59:24'

    else :
        raise NotImplementedError


initialize_global_constants()

INDEX_HOURS = range(5, 29) # range(NUM_HOURS)

###################################
#### constants : do not change ####
###################################

SAMPLING_FREQUENCY = 16384 # 16kHz

MIC_FOLDER_RAW = "data/outside/" + DATE_RECORDING + "/max_recordings"
STATION_FOLDER = "data/outside/" + DATE_RECORDING + "/weather_station_microbit/"
TIMESTAMP_COMPARISON_FILE = "data/outside/" + DATE_RECORDING + "/other/timestamp_comparison.csv"
OUTPUT_FOLDER = 'data/processed/data_' + DATE_RECORDING + '/'  'cropped/'



WIND_VECTOR = True

# Time of the first started
BASE_TIME = min(np.datetime64(MIC_START_DATE), np.datetime64(STATION_START_DATE))

REAL_START_TIMESTAMP = (np.datetime64(REAL_START_DATE) - BASE_TIME)/ np.timedelta64(1, 'ms')
REAL_END_TIMESTAMP = (np.datetime64(REAL_END_DATE) - BASE_TIME)/ np.timedelta64(1, 'ms')

SENSOR_FOLDER = 'data/outside/' + DATE_RECORDING + '/'
WAVE_FOLDER = SENSOR_FOLDER + 'wave_files/'
PICKLE_FOLDER = SENSOR_FOLDER + 'pickle_files/'
# Difference between the first started and the mic start time + converted in sec
MIC_START_TIMESTAMP = (np.datetime64(MIC_START_DATE) - BASE_TIME)/ np.timedelta64(1, 'ms')
MIC_HEADER = ['humidity', 'pressure', 'temperature', 'timestamp']

STATION_HEADER = ['timestamp', 'wind_dir', 'wind_count', 'rain_count']
STATION_FILENAMES = lambda: glob.glob(STATION_FOLDER + '*.TXT', recursive=True)
# Difference between the first started and the mic start time + converted in sec
STATION_START_TIMESTAMP = (np.datetime64(STATION_START_DATE) - BASE_TIME)/ np.timedelta64(1, 'ms')
# maxCount in microbit
THRESHOLD = 1000000
WIND_DIRECTIONS = {'N' : 0, 'NE' : 1, 'E' : 2, 'SE' : 3, 'S' : 4, 'SW' : 5, 'W' : 6, 'NW' : 7}
RAD_ANGLE = 2* np.pi / len(WIND_DIRECTIONS)


########################################
#### Function to get right filename ####
########################################

def get_filename(hour, suffix, with_prefix=False):
    '''Get filename of recording of sensor

    Args:
        hour (int): desired hour of recording
        suffix (string): get suffix : .pcm, .wav or .pkl
        with_prefix (bool, optional): only for extracting the .pcm data, to get the entire original name. Defaults to False.

    Returns:
        str : filename (without path)
    '''
    if with_prefix : 
        return MIC_PREFIX + str(MIC_ID) + '_hour_' + str(hour) + suffix
    else :
        return str(MIC_ID) + '_hour_' + str(hour) + suffix



def get_result_filename(hour=None, suffix=""):
    '''return the resulting dataframe path storing path

    Args:
        hour (int): to select the dataframe of which hour (one dataframe per hour), if None : get the entire dataframe
        suffix (str): suffix to add to the filename

    Returns:
        str: filename path
    '''
    if hour is None :
        return OUTPUT_FOLDER + 'data' + suffix + '.pkl'
    else :
        return OUTPUT_FOLDER + 'data_hour_' + str(hour) + suffix + '.pkl'





