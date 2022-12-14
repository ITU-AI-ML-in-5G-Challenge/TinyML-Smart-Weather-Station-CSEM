/*
; ============

;------------------------------------------------------------------------
; Author:	Jonathan Reymond, Robin Berguerand, Jona Beysens	2022-11-14
; Modifs:
;
; Goal:		Code for embedded board MAX78000FTHR
;
;   (c) 1992-2022
;   --------------------------
;
;   CSEM S.A.
;   Jaquet-Droz 1
;   CH-2000 Neuchâtel
;   http://www.csem.ch
;
;   THIS PROGRAM IS CONFIDENTIAL AND CANNOT BE DISTRIBUTED
;   WITHOUT THE CSEM PRIOR WRITTEN AGREEMENT.
;
;------------------------------------------------------------------------
*/

#include "microphone.h"
#include "bme_sensor.h"


struct bme280_data sensor_bme_data;

uint8_t pAI85Buffer[SAMPLE_HEADER_SIZE+ DATA_SIZE*SAMPLE_SIZE + sizeof(sensor_bme_data)]; 
uint8_t* bme_buffer;

uint32_t sampleCounter = 0;
uint16_t wordCounter = 0;


int main(void)
{
    int error = 0;
    /* Enable cache */
    MXC_ICC_Enable(MXC_ICC0);
	/* Initialize RTC */
	error = MXC_RTC_Init(0, 0);
    if(error != E_NO_ERROR){
        PR_DEBUG("error MXC_RTC_Init: %d\n", error);
    }
	error = MXC_RTC_Start();
    if(error != E_NO_ERROR){
        PR_DEBUG("error MXC_RTC_Start: %d\n", error);
    }
    configure_clock_source();

    // Wait for PMIC 1.8V to become available, about 180ms after power up.
    MXC_Delay(200000);

    PR_INFO("CSEM\nData Acquisition \nVer. %s \n", VERSION);
    PR_INFO("\n***** Init *****\n");
    memset(pAI85Buffer, 0x0, sizeof(pAI85Buffer));
    PR_DEBUG("pAI85Buffer: %d\n", sizeof(pAI85Buffer));
    // memset(bme_buffer, 0x0, sizeof(bme_buffer));
    // PR_DEBUG("bme_buffer: %d\n", sizeof(bme_buffer));

    bme_buffer = &pAI85Buffer[SAMPLE_HEADER_SIZE+ DATA_SIZE*SAMPLE_SIZE];


    sdhc_init();
    PR_INFO("mounting sdhc\n");
    while (error = sdhc_mount() != FR_OK) {
        MXC_Delay(SEC(1));
        printf("Retrying to mount SD card\n");
    }

    MXC_Delay(2000000);

    mic_init();
    MXC_Delay(200000);
    //false : assume that i2c already initialized
    init_bme_sensor(&sensor_bme_data, false);
    MXC_Delay(200000);

    char filename[60];
    int run_id = 0;
    int ret = 0;

    int timestamp_start = 0;
    uint32_t timestamp = 0;
    uint16_t timestamp_hour = 0;
    uint16_t num_files = 0;
    int8_t rslt;


    timestamp_start = utils_get_time_ms();
    run_id = utils_get_TRNG_sample(); // get random run id based on TRNG

  
    printf("Start timestamp of measurements: %d \n", timestamp_start);
    sdhc_ls(&num_files);

    int i = 0;
    while (1) {
        timestamp = utils_get_time_ms();
        // timestamp_hour = utils_ms_to_minutes(timestamp);
        timestamp_hour = utils_ms_to_hour(timestamp);

        snprintf(filename,sizeof(filename),"data_seq_%d_run_%d_hour_%d.pcm",num_files,run_id,timestamp_hour);
        printf("-----------------\n");
        printf("Writing to file with name: %s \n", filename);
        printf("Acquiring word: %d \n", i+1);

        LED_On(0); // red LED
        recordOneWord(pAI85Buffer, &wordCounter, &sampleCounter);
        rslt = record_one_bme_measure(&sensor_bme_data);
        print_sensor_data(&sensor_bme_data);
        bme280_data_to_buffer(&sensor_bme_data, bme_buffer);
        LED_Off(0);
        if(!sdhc_is_mounted()){
            PR_DEBUG("SDHC is not mounted\n");
        }

        // save to file
        LED_On(1); // green LED
        __disable_irq();
        ret = sdhc_appendFile(filename, pAI85Buffer, SAMPLE_HEADER_SIZE+DATA_SIZE*SAMPLE_SIZE + sizeof(sensor_bme_data));
        if(ret) {
            printf("Error in appending to file, retrying: \n");
            ret = sdhc_appendFile(filename, pAI85Buffer, SAMPLE_HEADER_SIZE+DATA_SIZE*SAMPLE_SIZE + sizeof(sensor_bme_data));
            if(ret) {
                printf("Did not succeed in writing data to file \n");
            }
        }
        __enable_irq();
        LED_Off(1);
        // MXC_Delay(SEC(10)); // Let debugger interrupt if needed
        i += 1;
    }
    PR_DEBUG("Total Samples:%d, Total Words: %d \n", sampleCounter, wordCounter);
    while (1);
}


































