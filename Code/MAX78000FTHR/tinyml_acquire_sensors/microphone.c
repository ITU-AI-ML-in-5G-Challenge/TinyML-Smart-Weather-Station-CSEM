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


volatile uint8_t i2s_flag = 0;
int32_t i2s_rx_buffer[I2S_RX_BUFFER_SIZE];
int16_t Max, Min;
uint16_t avg = 0;
int32_t tot_usec = -100000;
uint8_t pChunkBuff[DATA_SIZE*CHUNK];
uint16_t ai85Counter = 0;



void mic_init(){
    int error = 0;

    /* Enable microphone power on Feather board */
    error = Microphone_Power(POWER_ON);
    if(error != E_NO_ERROR){
        PR_DEBUG("error microphone power on: %d\n", error);
    }
    // // Get ticks based off of microseconds
    mxc_wut_cfg_t cfg;
    uint32_t ticks;

    error = MXC_WUT_GetTicks(WUT_USEC, MXC_WUT_UNIT_MICROSEC, &ticks);
    if(error != E_NO_ERROR){
        PR_DEBUG("error MXC_WUT_GetTicks: %d\n", error);
    }
    // config structure for one shot timer to trigger in a number of ticks
    cfg.mode = MXC_WUT_MODE_CONTINUOUS;
    cfg.cmp_cnt = ticks;
    // Init WUT
    MXC_WUT_Init(MXC_WUT_PRES_1);
    //Config WUT
    MXC_WUT_Config(&cfg);

    MXC_LP_EnableWUTAlarmWakeup();
    NVIC_EnableIRQ(WUT_IRQn);

    /* initialize I2S interface to Mic */
    I2SInit();

    MXC_WUT_Enable();  // Start WUT
}
/************************************************************************************/
void i2s_isr(void)
{
    i2s_flag = 1;
    /* Clear I2S interrupt flag */
    MXC_I2S_ClearFlags(MXC_F_I2S_INTFL_RX_THD_CH0);
}
/************************************************************************************/

void recordOneWord(uint8_t* pAI85Buffer, uint16_t* wordCounter, uint32_t* sampleCounter) {
    /* Read from Mic driver to get one word of samples */  
    uint8_t ret = 0;
    /* reset counters */
    ai85Counter = 0;

    uint32_t timestamp = utils_get_time_ms();
    PR_DEBUG("Timetamp: %d ms \n", timestamp);

    /* Write the timestamp (in ms) as a header, followed by the microphone data */
    pAI85Buffer[ai85Counter++] = (timestamp >> 24) & 0XFF;
    pAI85Buffer[ai85Counter++] = (timestamp >> 16) & 0XFF;
    pAI85Buffer[ai85Counter++] = (timestamp >> 8) & 0XFF;
    pAI85Buffer[ai85Counter++] = timestamp & 0XFF;

    /* Check if header has the correct size */
    if(ai85Counter != SAMPLE_HEADER_SIZE) {
        fail();
    }

    while (ai85Counter < DATA_SIZE* SAMPLE_SIZE) {
        /* Read from Mic driver to get CHUNK worth of samples, otherwise next sample*/
        while(MicReadChunk(pChunkBuff, &avg) == 0); // chunk buffer is not yet completely filled
        
        *sampleCounter += DATA_SIZE * CHUNK; // we are sure that CHUNK samples are read

        /* add sample, rearrange buffer */
        ret = AddBuffer(pChunkBuff, pAI85Buffer, DATA_SIZE * CHUNK, DATA_SIZE * SAMPLE_SIZE,
                            ai85Counter);

        /* increment number of stored samples */
        ai85Counter += DATA_SIZE * CHUNK;
        if (ai85Counter % 100 == 0)
        {
            printf("Collected chunk : %d/%d\n",ai85Counter,DATA_SIZE* SAMPLE_SIZE);
        }  
    }

    if (ai85Counter >= DATA_SIZE * SAMPLE_SIZE) {
        PR_DEBUG("Word: %d is collected with: %d samples and avg last: %d, min all: %d, max all: %d\n",*wordCounter, ai85Counter, avg, Min, Max);

        /* new word */
        *wordCounter = *wordCounter + 1;

        /* sanity check, last transpose should have returned 1, as enough samples should have already been added */
        if (ret != 1) {
            PR_DEBUG("ERROR: AddBuffer incomplete!\n");
            fail();
        }
        // PR_DEBUG("----------------------------------------- \n");
        Max = 0;
        Min = 0;
        //------------------------------------------------------------
    }

    PR_DEBUG("\n*** End of record of word ***\n\n");
}

/* **************************************************************************** */
#ifdef ENABLE_MIC_PROCESSING
void I2SInit()
{
    mxc_i2s_req_t req;
    int32_t err;

    PR_INFO("\n*** I2S & Mic Init ***\n");
    /* Initialize I2S RX buffer */
    memset(i2s_rx_buffer, 0, sizeof(i2s_rx_buffer));
    /* Configure I2S interface parameters */
    req.wordSize    = MXC_I2S_DATASIZE_WORD;
    req.sampleSize  = MXC_I2S_SAMPLESIZE_THIRTYTWO;
    req.justify     = MXC_I2S_MSB_JUSTIFY;
    req.wsPolarity  = MXC_I2S_POL_NORMAL;
    req.channelMode = MXC_I2S_INTERNAL_SCK_WS_0;
    /* Get only left channel data from on-board microphone. Right channel samples are zeros */
    req.stereoMode  = MXC_I2S_MONO_LEFT_CH;
    req.bitOrder    = MXC_I2S_MSB_FIRST;
    /* I2S clock = PT freq / (2*(req.clkdiv + 1)) */
    /* I2S sample rate = I2S clock/64 = 16kHz */
    req.clkdiv      = 5;
    req.rawData     = NULL;
    req.txData      = NULL;
    req.rxData      = i2s_rx_buffer;
    req.length      = I2S_RX_BUFFER_SIZE;

    if ((err = MXC_I2S_Init(&req)) != E_NO_ERROR) {
        PR_DEBUG("\nError in I2S_Init: %d\n", err);

        while (1);
    }
    /* Set I2S RX FIFO threshold to generate interrupt */
    MXC_I2S_SetRXThreshold(4);

#ifndef WUT_ENABLE
    NVIC_SetVector(I2S_IRQn, i2s_isr);
    NVIC_EnableIRQ(I2S_IRQn);
    /* Enable RX FIFO Threshold Interrupt */
    MXC_I2S_EnableInt(MXC_F_I2S_INTEN_RX_THD_CH0);
#endif

    MXC_I2S_RXEnable();
    __enable_irq();
}
#endif

/* **************************************************************************** */
void fail(void)
{
    PR_DEBUG("\n*** FAIL ***\n\n");
    while (1);
}

/* **************************************************************************** */
uint8_t AddBuffer(uint8_t* pIn, uint8_t* pOut, uint16_t inSize,
                     uint16_t outSize, uint16_t counter)
{
    uint16_t total = counter; // number of samples already in the buffer
    // pIn has always length with multiple of 128, so no need to check whether buffer will be overfull
    for (int i = 0; i < inSize; i++) {
        /* place sample in correct output location */
        pOut[counter+i] = pIn[i];

        total++;
    }
    // PR_DEBUG("total: %d\n",total);
    if (total >= outSize) {
        // PR_DEBUG("output buffer full\n");
        return 1;
    }
    else {
        return 0;
    }
}

/* **************************************************************************** */
uint8_t MicReadChunk(uint8_t* pBuff, uint16_t* avg)
{
    static uint16_t chunkCount = 0;
    static uint32_t index = 0;
#ifdef LOSSLESS_ACQUISITION
    static uint32_t sum = 0;
    uint32_t sample = 0;
    uint32_t temp = 0;
#else
    int32_t sample = 0;
    int16_t temp = 0;
    static uint16_t sum = 0;
#endif
    uint32_t rx_size = 0;

    /* sample not ready */
    if (!i2s_flag) {
        *avg = 0;
        return 0;
    }
    /* Clear flag */
    i2s_flag = 0;
    /* Read number of samples in I2S RX FIFO */
    rx_size = MXC_I2S->dmach0 >> MXC_F_I2S_DMACH0_RX_LVL_POS;
//  PR_DEBUG("%d ", rx_size);

    /* read until fifo is empty or enough samples are collected */
    while ((rx_size--) && (chunkCount < DATA_SIZE * CHUNK)) {
        /* Read microphone sample from I2S FIFO */
        sample = (uint32_t)MXC_I2S->fifoch0;
        /* The actual value is 18 MSB of 32-bit word */
        temp = sample >> 14;
#ifndef LOSSLESS_ACQUISITION
        sample = (int16_t)temp;
#endif //LOSSLESS_ACQUISITION
        /* Discard first 10k samples due to microphone charging cap effect */
        if (index++ < 10000) {
            continue;
        }     
        /* absolute for averaging */
        if (sample >= 0) {
            sum += sample;
        }
        else {
            sum -= sample;
        }

        /* What happens is the following:
        1) conversion from 32 bit to 8 bit: upper 24 bits are just cut off and lost
        2) data type is just a matter of interpretation. Suppose the data is 0x81, then in int8 it will represent -127, 
        whereas in uint8 it will represent 129
        Conclusion: we do as if it is a unsigned integer, but in fact it is still a signed integer

        Nevertheless I think there is an error in the code. When sample = -14000, then output = 38, which does not seem correct
        Test code:    
        int32_t sample = -14000;
        int8_t sample_conv = (int8_t) (sample * 4 / 256);
        printf("sample: 0x%x / %d -> 0x%x / %d", sample, sample, sample_conv,sample_conv);
        */
#ifdef LOSSLESS_ACQUISITION
        sample = temp>>2;
        // printf("Sample: %lu\n", (uint32_t)sample);
        // pBuff[chunkCount] = (sample >> 16) & 0XFF;
        // printf("chunkCount0: %ld\n", pBuff[chunkCount]);
        // chunkCount++;
        pBuff[chunkCount] = (sample >> 8) & 0XFF;
        // printf("chunkCount1: %ld\n", pBuff[chunkCount]);
        chunkCount++;
        pBuff[chunkCount] = sample & 0XFF;
        // printf("chunkCount2: %ld\n", pBuff[chunkCount]);
        chunkCount++;
        temp = sample;
#else
        /* Convert to 8 bit unsigned */
        pBuff[chunkCount] = (uint8_t)((sample) * SAMPLE_SCALE_FACTOR / 256);
        temp = (int8_t)pBuff[chunkCount];
        chunkCount++;
#endif //LOSSLESS_ACQUISITION
        /* record max and min */
        if (temp > Max) {
            Max = temp;
        }
        if (temp < Min) {
            Min = temp;
        }
    }

    /* if not enough samples, return 0 */
    if (chunkCount < DATA_SIZE * CHUNK) {
        *avg = 0;
        return 0;
    }
    /* enough samples are collected, calculate average and return 1 */
    *avg = ((uint16_t)(sum / CHUNK));
    chunkCount = 0;
    sum = 0;
    return 1;
}

/************************************************************************************/
void configure_clock_source(void) {
        switch (CLOCK_SOURCE) {
    case 0:
        MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);
        MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
        MXC_GCR->pm &= ~MXC_F_GCR_PM_IPO_PD;  // enable IPO during sleep
        break;

    case 1:
        MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_ISO);
        MXC_SYS_Clock_Select(MXC_SYS_CLOCK_ISO);
        MXC_GCR->pm &= ~MXC_F_GCR_PM_ISO_PD;  // enable ISO during sleep
        break;

    case 2:
        MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IBRO);
        MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IBRO);
        MXC_GCR->pm &= ~MXC_F_GCR_PM_IBRO_PD;  // enable IBRO during sleep
        break;

    default:
        printf("UNKNOWN CLOCK SOURCE \n");

        while (1);
    }
    SystemCoreClockUpdate();
}


#ifdef WUT_ENABLE
void WUT_IRQHandler()
{
    i2s_flag = 1;
    MXC_WUT_IntClear();

    tot_usec += WUT_USEC ;
}
#endif


