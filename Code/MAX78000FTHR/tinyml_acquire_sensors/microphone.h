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


#ifndef __MICROPHONE_H__
#define __MICROPHONE_H__

/* **** Includes **** */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "mxc_sys.h"
#include "fcr_regs.h"
#include "icc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "nvic_table.h"
#include "i2s_regs.h"
#include "board.h"
#include "mxc.h"
#include "i2s.h"
#include "tmr.h"
#include "dma.h"
#include "led.h"
#include "pb.h"
#include "sdhc.h"
#include "utils.h"



/* ********************* */
/* **** Definitions **** */
/* ********************* */
#define VERSION   "1.0.0 (28/07/22)"
#define CLOCK_SOURCE    0   // 0: IPO,  1: ISO, 2: IBRO
#define SLEEP_MODE      0   // 0: no sleep,  1: sleep,   2:deepsleep(LPM)
#define WUT_ENABLE          // enables WUT timer
#define WUT_USEC    380     // continuous WUT duration close to I2S polling time in usec
//#define ENERGY            // if enabled, turn off LED2, toggle LED1 for 10sec for energy measurements on Power monitor (System Power)

#if SLEEP_MODE == 2   // need WakeUp Timer (WUT) for deepsleep (LPM)
#ifndef WUT_ENABLE
#define WUT_ENABLE
#endif
#endif

/* Enable/Disable Features */
//#define ENABLE_CLASSIFICATION_DISPLAY  // enables printing classification result
// #define ENABLE_SILENCE_DETECTION         // Starts collecting only after avg > THRESHOLD_HIGH, otherwise starts from first sample
#undef EIGHT_BIT_SAMPLES                 // samples from Mic or Test vectors are eight bit, otherwise 16-bit
#define ENABLE_MIC_PROCESSING            // enables capturing Mic, otherwise a header file Test vector is used as sample data

/* keep following unchanged */
#define SAMPLE_SIZE         16384   // size of input vector for CNN, keep it multiple of 128
#define CHUNK               128     // number of data points to read at a time and average for threshold, keep multiple of 128
#ifdef LOSSLESS_ACQUISITION
#define DATA_SIZE         2       // 18 bits
#else
#define DATA_SIZE         1       // 8 bits
#endif
#define TRANSPOSE_WIDTH     128     // width of 2d data model to be used for transpose
#define NUM_OUTPUTS         21      // number of classes
#define I2S_RX_BUFFER_SIZE  64      // I2S buffer size
#define TFT_BUFF_SIZE       50      // TFT buffer size
/*-----------------------------*/

/* Adjustables */
#ifdef ENABLE_MIC_PROCESSING
#define SAMPLE_SCALE_FACTOR         4       // multiplies 16-bit samples by this scale factor before converting to 8-bit
#define THRESHOLD_HIGH              350     // voice detection threshold to find beginning of a keyword
#define THRESHOLD_LOW               100     // voice detection threshold to find end of a keyword
#define SILENCE_COUNTER_THRESHOLD   20      // [>20] number of back to back CHUNK periods with avg < THRESHOLD_LOW to declare the end of a word
#define PREAMBLE_SIZE               30*CHUNK// how many samples before beginning of a keyword to include
#define INFERENCE_THRESHOLD         49      // min probability (0-100) to accept an inference

#define SAMPLE_HEADER_SIZE          4       // Size of header for every sample (containing timestamp of sample)
#endif

#define MAX_WORDS_ACQ               24 * 60 * 60      // Max number of words in a single acquisition (in seconds, as every word has a duration of 1 second)

#define PR_DEBUG(fmt, args...)  if(1) printf(fmt, ##args )
#define PR_INFO(fmt, args...)  if(1) printf(fmt, ##args )

/* ***************** */
/* **** Globals **** */
/* ***************** */

void i2s_isr(void);
void    fail(void);

void recordOneWord(uint8_t* pAI85Buffer, uint16_t* wordCounter, uint32_t* sampleCounter);

uint8_t MicReadChunk(uint8_t* pBuff, uint16_t* avg);
uint8_t AddBuffer(uint8_t* pIn, uint8_t* pOut, uint16_t inSize,
                     uint16_t outSize, uint16_t counter);
void    I2SInit();
void    configure_clock_source(void);
void mic_init();

#ifdef WUT_ENABLE
void WUT_IRQHandler();
#endif

#endif