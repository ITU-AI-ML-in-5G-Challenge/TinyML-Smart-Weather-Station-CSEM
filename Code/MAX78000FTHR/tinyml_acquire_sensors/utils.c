/*******************************************************************************
* Copyright (C) 2020-2021 Maxim Integrated Products, Inc., All Rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*
******************************************************************************/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mxc_device.h"
#include "board.h"
#include "mxc_delay.h"
#include "uart.h"
#include "rtc.h"
#include "utils.h"
#include "i2c_regs.h"
#include "i2c.h"
#include "trng.h"

/***************************** VARIABLES *************************************/


/************************    PUBLIC FUNCTIONS  *******************************/

uint32_t utils_get_time_ms(void)
{
    int sec;
    double subsec;
    uint32_t ms;

    subsec = MXC_RTC_GetSubSecond() / 4096.0;
    sec = MXC_RTC_GetSecond();

    ms = (sec * 1000) + (int)(subsec * 1000);

    // printf("subsec: %d \n", subsec);

    return ms;
}

uint16_t utils_ms_to_hour(uint32_t timestamp_ms)
{
    uint16_t timestamp_hour;
    timestamp_hour = (uint16_t) (timestamp_ms / 1000 / 60 / 60);

    return timestamp_hour;
}

uint16_t utils_ms_to_minutes(uint32_t timestamp_ms)
{
    uint16_t timestamp_min;
    timestamp_min = (uint16_t) (timestamp_ms / 1000 / 60);

    return timestamp_min;
}

uint32_t utils_ms_to_seconds(uint32_t timestamp_ms)
{
    uint32_t timestamp_sec;
    timestamp_sec = (timestamp_ms / 1000);

    return timestamp_sec;
}

uint16_t utils_get_TRNG_sample(void) {
    uint16_t result = 0;
    uint8_t var_rnd_no[2] = {0};
    int num_bytes = 2;
    
    memset(var_rnd_no, 0, sizeof(var_rnd_no));
    
    MXC_TRNG_Init();
    MXC_TRNG_Random(var_rnd_no, num_bytes);
    MXC_TRNG_Shutdown();

    // printf("Test: %d \n", var_rnd_no[0]);
    
    result = (var_rnd_no[1] << 8) + var_rnd_no[0];

    return result;
}


static void utils_send_byte(mxc_uart_regs_t* uart, uint8_t value)
{
    while (MXC_UART_WriteCharacter(uart, value) == E_OVERFLOW) { }
}

void utils_send_bytes(mxc_uart_regs_t* uart, uint8_t* ptr, int length)
{
    int i;

    for (i = 0; i < length; i++) {
        utils_send_byte(uart, ptr[i]);
    }
}

int8_t init_i2c_with_print(){
    int8_t error = 0;
    // Master : MXC_I2C1
     error = MXC_I2C_Init(MXC_I2C1, 1, 0);
        if (error != E_NO_ERROR) {
        printf("-->Failed master\n");
        return FALSE;
        }
        else {
        printf("\n-->I2C Master Initialization Complete\n");
        }
}


#pragma GCC optimize ("-O0")

#define DEBUG_COMPORT   MXC_UART0

/***************************** VARIABLES *************************************/


