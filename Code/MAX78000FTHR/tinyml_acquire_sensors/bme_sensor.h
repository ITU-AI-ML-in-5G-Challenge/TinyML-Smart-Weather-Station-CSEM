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


#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mxc_device.h"
#include "mxc_delay.h"
#include "nvic_table.h"
#include "i2c_regs.h"
#include "i2c.h"
#include "dma.h"
#include "BME280_driver/bme280.h"
#include "utils.h"


#define I2C_MASTER      MXC_I2C1
#define I2C_FREQ        100000
// This example may become unreliable at I2C frequencies above 100kHz.
// This is only an issue in the loopback configuration, where the I2C block is
// connected to itself.
#define I2C_SLAVE_ADDR  (0x76)
#define I2C_MAX_LENGTH  16 // max data length to send in a I2C command

#define PR_DEBUG(fmt, args...)  if(1) printf(fmt, ##args )
#define PR_INFO(fmt, args...)  if(1) printf(fmt, ##args )


int8_t init_bme_sensor(struct bme280_data* sensor_bme_data, bool init_i2c);
void user_delay_us(uint32_t period, void *intf_ptr);
int8_t user_i2c_read(uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr);
int8_t user_i2c_write(uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr);
int8_t record_one_bme_measure(struct bme280_data* sensor_bme280_data);
int8_t stream_sensor_data_forced_mode_weather();
void print_sensor_data(struct bme280_data *comp_data);
void bme280_data_to_buffer(struct bme280_data* sensor_bme_data, uint8_t* bme_buffer);



