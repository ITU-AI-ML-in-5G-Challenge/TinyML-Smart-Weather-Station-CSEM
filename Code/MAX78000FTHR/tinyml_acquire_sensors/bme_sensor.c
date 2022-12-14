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


#include "bme_sensor.h"

static mxc_i2c_req_t i2c_req;
struct bme280_dev dev;

/*Calculate the minimum delay required between consecutive measurement based upon the sensor enabled
*  and the oversampling configuration. */
uint32_t req_delay;


int8_t init_bme_sensor(struct bme280_data* sensor_bme280_data, bool init_i2c){
    PR_INFO("\n*** I2C & BME280 Init ***\n");

    sensor_bme280_data->humidity = 0;
    sensor_bme280_data->pressure = 0;
    sensor_bme280_data->temperature = 0;

    int8_t error = 0;
    //Setup the I2CM
    if(init_i2c){
        error = init_i2c_with_print();
        MXC_I2C_SetFrequency(I2C_MASTER, I2C_FREQ);
    }
    
    //init i2c properties that will not change
    i2c_req.i2c = I2C_MASTER;
    i2c_req.addr = I2C_SLAVE_ADDR;
    i2c_req.restart = 0;
    i2c_req.callback = NULL;

    //init dev
    uint8_t dev_addr = I2C_SLAVE_ADDR;
    dev.intf_ptr = &dev_addr;
    dev.intf = BME280_I2C_INTF;
    dev.read = user_i2c_read;
    dev.write = user_i2c_write;
    dev.delay_us = user_delay_us;

    int8_t rslt = bme280_init(&dev);
    if (rslt != BME280_OK) {
        printf("Error with initialisation BME280_init phase\n");
        return rslt;
    }
	uint8_t settings_sel;

	/* Recommended mode of operation: Weather monitoring */
	dev.settings.osr_h = BME280_OVERSAMPLING_1X; // for humidity oversampling is not needed
	dev.settings.osr_p = BME280_OVERSAMPLING_1X;
	dev.settings.osr_t = BME280_OVERSAMPLING_1X;
	dev.settings.filter = BME280_FILTER_COEFF_OFF;

	settings_sel = BME280_OSR_PRESS_SEL;
	settings_sel |= BME280_OSR_TEMP_SEL;
	settings_sel |= BME280_OSR_HUM_SEL;
	settings_sel |= BME280_STANDBY_SEL;
	settings_sel |= BME280_FILTER_SEL;

	rslt = bme280_set_sensor_settings(settings_sel, &dev);

    req_delay = bme280_cal_meas_delay(&(&dev)->settings);
    return rslt;
}


void user_delay_us(uint32_t period, void *intf_ptr)
{
    /*
     * Return control or wait,
     * for a period amount of microseconds
     */
    MXC_Delay(period);
}

int8_t user_i2c_read(uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr)
{
    int8_t rslt; /* Return 0 for Success, non-zero for failure */
    uint8_t buf[1] = { reg_addr };

    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = reg_data;
    i2c_req.rx_len = len;

    rslt = MXC_I2C_MasterTransaction(&i2c_req);
    /*
     * The parameter intf_ptr can be used as a variable to store the I2C address of the device
     */

    /*
     * Data on the bus should be like
     * |------------+---------------------|
     * | I2C action | Data                |
     * |------------+---------------------|
     * | Start      | -                   |
     * | Write      | (reg_addr)          |
     * | Stop       | -                   |
     * | Start      | -                   |
     * | Read       | (reg_data[0])       |
     * | Read       | (....)              |
     * | Read       | (reg_data[len - 1]) |
     * | Stop       | -                   |
     * |------------+---------------------|
     */

    if (rslt != BME280_OK) {
        printf("Error with reading from BME280, error code %d\n", rslt);
    }

    return rslt;
}

int8_t user_i2c_write(uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr)
{
    int8_t rslt; /* Return 0 for Success, non-zero for failure */
    uint8_t buf[I2C_MAX_LENGTH] = {0};

    buf[0] = reg_addr;
    for(uint8_t i = 0; i< len; i++) {
        buf[i+1] = *(reg_data+i);
    }
    i2c_req.tx_buf = buf;
    i2c_req.tx_len = 1+len; // 1 for address, len for data
    i2c_req.rx_len = 0;

    rslt = MXC_I2C_MasterTransaction(&i2c_req);

    /*
     * The parameter intf_ptr can be used as a variable to store the I2C address of the device
     */

    /*
     * Data on the bus should be like
     * |------------+---------------------|
     * | I2C action | Data                |
     * |------------+---------------------|
     * | Start      | -                   |
     * | Write      | (reg_addr)          |
     * | Write      | (reg_data[0])       |
     * | Write      | (....)              |
     * | Write      | (reg_data[len - 1]) |
     * | Stop       | -                   |
     * |------------+---------------------|
     */

    if (rslt != BME280_OK) {
        printf("Error with writing to BME280, error code %d\n", rslt);
    }
    return rslt;
}



void print_sensor_data(struct bme280_data *comp_data)
{
#ifdef BME280_FLOAT_ENABLE
        printf("Temperature: %0.2f degrees, Pressure: %0.2f Pa, Relative humidity %0.2f %%\r\n",comp_data->temperature, comp_data->pressure, comp_data->humidity);
#else
        printf("%ld, %ld, %ld\r\n",comp_data->temperature, comp_data->pressure, comp_data->humidity);
#endif
}

int8_t record_one_bme_measure(struct bme280_data* sensor_bme280_data){
    int8_t rslt;
    rslt = bme280_set_sensor_mode(BME280_FORCED_MODE, &dev);
    if (rslt != BME280_OK) {
        printf("Error when setting sensor mode of BME280, error code %d\n", rslt);
    }
    rslt = bme280_get_sensor_data(BME280_ALL, sensor_bme280_data, &dev);
    if (rslt != BME280_OK) {
        printf("Error when getting data of BME280, error code %d\n", rslt);
    }
    return rslt;
}


int8_t stream_sensor_data_forced_mode_weather()
{
	int8_t rslt;
	struct bme280_data comp_data;

    printf("Temperature, Pressure, Humidity\r\n");
    /* Continuously stream sensor data */
    while (1) {
        rslt = record_one_bme_measure(&comp_data);
        print_sensor_data(&comp_data);

        MXC_Delay(MXC_DELAY_SEC(1));
    }
	return rslt;
}

void fill_double_buffer(uint8_t* buffer, int* counter, double value){
    uint8_t* value_arr = (char *) &value;
    for(int i = 0; i < 8; i++){
        buffer[*counter + i] = value_arr[i];
    }
    *counter += + 8;
}

double doubleFromBytes(uint8_t *buffer, int index_p) {
    double result;
    // legal and portable C provided buffer contains a valid double representation
    memcpy(&result, buffer + index_p * sizeof(double), sizeof(double));
    return result;
}



void bme280_data_to_buffer(struct bme280_data* sensor_bme_data, uint8_t* bme_buffer){
    int counter = 0;
    fill_double_buffer(bme_buffer, &counter, sensor_bme_data->humidity);
    fill_double_buffer(bme_buffer, &counter, sensor_bme_data->pressure);
    fill_double_buffer(bme_buffer, &counter, sensor_bme_data->temperature);

}
