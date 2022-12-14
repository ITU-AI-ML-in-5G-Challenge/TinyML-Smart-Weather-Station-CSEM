/*
; main.c.
; =========

;------------------------------------------------------------------------
; Author:	Jona Beysens, Jonathan Reymond, Robin Berguerand	The 2022-12-14
; Modifs:
;
; Goal:	Tensorflow lite micro model
;
;   (c) 1992-2022, Edo. Franzi
;   --------------------------
;
;   CSEM S.A.
;   Jaquet-Droz 1
;   CH-2000 Neuchâtel
;   http://www.csem.ch
;
;   ____________________/\\\______/\\\______/\\\_
;    ________________/\\\\\\\__/\\\\\\\__/\\\\\\\_
;     _______________\/////\\\_\/////\\\_\/////\\\_
;      __/\\\____/\\\_____\/\\\_____\/\\\_____\/\\\_
;       _\/\\\___\/\\\_____\/\\\_____\/\\\_____\/\\\_
;        _\//\\\\\\\\\______\/\\\_____\/\\\_____\/\\\_
;         __\/////////_______\///______\///______\///_
;
;   THIS PROGRAM IS CONFIDENTIAL AND CANNOT BE DISTRIBUTED
;   WITHOUT THE CSEM PRIOR WRITTEN AGREEMENT.
;
;   u111 is an optimised branch of uKOS-III package.
;   CSEM is the owner of this branch and is authorised to use, to modify
;   and to keep confidential all new adaptations of this branch.
;------------------------------------------------------------------------
*/
#include "main_functions.h"
#include "max20303/max20303.h"
#include <BME280/BME280.h>
#define MODEL_INPUT 8192
#define DATA_SIZE 4						   //  Float32_t
#define SAMPLE_SIZE MODEL_INPUT *DATA_SIZE // size of input vector for CNN, keep it multiple of 128
#define MAX_SIZE_DATA 8
#define ONE_MINUTE 60000 //in mS

static struct bme280_dev dev;

/**
 * @brief Print sensor data
 *
 * @param comp_data Structure containing information about temperature, pressure and humidity
 */
static void print_sensor_data(struct bme280_data *comp_data)
{
#ifdef BME280_FLOAT_ENABLE
	printf("[BME280] Temperature: %ld.%02ld degrees, Pressure: %ld.%02ld Pa, Relative humidity: %ld.%02ld %%\r\n", FLOAT_2(comp_data->temperature), FLOAT_2(comp_data->pressure), FLOAT_2(comp_data->humidity));
#else
	printf("[BME280] Temperature: %ld, Pressure: %ld, Relative humidity: %ld\r\n", comp_data->temperature, comp_data->pressure, comp_data->humidity);
#endif
}
void Acquire_data(float32_t *input)
{
	uint32_t i = 0;
	uint32_t temp_size = -1;
	uint8_t temp_data[MAX_SIZE_DATA];
	float32_t divider = (float32_t)32768;
	max20303_mic_power(TRUE);
	// Wait for the Microphone to wake up and collect 10000 fake sample.
	RTC_SET(ONE_SEC);
	MXC_LP_EnterSleepMode();
	while (i < 10000)
	{
		temp_size = MAX_SIZE_DATA;
		mic_read(&temp_data, &temp_size, KWAITINFINITY);
		i += temp_size / 2;
	}
	// Collect real sample
	i = 0;
	while (i < MODEL_INPUT)
	{
		temp_size = MAX_SIZE_DATA;

		mic_read(&temp_data, &temp_size, KWAITINFINITY);
		for (size_t j = 0; j < temp_size; j += 2)
		{
			int16_t tempi = (int16_t)(temp_data[j] << 8) | (temp_data[j + 1]);
			float32_t data = (float32_t)tempi;
			input[i] = (data) / divider;
			i++;
		}
	}
	max20303_mic_power(FALSE);

}
void Periph_init(void)
{

	// Init the microphone
	max20303_init();
	max20303_mic_power(TRUE);
	int32_t ret;
	struct bme280_data comp_data;

	if ((ret = bme280_setup(&dev)) == BME280_OK)
	{
	iotx_printf(KSYST, "[BME280] Successfully set up.\n");
	}
	else
	{
		iotx_printf(KSYST, "[BME280] Failure with setup.\n");
		exit(EXIT_FAILURE);
	}
	/* Configure the BME280 device to operate in weather mode */
	bme280_configure_weather(&dev);

	/* Configure I2S interface parameters for the microphone */
	cnfI2sx_t configureI2S = {
		.oChannelMode = KINTERNAL_SCK_WS_1,
		.oStereoMode = KLEFT,
		.oWordSize = KWORD,
		.oJustify = KMSB,
		.oBitOrder = KMSB,
		.oWsPolarity = KNORMALPOLARITY,
		.oSampleSize = KSPSTHIRTYTWO,
		.oClkdiv = 5,
		.oRxThreshold = 8};
	i2s0_configure(&configureI2S);
}

void main(uint32_t argc, const char_t *argv[])
{
	Periph_init();
	float *input;
	model_setup(&input);
	int rslt = BME280_OK;
	struct bme280_data comp_data;
	while (TRUE)
	{
		Acquire_data(input);
		model_call();
		rslt = bme280_get_single_measurement(&dev, &comp_data);
		if (rslt != BME280_OK)
		{
			printf("[BME280] Error with reading measurement\n");
			exit(EXIT_FAILURE);
		}
		print_sensor_data(&comp_data);
		//Sleep for 10 Minutes;
		RTC_SET(10* ONE_MINUTE);
		MXC_LP_EnterMicroPowerMode()
	}
}