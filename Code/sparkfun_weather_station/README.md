# TinyML Challenge 2022 - Code for Sparkfun weather station
Here are the instructions to mount and deploy the weather station based on the microbit car.
## Material needed :
- Sparkfun weather station kit ([link](https://www.sparkfun.com/products/16274))
- SD card formatted either with FAT16 or FAT32
- Microbit card ([link](https://www.sparkfun.com/products/17287))
- Battery ([link](https://www.adafruit.com/?q=lithium+ion+battery&sort=BestMatch))

## Mounting and deploying
To mount the weather station, please follow the instructions on this [link](https://learn.sparkfun.com/tutorials/weather-meter-hookup-guide/all).  

Afterwards, install the [microbit application](https://makecode.microbit.org/offline-app). Then drag the .hex file into the opened application.  
To flash the card, just connect it to your computer and click the 'Download' button in the microbit app.  
Then connect your weather station with the card like the ![picture below](setup_weather_station.jpg). To start recording, press the button 'A'.

- Make sure that the SD card has being inserted into the SparkFun OpenLog.
- To start/stop recording, press the button 'A', the cross will disappear/appear
- Make sure to have stopped recording before removing the SD card

## Further
If you need more information or need to add the other sensors available in the weather station kit, to get more insights you could follow the [weather station experiment guide](https://learn.sparkfun.com/tutorials/microclimate-kit-experiment-guide).
