# TinyML Challenge 2022 - Code for MAX78000FTHR

## Introduction
This folder contains the firmware code for the MAX78000 board, for a selection of applications. It contains the MAX78000 SDK as git submodule. 
## Preparation

- get SDK for the MAX78000 from github and add it as git submodule to this repository:
```
git submodule add https://github.com/MaximIntegratedAI/MAX78000_SDK.git bare_metal/MAX78000_SDK
```
## Installation
- To flash and debug the board, you need openocd specifically tailored for the MAX78000. The repo can be found here: https://github.com/MaximIntegratedMicros/openocd. To run openocd on macOS, you need to build it from source.
- Add the GCC compiler to the PATH to be able to compile the examples from Maxim (this is done for you by the `setup.sh` script, as explained below):
    ```
    export PATH="/opt/uKOS/cross/gcc-arm-none-eabi-9-2019-q4-major/bin:$PATH" 
    ```


## Minimal working example
- Set the environment variables by running `source setup.sh`. 
- Go to the application of interest. If the application is located in the Examples folder of the MAX78000 SDK, then move the file `jtag_dap.cfg` to the root folder of the example of interest (I2S in this case:  `MAX78000FTHR/tinyml_acquire_sensors` ).
- Make sure to configure to use the FTHR board by setting `BOARD=FTHR_RevA` in the Makefile of the example. 
- Run following commands to compile and flash the code on the board (make sure that the PATH variable consists of `/opt/uKOS/cross/openocd-0.11.0/max78000/bin/`):
    ```
    make clean && make
    openocd-max78000 -f jtag_dap.cfg
    ```
- Open a serial terminal in CoolTerm and configure it as follows:
    - baudrate: 115200
    - databits: 8
    - parity: none
    - stop bit: 1
    - Key emulation: CR+LF (for interactive commands)
- Connect the serial terminal to the device and inspect the messages. Enjoy!


## Python scripts
- To run the Python scripts, you should install the required packages in `Utility/requirements.txt`. I used `pyenv` for managing different Python versions and a virtual environment within pyenv to create an environment specifically for this project. The used commands are: 
    ```
    Go to the root folder of this repository
    pyenv local 3.10.1
    pyenv virtualenv tinymlchallenge2022
    pyenv activate tinymlchallenge2022
    pip install -r requirements.txt
    ```



## Troubleshooting
- If you have problems to send a command over UART to the board, make sure that the serial terminal (e.g., CoolTerm) on your PC is in the "CR+LF" key emulation mode. 
- To analyze the behaviour of the MAX78000 over a longer time, you can connect the board to a Raspberry PI and save the UART console to a text file. To do so, login on a RPi and execute following command (option `-S` gives the session a name and option `-L` activates the logging):
    ```
    screen -S max78000 -L /dev/ttyACM0 115200
    ```
    This will output the UART console to the file `screenlog.0`. To stop the recording, run:
    ```
    screen -X -S max78000 quit
    ```
