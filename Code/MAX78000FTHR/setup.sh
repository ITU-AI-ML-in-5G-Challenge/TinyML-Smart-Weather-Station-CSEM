#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


export MAXIM_PATH=${DIR}/MAX78000_SDK
export PATH=$PATH:/opt/uKOS/cross/gcc-arm-none-eabi-9-2019-q4-major/bin:/opt/uKOS/cross/openocd-0.11.0/max78000/bin/

echo "Set the environment path for this TinyML project"
