/*
; main_functions.cc.
; =========

;------------------------------------------------------------------------
; Author:	Jona Beysens, Jonathan Reymond, Robin Berguerand		The 2022-12-14
; Modifs:
;
; Goal:	Tensorflow lite micro helpr header
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


/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C"
{
#endif

    // Initializes all data needed for the example. The name is important, and needs
    // to be setup() for Arduino compatibility.
    void model_setup(float **input);

    // Runs one iteration of data gathering and inference. This should be called
    // repeatedly from the application code. The name needs to be loop() for Arduino
    // compatibility.
    void model_call(void);

#ifdef __cplusplus
}
#endif

#endif // TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_
