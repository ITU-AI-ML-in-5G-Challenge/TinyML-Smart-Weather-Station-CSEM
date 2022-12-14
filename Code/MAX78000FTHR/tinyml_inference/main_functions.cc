/*
; main_functions.cc.
; =========

;------------------------------------------------------------------------
; Author:	Jona Beysens, Jonathan Reymond, Robin Berguerand		The 2022-12-14
; Modifs:
;
; Goal:	Tensorflow lite micro helper
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "main_functions.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"

namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;
  // int inference_count = 0;
  tflite::MicroInterpreter *interpreter = nullptr;

  const int tensor_arena_size = 85 * 1024;
} // namespace

void model_setup(float32_t **input_model)
{
  tflite::InitializeTarget();
  model = tflite::GetModel(final_model);
  // model = tflite::GetModel(m5);
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  // Register callback for printing debug log
  RegisterDebugLogCallback(debug_log_printf);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  // Setup MicroOps
  static tflite::MicroMutableOpResolver<10> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddQuantize() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddStridedSlice() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddConv2D() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddMean() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddDequantize() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddExpandDims() != kTfLiteOk)
  {
    return;
  }
  // Malloc tensor Area
  uint8_t *tensor_arena = (uint8_t *)malloc(tensor_arena_size * sizeof(uint8_t));
  if (tensor_arena == NULL)
  {
    printf("Not enough Memory to allocate tensor_arena\n");
  }

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, tensor_arena_size, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  *input_model = input->data.f;
  output_rain = interpreter->output(0);
  output_wind = interpreter->output(0);
}
// The name of this function is important for Arduino compatibility.
void model_call()
{
  //Invoke the model
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(x));
    return;
  }
  size_t max_ind_rain = 0;
  float32_t max_val_rain = output->output_rain.f[0];
  for (size_t j = 1; j < 3; j++)
  {
    if (output->data.f[j] > max_val_rain)
    {
      max_val_rain = output->output_rain.f[j];
      max_ind_rain = j;
    }
  }
  size_t max_ind_wind = 0;
  float32_t max_val_wind = output->output_rain.f[0];
  for (size_t j = 1; j < 3; j++)
  {
    if (output->data.f[j] > max_val_wind)
    {
      max_val_wind = output->output_rain.f[j];
      max_ind_wind = j;
    }
  }
  //Print the results
  printf("(%lld s) Result are: ", tick2 / 1000000);
  switch (max_ind_rain)
  {
  case 0:
    printf(KSYST, "No Rain");
    break;
  case 1:
    printf(KSYST, "Little rain");
    break;
  case 2:
    printf(KSYST, "Rain");
    break;
  default:
    break;
  }
  printf(", ");
  switch (max_ind_wind)
  {
  case 0:
    printf(KSYST, "No Wind\n");
    break;
  case 1:
    printf(KSYST, "Little Wind\n");
    break;
  case 2:
    printf(KSYST, "Wind\n");
    break;
  default:
    break;
  }
}
