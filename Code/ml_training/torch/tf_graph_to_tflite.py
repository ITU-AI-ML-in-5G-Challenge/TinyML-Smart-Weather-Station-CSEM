"""""
 *  \brief     tf_graph_to_tflite.py
 *  \author    Jonathan Reymond
 *  \version   1.0
 *  \date      2023-02-14
 *  \pre       None
 *  \copyright (c) 2022 CSEM
 *
 *   CSEM S.A.
 *   Jaquet-Droz 1
 *   CH-2000 Neuchâtel
 *   http://www.csem.ch
 *
 *
 *   THIS PROGRAM IS CONFIDENTIAL AND CANNOT BE DISTRIBUTED
 *   WITHOUT THE CSEM PRIOR WRITTEN AGREEMENT.
 *
 *   CSEM is the owner of this source code and is authorised to use, to modify
 *   and to keep confidential all new modifications of this code.
 *
 """

from cgi import test
from cmath import exp
from webbrowser import get
import tensorflow as tf
import pathlib
import numpy as np
from dataset_vent import AudioDatasetVent
import torch
from torch.nn import functional as F
# import utils
# if(torch.cuda.is_available() ):
#     device = utils.get_a_free_gpu()
#     torch.cuda.empty_cache()
# else:
#     device = torch.device("cpu")



def representative_data_gen(dataset, num_samples=100, seed=1):
    size = len(dataset)
    arr = np.arange(size)
    np.random.seed(seed)
    np.random.shuffle(arr)
    for idx in arr[:num_samples]:
        yield [dataset[idx][0].unsqueeze(0)]


def get_tflite_interpreter(export_dir, with_optimization, with_quantization, store_path=None, dataset=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    if with_optimization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if with_quantization:
        converter.representative_dataset = lambda : representative_data_gen(dataset)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 
    tflite_model = converter.convert()
    if store_path is not None:
        tflite_model_file = pathlib.Path(store_path)
        tflite_model_file.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]['index']
    output_details = interpreter.get_output_details()[0]['index']
    return {'interpreter': interpreter, 'input':input_details, 'output':output_details}

def get_tflite_output(interpreter_dict, test_sample):
    interpreter = interpreter_dict['interpreter']
    interpreter.set_tensor(interpreter_dict['input'], test_sample)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter_dict['output'])

export_dir = "results/ventilator/M3.pb"
dataset = AudioDatasetVent("data/office/ventilator/", resample_rate=None, normalize=False)
tflite_interp = get_tflite_interpreter(export_dir, False, False, store_path="results/ventilator/M3.tflite")
tflite_interp_opt = get_tflite_interpreter(export_dir, True, False, store_path="results/ventilator/M3_opt.tflite")
tflite_interp_quant = get_tflite_interpreter(export_dir, with_optimization=True, with_quantization=True, 
                                            dataset=dataset, store_path="results/ventilator/M3_quant.tflite")



torch_model_path = '/local/user/jrn/tinyml-challenge-2022/results/ventilator/M3.pt'
torch_model = torch.load(torch_model_path)

imported = tf.saved_model.load(export_dir)
print(type(imported))
infer_tf_model = imported.signatures["serving_default"]
print(type(infer_tf_model))

arr = np.arange(len(dataset))
np.random.seed(2)
np.random.shuffle(arr)
print(len(dataset))
test_list = [dataset[idx] for idx in arr[:100]]

def is_correct(label, output):
    pred = np.argmax(output, 1)[0]
    return 1 if pred == label else 0

print('=========================================')
counter_torch = 0
counter_tf = 0
counter_tflite = 0
counter_tflite_opt = 0
counter_tflite_quant = 0

for i, (test_sample, label) in enumerate(test_list):
    if i % 50 == 0 :
        print(i)
    test_sample = test_sample.unsqueeze(0)
    torch_model.eval()
    output_torch = torch_model(test_sample.cuda()).cpu().detach().numpy()

    test_sample = tf.convert_to_tensor(test_sample.numpy())
    output_tf =  infer_tf_model(test_sample)['output'].numpy()
    output_tflite = get_tflite_output(tflite_interp, test_sample)
    output_tflite_opt =  get_tflite_output(tflite_interp_opt, test_sample)
    output_tflite_quant =  get_tflite_output(tflite_interp_quant, test_sample)

    counter_torch += is_correct(label, output_torch)
    counter_tf += is_correct(label, output_tf)
    counter_tflite += is_correct(label, output_tflite)
    counter_tflite_opt += is_correct(label, output_tflite_opt)
    counter_tflite_quant += is_correct(label, output_tflite_quant)


print(counter_torch/len(test_list))
print(counter_tf/len(test_list))
print(counter_tflite/len(test_list))
print(counter_tflite_opt/len(test_list))
print(counter_tflite_quant/len(test_list))



    
