"""""
 *  \brief     onnx_to_tflite.py
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

#from https://siliconlabs.github.io/mltk/mltk/tutorials/onnx_to_tflite.html

import os
from mltk.utils.path import create_tempdir
import numpy as np
from dataset_vent import AudioDatasetVent
import onnxsim
import onnx
# from torch.utils.tensorboard import SummaryWriter
# from openvino

from openvino.tools.mo import main as mo_main
# import sys
# sys.path.insert(1, '/local/user/jrn/tinyml-challenge-2022/ml_training/openvino/tools/mo/openvino/tools/mo')

# from openvino.tools.mo import main as mo_main
from onnx_tf.backend import prepare
from mltk.utils.shell_cmd import run_shell_cmd
import sys
import os

ONNX_MODEL_PATH = 'results/ventilator/M3.onnx'

WORKING_DIR = "results/ventilator/temp"
assert os.path.exists(ONNX_MODEL_PATH)
os.makedirs(WORKING_DIR, exist_ok=True)

MODEL_NAME = os.path.basename(ONNX_MODEL_PATH)[:-len('.onnx')]

print(f'ONNX_MODEL_PATH = {ONNX_MODEL_PATH}')
print(f'MODEL_NAME = {MODEL_NAME}')
print(f'WORKING_DIR = {WORKING_DIR}')

#####################
## get dataset ######
#####################
# def representative_data_gen(dataset, num_samples=100, seed=1):
#     size = len(dataset)
#     arr = np.arange(size)
#     np.random.seed(seed)
#     np.random.shuffle(arr)
#     for idx in arr[:num_samples]:
#         yield [dataset[idx][0].unsqueeze(0)]
# dataset = AudioDatasetVent("data/office/ventilator/", resample_rate=None, normalize=False)

#Simply onnx model
simplified_onnx_model, success = onnxsim.simplify(ONNX_MODEL_PATH)
assert success, 'Failed to simplify the ONNX model. You may have to skip this step'
simplified_onnx_model_path =  f'{WORKING_DIR}/{MODEL_NAME}.simplified.onnx'
print(f'Generating {simplified_onnx_model_path} ...')
onnx.save(simplified_onnx_model, simplified_onnx_model_path)

# writer = SummaryWriter('/local/user/jrn/tinyml-challenge-2022/results/ventilator')
# writer.add_onnx_graph('/local/user/jrn/tinyml-challenge-2022/results/ventilator')
print('done')

# Load the ONNX model , here not simplified
onnx_model = onnx.load(ONNX_MODEL_PATH)
tf_rep = prepare(onnx_model)

# Get the input tensor shape
input_tensor = tf_rep.signatures[tf_rep.inputs[0]]
input_shape = input_tensor.shape
input_shape_str = '[' + ','.join([str(x) for x in input_shape]) + ']'

openvino_out_dir = f'{WORKING_DIR}/openvino'
os.makedirs(openvino_out_dir, exist_ok=True)

print(f'Generating openvino at: {openvino_out_dir}')
cmd = [ 
    sys.executable, mo_main.__file__, 
    '--input_model', simplified_onnx_model_path,
    '--input_shape', input_shape_str,
    '--output_dir', openvino_out_dir,
    '--data_type', 'FP32'

]
retcode, retmsg = run_shell_cmd(cmd,  outfile=sys.stdout)
assert retcode == 0, 'Failed to do conversion' 

print('done')
