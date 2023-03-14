"""""
 *  \brief     torch_to_tf_graph.py
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

import io
import numpy as np

from onnx_tf.backend import prepare
import onnx
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import models
from utils import get_a_free_gpu
from torch.utils.tensorboard import SummaryWriter
import onnxsim
# import tensorflow as tf



if(torch.cuda.is_available() ):
    device = get_a_free_gpu()
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

# device = torch.device('cpu')

store_base = 'results/ventilator/M3.pt'
store_output = 'results/ventilator/M3.onnx'
m3_model = torch.load(store_base, map_location=device)

m3_model.eval()

batch_size = 1    # just a random number
x = torch.randn(batch_size, 1, 32000, requires_grad=False).to(device)

writer = SummaryWriter('/local/user/jrn/tinyml-challenge-2022/results/ventilator')
# writer.add_graph(m3_model, x)

torch_out = m3_model(x)
# Export the model
torch.onnx.export(m3_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  store_output,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                #   training=torch.onnx.TrainingMode.TRAINING,
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})




onnx_model = onnx.load(store_output)

onnx_model, success = onnxsim.simplify(store_output)
assert success, 'Failed to simplify the ONNX model. You may have to skip this step'
simplified_onnx_model_path =  f'{store_output}.simplified.onnx'
print(f'Generating {simplified_onnx_model_path} ...')
onnx.save(onnx_model, simplified_onnx_model_path)

# writer.add_onnx_graph(store_output)
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph("results/ventilator/M3_no_grad_last.pb")  # export the model

# tf_model = tf.keras.models.load_model("results/ventilator/saved_model.pb")

# print(tf_model.summary())
print('done')
