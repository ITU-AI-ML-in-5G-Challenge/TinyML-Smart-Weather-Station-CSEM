"""""
 *  \brief     tf_learning_hts.py
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

import sys
import numpy as np
import torch 
import torch.nn as nn
import pytorch_lightning as pl

import config_htsat as config
print(config.htsat_spec_size)

sys.path.insert(0, '/local')
sys.path.insert(0, '/local/user')
sys.path.insert(0, '/local/user/jrn/HTS-Audio-Transformer')
sys.path.insert(0, '/local/user/jrn/HTS-Audio-Transformer/model')

from model.htsat import HTSAT_Swin_Transformer


def get_hts_model(num_classes, num_layers_trainable=0):
    sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        )



    if config.resume_checkpoint is not None:
            ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
            ckpt["state_dict"].pop("sed_model.head.weight")
            ckpt["state_dict"].pop("sed_model.head.bias")
            sed_model.load_state_dict(ckpt["state_dict"], strict=False)
            
    for param in sed_model.parameters():
        param.requires_grad = False

    # change classifier
    sed_model.num_classes = num_classes
    SF = sed_model.spec_size // (2 ** (len(sed_model.depths) - 1)) // sed_model.patch_stride[0] // sed_model.freq_ratio
    sed_model.tscam_conv = nn.Conv2d(
                    in_channels = sed_model.num_features,
                    out_channels = num_classes,
                    kernel_size = (SF,3),
                    padding = (0,1)
                )
    sed_model.head = nn.Linear(num_classes, num_classes)
    
        
    print()
    NUM_HEAD_PARAMS = 4
    params = list(sed_model.parameters())
    for param in params[len(params)- num_layers_trainable - NUM_HEAD_PARAMS:]:
        param.requires_grad = True
    
    return sed_model

