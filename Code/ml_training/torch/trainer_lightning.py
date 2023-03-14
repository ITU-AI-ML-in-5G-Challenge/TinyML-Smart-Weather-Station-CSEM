"""""
 *  \brief     train_lightning.py
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

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch.optim as optim
import numpy as np
import bisect



class SEDWrapper(pl.LightningModule):
    def __init__(self, sed_model, config, dataset):
        super().__init__()
        self.sed_model = sed_model
        self.config = config
        self.dataset = dataset
        self.loss = config['loss_type']
        self.metrics = config['metrics']
        self.optim = config['optim']
        
        
    def evaluate_metric(self, pred, ans):
        acc = accuracy_score(ans, np.argmax(pred, 1))
        return {"acc": acc}
    
    def forward(self, x):
        output_dict = self.sed_model(x)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def inference(self, x):
        self.device_type = next(self.parameters()).device
        self.eval()
        x = torch.from_numpy(x).float().to(self.device_type)
        output_dict = self.sed_model(x, None, True)
        for key in output_dict.keys():
            output_dict[key] = output_dict[key].detach().cpu().numpy()
        return output_dict

    def training_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        pred, _ = self(batch["audio"])
        loss = self.loss(pred, batch["label"])
        self.log("loss", loss, on_epoch= True, prog_bar=True)
        return loss
        
    def training_epoch_end(self, outputs):
        self.dataset.generate_queue()


    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch["audio"])
        return [pred.detach(), batch["label"].detach()]
    
    def validation_epoch_end(self, validation_step_outputs):
        self.device_type = next(self.parameters()).device
        pred = torch.cat([d[0] for d in validation_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in validation_step_outputs], dim = 0)

        if torch.cuda.device_count() > 1:
            gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
            gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
            dist.barrier()

        metric_dict = {
                "acc":0.
            }
        if torch.cuda.device_count() > 1:
            dist.all_gather(gather_pred, pred)
            dist.all_gather(gather_target, target)
            
            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
                gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
                metric_dict = self.evaluate_metric(gather_pred, gather_target)
                print(self.device_type, dist.get_world_size(), metric_dict, flush = True)
        
            self.log("acc", metric_dict["acc"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            dist.barrier()
        else:
            gather_pred = pred.cpu().numpy()
            gather_target = target.cpu().numpy()
            
            metric_dict = self.evaluate_metric(gather_pred, gather_target)
            print(self.device_type, metric_dict, flush = True)
        
            self.log("acc", metric_dict["acc"], on_epoch = True, prog_bar=True, sync_dist=False)
            
        
    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis = 1)
        return new_sample 

    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        preds = []
        # time shifting optimization
        shift_num = 1 # framewise localization cannot allow the time shifting
        
        for i in range(shift_num):
            pred, pred_map = self(batch["audio"])
            preds.append(pred.unsqueeze(0))
            batch["audio"] = self.time_shifting(batch["audio"], shift_len = 100 * (i + 1))
        preds = torch.cat(preds, dim=0)
        pred = preds.mean(dim = 0)
        
        return [pred.detach(), batch["label"].detach()]

    def test_epoch_end(self, test_step_outputs):
        self.device_type = next(self.parameters()).device
        
        self.device_type = next(self.parameters()).device
        pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in test_step_outputs], dim = 0)
        gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
        dist.barrier()
        metric_dict = {
                "acc":0.
            }
        dist.all_gather(gather_pred, pred)
        dist.all_gather(gather_target, target)
        if dist.get_rank() == 0:
            gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
            gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()

            metric_dict = self.evaluate_metric(gather_pred, gather_target)
            print(self.device_type, dist.get_world_size(), metric_dict, flush = True)
            
        self.log("acc", metric_dict["acc"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
        dist.barrier()
    

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.config.learning_rate, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.05, 
        )

        def lr_foo(epoch):       
            if epoch < 3:
                # warm up lr
                lr_scale = self.config.lr_rate[epoch]
            else:
                # warmup schedule
                lr_pos = int(-1 - bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))
                if lr_pos < -3:
                    lr_scale = max(self.config.lr_rate[0] * (0.98 ** epoch), 0.03 )
                else:
                    lr_scale = self.config.lr_rate[lr_pos]
            return lr_scale
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        
        return [optimizer], [scheduler]