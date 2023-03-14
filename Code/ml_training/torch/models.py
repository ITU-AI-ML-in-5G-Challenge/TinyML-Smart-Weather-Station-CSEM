"""""
 *  \brief     models.py
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
import torch
import torch.nn as nn
import torch.nn.functional as F

#from https://github.com/danielajisafe/Audio_WaveForm_Paper_Implementation


def init_weights(m, seed):
   if type(m) == nn.Conv1d or type(m) == nn.Linear: #This initializes all layers that we have at the start.
       torch.manual_seed(seed)
       nn.init.xavier_uniform_(m.weight.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(name, datasets, seed):
    num_outputs = datasets[0].get_num_outputs()
    if name == "M3":
        torch.manual_seed(seed)
        model = M3(num_outputs)
    elif name == "M5":
        torch.manual_seed(seed)
        model = M5(num_outputs)
    elif name == "M11":
        torch.manual_seed(seed)
        model = M11(num_outputs)
    elif name == "M18":
        torch.manual_seed(seed)
        model = M18(num_outputs)
    elif name == "M34":
        torch.manual_seed(seed)
        model = M34(num_outputs)
    elif name == "ACDNet":
        sample_length = datasets[0].shape()[1][1]
        torch.manual_seed(seed)
        model = ACDNetV2(sample_length, num_outputs, datasets[0].fs, None)

    else :
        model = None
    print('number of parameters for',name, count_parameters(model))
    return model

        
class M3(nn.Module):                           # this is m3 architecture
    def __init__(self, num_outputs):
        super(M3, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, 80, 4)  #(in, out=num_filters, filter=kernel size, stride).
        self.bn1 = nn.BatchNorm1d(256)         # this is used to normalize.
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(256, 256, 3)    # by default,the stride is 1 if it is not specified here.
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(4)

        self.avgPool = nn.AdaptiveAvgPool1d(1) # 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, num_outputs)          # this is the output layer.
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.avgPool(x)
        x = self.flatten(x) 
        x = self.fc1(x)                        # this is the output layer, [n,1, 10] i.e 10 probs for each audio files 
        return x

class M5(nn.Module):                           # this is m5 architecture
    def __init__(self, num_outputs):
        super(M5, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)  #(in, out, filter size, stride)
        self.bn1 = nn.BatchNorm1d(128)         #normalize 
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)    #by default,the stride is 1 if it is not specified here.
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        
        self.avgPool = nn.AdaptiveAvgPool1d(1) #insteads of using nn.AvgPool1d(30) (where I need to manually check the dimension that comes in). I use adaptive n flatten
        #the advantage of adaptiveavgpool is that it manually adjust to avoid dimension issues
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, num_outputs)          # this is the output layer.
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        # print(x.shape)
        x = self.avgPool(x)
        x = self.flatten(x)  # replaces permute(0,2,1) with flatten
        x = self.fc1(x)       #output layer ([n,1, 10] i.e 10 probs. for each audio files) 
        return x # we didnt use softmax here becuz we already have that in cross entropy



class M11(nn.Module):                          # this is m11 architecture
    def __init__(self, num_outputs):
        super(M11, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 80, 4)   #(in, out, filter size, stride)
        self.bn1 = nn.BatchNorm1d(64)          # this is used to normalize 
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(64, 64, 3)      # by default, the stride is 1 if it is not specified here.
        self.bn2 = nn.BatchNorm1d(64)
        self.conv2b = nn.Conv1d(64, 64, 3)     # by default, the stride is 1 if it is not specified here.
        self.bn2b = nn.BatchNorm1d(64)
    
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(64, 128, 3)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv3b = nn.Conv1d(128, 128, 3)
        self.bn3b = nn.BatchNorm1d(128)


        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(128, 256, 3)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv4b = nn.Conv1d(256, 256, 3)
        self.bn4b = nn.BatchNorm1d(256)
        self.conv4c = nn.Conv1d(256, 256, 3)
        self.bn4c = nn.BatchNorm1d(256)

        self.pool4 = nn.MaxPool1d(4)
        self.conv5 = nn.Conv1d(256, 512, 3)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv5b = nn.Conv1d(512, 512, 3)
        self.bn5b = nn.BatchNorm1d(512)

        # self.avgPool = nn.AvgPool1d(25)      #replaced with ADaptive + flatten
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, num_outputs)          # this is the output layer.
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv2b(x)
        x = F.relu(self.bn2b(x))

        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv3b(x)
        x = F.relu(self.bn3b(x))

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.conv4b(x)
        x = F.relu(self.bn4b(x))
        x = self.conv4c(x)
        x = F.relu(self.bn4c(x))

        x = self.pool4(x)
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv5b(x)
        x = F.relu(self.bn5b(x))

        x = self.avgPool(x)
        x = self.flatten(x) 
        x = self.fc1(x)                        # this is the output layer. [n,1, 10] i.e 10 probs for each audio files 
        return x




#from https://github.com/mohaimenz/acdnet/
class ACDNetV2(nn.Module):
    def __init__(self, input_length, n_class, fs, ch_conf=None):
        super(ACDNetV2, self).__init__();
        self.input_length = input_length;
        self.ch_config = ch_conf;

        stride1 = 2;
        stride2 = 2;
        channels = 8;
        k_size = (3, 3);
        n_frames = (fs/1000)*10; #No of frames per 10ms

        sfeb_pool_size = int(n_frames/(stride1*stride2));
        # tfeb_pool_size = (2,2);
        if self.ch_config is None:
            self.ch_config = [channels, channels*8, channels*4, channels*8, channels*8, channels*16, channels*16, channels*32, channels*32, channels*64, channels*64, n_class];
        # avg_pool_kernel_size = (1,4) if self.ch_config[1] < 64 else (2,4);
        fcn_no_of_inputs = self.ch_config[-1];
        conv1, bn1 = self.make_layers(1, self.ch_config[0], (1, 9), (1, stride1));
        conv2, bn2 = self.make_layers(self.ch_config[0], self.ch_config[1], (1, 5), (1, stride2));
        conv3, bn3 = self.make_layers(1, self.ch_config[2], k_size, padding=1);
        conv4, bn4 = self.make_layers(self.ch_config[2], self.ch_config[3], k_size, padding=1);
        conv5, bn5 = self.make_layers(self.ch_config[3], self.ch_config[4], k_size, padding=1);
        conv6, bn6 = self.make_layers(self.ch_config[4], self.ch_config[5], k_size, padding=1);
        conv7, bn7 = self.make_layers(self.ch_config[5], self.ch_config[6], k_size, padding=1);
        conv8, bn8 = self.make_layers(self.ch_config[6], self.ch_config[7], k_size, padding=1);
        conv9, bn9 = self.make_layers(self.ch_config[7], self.ch_config[8], k_size, padding=1);
        conv10, bn10 = self.make_layers(self.ch_config[8], self.ch_config[9], k_size, padding=1);
        conv11, bn11 = self.make_layers(self.ch_config[9], self.ch_config[10], k_size, padding=1);
        conv12, bn12 = self.make_layers(self.ch_config[10], self.ch_config[11], (1, 1));
        self.fcn = nn.Linear(fcn_no_of_inputs, n_class);
        nn.init.kaiming_normal_(self.fcn.weight, nonlinearity='sigmoid') # kaiming with sigoid is equivalent to lecun_normal in keras
        self.sfeb = nn.Sequential(
            #Start: Filter bank
            conv1, bn1, nn.ReLU(),\
            conv2, bn2, nn.ReLU(),\
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        );

        tfeb_modules = [];
        self.tfeb_width = int(((self.input_length / fs)*1000)/10); # 10ms frames of audio length in seconds
        tfeb_pool_sizes = self.get_tfeb_pool_sizes(self.ch_config[1], self.tfeb_width);
        p_index = 0;
        for i in [3,4,6,8,10]:
            tfeb_modules.extend([eval('conv{}'.format(i)), eval('bn{}'.format(i)), nn.ReLU()]);

            if i != 3:
                tfeb_modules.extend([eval('conv{}'.format(i+1)), eval('bn{}'.format(i+1)), nn.ReLU()]);

            h, w = tfeb_pool_sizes[p_index];
            if h>1 or w>1:
                tfeb_modules.append(nn.MaxPool2d(kernel_size = (h,w)));
            p_index += 1;

        tfeb_modules.append(nn.Dropout(0.2));
        tfeb_modules.extend([conv12, bn12, nn.ReLU()]);
        h, w = tfeb_pool_sizes[-1];
        if h>1 or w>1:
            tfeb_modules.append(nn.AvgPool2d(kernel_size = (h,w)));
        tfeb_modules.extend([nn.Flatten(), self.fcn]);

        self.tfeb = nn.Sequential(*tfeb_modules);

        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        );

    def forward(self, x):
        l = list(x.size())
        x = x.view([l[0]] + [1] + l[1:])
        x = self.sfeb(x);
        #swapaxes
        x = x.permute((0, 2, 1, 3));
        x = self.tfeb(x);
        y = self.output[0](x);
        return y;


    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels);
        return conv, bn;

    def get_tfeb_pool_sizes(self, con2_ch, width):
        h = self.get_tfeb_pool_size_component(con2_ch);
        w = self.get_tfeb_pool_size_component(width);
        # print(w);
        pool_size = [];
        for  (h1, w1) in zip(h, w):
            pool_size.append((h1, w1));
        return pool_size;

    def get_tfeb_pool_size_component(self, length):
        c = [];
        index = 1;
        while index <= 6:
            if length >= 2:
                if index == 6:
                    c.append(length);
                else:
                    c.append(2);
                    length = length // 2;
            else:
               c.append(1);
            index += 1;
        return c;




class M18(nn.Module):                          # this is m18 architecture
    def __init__(self, num_outputs):
        super(M18, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 80, 4)   # (in, out, filter size, stride)
        self.bn1 = nn.BatchNorm1d(64)          # this is used to normalize. 
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(64, 64, 3)      # by default, the stride is 1 if it is not specified here.
        self.bn2 = nn.BatchNorm1d(64)
        self.conv2b = nn.Conv1d(64, 64, 3)     # by default, the stride is 1 if it is not specified here.
        self.bn2b = nn.BatchNorm1d(64)
        self.conv2c = nn.Conv1d(64, 64, 3)     
        self.bn2c = nn.BatchNorm1d(64)
        self.conv2d = nn.Conv1d(64, 64, 3)     
        self.bn2d = nn.BatchNorm1d(64)
    
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(64, 128, 3)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv3b = nn.Conv1d(128, 128, 3)
        self.bn3b = nn.BatchNorm1d(128)
        self.conv3c = nn.Conv1d(128, 128, 3)
        self.bn3c = nn.BatchNorm1d(128)
        self.conv3d = nn.Conv1d(128, 128, 3)
        self.bn3d = nn.BatchNorm1d(128)


        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(128, 256, 3)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv4b = nn.Conv1d(256, 256, 3)
        self.bn4b = nn.BatchNorm1d(256)
        self.conv4c = nn.Conv1d(256, 256, 3)
        self.bn4c = nn.BatchNorm1d(256)
        self.conv4d = nn.Conv1d(256, 256, 3)
        self.bn4d = nn.BatchNorm1d(256)

        self.pool4 = nn.MaxPool1d(4)
        self.conv5 = nn.Conv1d(256, 512, 3)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv5b = nn.Conv1d(512, 512, 3)
        self.bn5b = nn.BatchNorm1d(512)
        self.conv5c = nn.Conv1d(512, 512, 3)
        self.bn5c = nn.BatchNorm1d(512)
        self.conv5d = nn.Conv1d(512, 512, 3)
        self.bn5d = nn.BatchNorm1d(512)
        
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, num_outputs)           # this is the output layer.
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv2b(x)
        x = F.relu(self.bn2b(x))
        x = self.conv2c(x)
        x = F.relu(self.bn2c(x))
        x = self.conv2d(x)
        x = F.relu(self.bn2d(x))

        x = self.pool2(x)
      
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv3b(x)
        x = F.relu(self.bn3b(x))
        x = self.conv3c(x)
        x = F.relu(self.bn3c(x))
        x = self.conv3d(x)
        x = F.relu(self.bn3d(x))

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.conv4b(x)
        x = F.relu(self.bn4b(x))
        x = self.conv4c(x)
        x = F.relu(self.bn4c(x))
        x = self.conv4d(x)
        x = F.relu(self.bn4d(x))

        x = self.pool4(x)
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv5b(x)
        x = F.relu(self.bn5b(x))
        x = self.conv5c(x)
        x = F.relu(self.bn5c(x))
        x = self.conv5d(x)
        x = F.relu(self.bn5d(x))

        x = self.avgPool(x)
        x = self.flatten(x)
        x = self.fc1(x)                      # this is the output layer. [n,1, 10] i.e 10 probs for each audio files
        return x

def res_upsamp(A,m,n):
  upsample = nn.Upsample(size=(m,n), mode='nearest')
  A = torch.unsqueeze(A, 0)
  A = upsample(A)
  A = A.view(-1,m,n)
  return A

class M34(nn.Module):                          # This is m34 architecture. It is actually the implementation of Resnet34.
    def __init__(self, num_outputs):
        super(M34, self).__init__()
        self.conv1 = nn.Conv1d(1, 48, 80, 4)   # (in, out, filter size, stride)
        self.bn1 = nn.BatchNorm1d(48)          # this is used to normalize. 
        self.pool1 = nn.MaxPool1d(4)

        #upsample residual, pad X via the convolutions

        #X3
        self.conv2a = nn.Conv1d(48, 48, 3, padding = 1)   # by default, the stride is 1 if it is not specified here.
        self.bn2a = nn.BatchNorm1d(48)
        self.conv2a2 = nn.Conv1d(48, 48, 3, padding = 1)  
        self.bn2a2 = nn.BatchNorm1d(48)
        self.bn2a3 = nn.BatchNorm1d(48)
        self.pool2 = nn.MaxPool1d(4)

        self.conv2b = nn.Conv1d(48, 48, 3, padding = 1)   
        self.bn2b = nn.BatchNorm1d(48)
        self.conv2b2 = nn.Conv1d(48, 48, 3, padding=1)    
        self.bn2b2 = nn.BatchNorm1d(48)
        self.bn2b3 = nn.BatchNorm1d(48)

        self.conv2c = nn.Conv1d(48, 48, 3, padding=1)     
        self.bn2c = nn.BatchNorm1d(48)
        self.conv2c2 = nn.Conv1d(48, 48, 3, padding=1)   
        self.bn2c2 = nn.BatchNorm1d(48)
        self.bn2c3 = nn.BatchNorm1d(48)

        #X4
        self.conv3a = nn.Conv1d(48, 96, 3)     # upsampling on the 1st two convulutions, padding on the rest.
        self.bn3a = nn.BatchNorm1d(96)
        self.conv3a2 = nn.Conv1d(96, 96,3)     # by default, the stride is 1 if it is not specified here.
        self.bn3a2 = nn.BatchNorm1d(96)
        self.bn3a3 = nn.BatchNorm1d(96)
        self.pool3 = nn.MaxPool1d(4)

        self.conv3b = nn.Conv1d(96, 96, 3, padding=1)  
        self.bn3b = nn.BatchNorm1d(96)
        self.conv3b2 = nn.Conv1d(96, 96, 3, padding=1)    
        self.bn3b2 = nn.BatchNorm1d(96)
        self.bn3b3 = nn.BatchNorm1d(96)

        self.conv3c = nn.Conv1d(96, 96, 3, padding=1)    
        self.bn3c = nn.BatchNorm1d(96)
        self.conv3c2 = nn.Conv1d(96, 96, 3, padding=1)   
        self.bn3c2 = nn.BatchNorm1d(96)
        self.bn3c3 = nn.BatchNorm1d(96)

        self.conv3d = nn.Conv1d(96, 96, 3, padding=1)   
        self.bn3d = nn.BatchNorm1d(96)
        self.conv3d2 = nn.Conv1d(96, 96, 3, padding=1)  
        self.bn3d2 = nn.BatchNorm1d(96)
        self.bn3d3 = nn.BatchNorm1d(96)
        
        #X6
        self.conv4a = nn.Conv1d(96, 192, 3)    # by default, the stride is 1 if it is not specified here.
        self.bn4a = nn.BatchNorm1d(192)
        self.conv4a2 = nn.Conv1d(192, 192, 3) 
        self.bn4a2 = nn.BatchNorm1d(192)
        self.bn4a3 = nn.BatchNorm1d(192)

        self.conv4b = nn.Conv1d(192, 192, 3, padding=1) 
        self.bn4b = nn.BatchNorm1d(192)
        self.conv4b2 = nn.Conv1d(192, 192, 3, padding=1) 
        self.bn4b2 = nn.BatchNorm1d(192)
        self.bn4b3 = nn.BatchNorm1d(192)

        self.conv4c = nn.Conv1d(192, 192, 3, padding=1)  
        self.bn4c = nn.BatchNorm1d(192)
        self.conv4c2 = nn.Conv1d(192, 192, 3, padding=1)
        self.bn4c2 = nn.BatchNorm1d(192)
        self.bn4c3 = nn.BatchNorm1d(192)

        self.conv4d = nn.Conv1d(192, 192, 3, padding=1)
        self.bn4d = nn.BatchNorm1d(192)
        self.conv4d2 = nn.Conv1d(192, 192, 3, padding=1)
        self.bn4d2 = nn.BatchNorm1d(192)
        self.bn4d3 = nn.BatchNorm1d(192)
        self.pool4 = nn.MaxPool1d(4)

        self.conv4e = nn.Conv1d(192, 192, 3, padding=1) 
        self.bn4e = nn.BatchNorm1d(192)
        self.conv4e2 = nn.Conv1d(192, 192, 3, padding=1) 
        self.bn4e2 = nn.BatchNorm1d(192)
        self.bn4e3 = nn.BatchNorm1d(192)

        self.conv4f = nn.Conv1d(192, 192, 3, padding=1) 
        self.bn4f = nn.BatchNorm1d(192)
        self.conv4f2 = nn.Conv1d(192, 192, 3, padding=1) 
        self.bn4f2 = nn.BatchNorm1d(192)
        self.bn4f3 = nn.BatchNorm1d(192)
        
        #X3
        self.conv5a = nn.Conv1d(192, 384, 3) 
        self.bn5a = nn.BatchNorm1d(384)
        self.conv5a2 = nn.Conv1d(384, 384, 3) 
        self.bn5a2 = nn.BatchNorm1d(384)
        self.bn5a3 = nn.BatchNorm1d(384)

        self.conv5b = nn.Conv1d(384, 384, 3, padding=1)  
        self.bn5b = nn.BatchNorm1d(384)
        self.conv5b2 = nn.Conv1d(384, 384, 3, padding=1) 
        self.bn5b2 = nn.BatchNorm1d(384)
        self.bn5b3 = nn.BatchNorm1d(384)

        self.conv5c = nn.Conv1d(384, 384, 3, padding=1)  
        self.bn5c = nn.BatchNorm1d(384)
        self.conv5c2 = nn.Conv1d(384, 384, 3, padding=1) 
        self.bn5c2 = nn.BatchNorm1d(384)
        self.bn5c3 = nn.BatchNorm1d(384)
        
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(384, num_outputs)         # this is the output layer.
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)


        #X3
        residual = x                          # update once for the 3 blocks
        x = F.relu(self.bn2a(self.conv2a(x)))
        # x += residual
        x = self.conv2a2(x)
        x = self.bn2a2(x)
        #add, skip connection
        
        x += residual
        x = F.relu(self.bn2a3(x))

        x = self.conv2b(x)
        x = F.relu(self.bn2b(x))
        x = self.conv2b2(x)
        x = self.bn2b2(x)
        #add
        x += residual
        x = F.relu(self.bn2b3(x))

        x = self.conv2c(x)
        x = F.relu(self.bn2c(x))
        x = self.conv2c2(x)
        x = self.bn2c2(x)
        #add
        x += residual
        x = F.relu(self.bn2c3(x))

        x = self.pool2(x)
        
        #X4
        residual = x                         # update once for the 4 blocks
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3a2(self.conv3a2(x))
        #add, skip connection
        residual = res_upsamp(residual,x.shape[1],x.shape[2])  # we pad at the beginning of every block which is after the first two(2) convolutions.
        x += residual
        x = F.relu(self.bn3a3(x))

  
        x = self.conv3b(x)
        x = F.relu(self.bn3b(x))
        x = self.conv3b2(x)
        x = self.bn3b2(x)
        #add, skip connection
        x += residual
        x = F.relu(self.bn3b3(x))

        x = self.conv3c(x)
        x = F.relu(self.bn3c(x))
        x = self.conv3c2(x)
        x = self.bn3c2(x)
        #add, skip connection
        x += residual
        x = F.relu(self.bn3c3(x))

        x = self.conv3d(x)
        x = F.relu(self.bn3d(x))
        x = self.conv3d2(x)
        x = self.bn3d2(x)
        #add, skip connection
        x += residual
        x = F.relu(self.bn3d3(x))
        x = self.pool3(x)

        #x6
        residual = x                    # update once for the 6 blocks
        x = self.conv4a(x)
        x = F.relu(self.bn4a(x))
        x = self.conv4a2(x)
        x = self.bn4a2(x)
        #add, skip connection

        residual = res_upsamp(residual,x.shape[1],x.shape[2]) # we pad at the beginning of every block which is after the first two(2) convolutions.

        x += residual
        x = F.relu(self.bn4a3(x))
        x = self.conv4b(x)
        x = F.relu(self.bn4b(x))
        x = self.conv4b2(x)
        x = self.bn4b2(x)
        #add, skip connection
        x += residual
        x = F.relu(self.bn4b3(x))

        x = self.conv4c(x)
        x = F.relu(self.bn4c(x))
        x = self.conv4c2(x)
        x = self.bn4c2(x)
        #add, skip connection
        x += residual
        x = F.relu(self.bn4c3(x))

        x = self.conv4d(x)
        x = F.relu(self.bn4d(x))
        x = self.conv4d2(x)
        x = self.bn4d2(x)
        #add, skip connection
        x += residual
        x = F.relu(self.bn4d3(x))

        x = self.conv4e(x)
        x = F.relu(self.bn4e(x))
        x = self.conv4e2(x)
        x = self.bn4e2(x)
        #add, skip connection
        x += residual
        x = F.relu(self.bn4e3(x))

        x = self.conv4f(x)
        x = F.relu(self.bn4f(x))
        x = self.conv4f2(x)
        x = self.bn4f2(x)
        #add, skip connection
        x += residual
        x = F.relu(self.bn4f3(x))
        x = self.pool4(x)

        #X3
        residual = x                           # update once for the 3 blocks
        x = self.conv5a(x)
        x = F.relu(self.bn5a(x))
        x = self.conv5a2(x)
        x = self.bn5a2(x)
        #add, skip connection
        residual = res_upsamp(residual,x.shape[1],x.shape[2])# we pad at the beginning of every block which is after the first two(2) convolutions.
        x += residual
        x = F.relu(self.bn5a3(x))

        x = self.conv5b(x)
        x = F.relu(self.bn5b(x))
        x = self.conv5b2(x)
        x = self.bn5b2(x)
        #add, skip connection
        x += residual
        x = F.relu(self.bn5b3(x))

        x = self.conv5c(x)
        x = F.relu(self.bn5c(x))
        x = self.conv5c2(x)
        x = self.bn5c2(x)
        #add, skip connection
        x += residual
        x = F.relu(self.bn5c3(x))


      
        x = self.avgPool(x)
        x = self.flatten(x) 
        x = self.fc1(x)                   # this is the output layer, [n,1, 10] i.e 10 probs for each audio files 
        return x
