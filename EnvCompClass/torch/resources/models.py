import torch;
import torch.nn as nn;
import numpy as np;
import random;
from tensorflow import keras
from torchsummary import summary

#Reproducibility
seed = 42;
random.seed(seed);
np.random.seed(seed);
torch.manual_seed(seed);
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed);
torch.backends.cudnn.deterministic = True;
torch.backends.cudnn.benchmark = False;
###########################################

class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SEACDNetV2(nn.Module):
    def __init__(self, input_length, n_class, sr, ch_conf=None):
        super(SEACDNetV2, self).__init__();
        self.input_length = input_length;
        self.ch_config = ch_conf;

        stride1 = 2;
        stride2 = 2;
        channels = 8;
        k_size = (3, 3);
        n_frames = (sr/1000)*10; #No of frames per 10ms

        sfeb_pool_size = int(n_frames/(stride1*stride2));
        # tfeb_pool_size = (2,2);
        if self.ch_config is None:
            self.ch_config = [channels, channels*8, channels*4, channels*8, channels*8, channels*16, channels*16, channels*32, channels*32, channels*64, channels*64, n_class];
        # avg_pool_kernel_size = (1,4) if self.ch_config[1] < 64 else (2,4);
        fcn_no_of_inputs = self.ch_config[-1];
        conv1, bn1 = self.make_layers(1, self.ch_config[0], (1, 9), (1, stride1));
        conv2, bn2 = self.make_layers(self.ch_config[0], self.ch_config[1], (1, 5), (1, stride2));
        perm = nn.Conv2d(1,3,1,1,False)
        conv3, bn3 = self.make_layers(1, self.ch_config[2], k_size, padding=1);
        # max_pool1 = nn.MaxPool2d()
        conv4, bn4 = self.make_layers(self.ch_config[2], self.ch_config[3], k_size, padding=1);
        se4 = SE_Block(self.ch_config[3],16)
        conv5, bn5 = self.make_layers(self.ch_config[3], self.ch_config[4], k_size, padding=1);
        conv6, bn6 = self.make_layers(self.ch_config[4], self.ch_config[5], k_size, padding=1);
        se6 = SE_Block(self.ch_config[5],16)
        conv7, bn7 = self.make_layers(self.ch_config[5], self.ch_config[6], k_size, padding=1);
        conv8, bn8 = self.make_layers(self.ch_config[6], self.ch_config[7], k_size, padding=1);
        se8 = SE_Block(self.ch_config[7],16)
        conv9, bn9 = self.make_layers(self.ch_config[7], self.ch_config[8], k_size, padding=1);
        conv10, bn10 = self.make_layers(self.ch_config[8], self.ch_config[9], k_size, padding=1);
        se10 = SE_Block(self.ch_config[9],16)
        conv11, bn11 = self.make_layers(self.ch_config[9], self.ch_config[10], k_size, padding = 1);
        # conv12, bn12 = self.make_layers(self.ch_config[10], self.ch_config[11], (1, 1));
        GAP = nn.AdaptiveAvgPool2d(1)
        fcn_1 = nn.Linear(self.ch_config[10], int(self.ch_config[10]/2))
        nn.init.kaiming_normal_(fcn_1.weight, nonlinearity='sigmoid') # kaiming with sigoid is equivalent to lecun_normal in keras
        fcn_2 = nn.Linear(int(self.ch_config[10]/2), int(self.ch_config[10]/4))
        nn.init.kaiming_normal_(fcn_2.weight, nonlinearity='sigmoid') # kaiming with sigoid is equivalent to lecun_normal in keras
        fcn = nn.Linear(int(self.ch_config[10]/4), n_class);
        nn.init.kaiming_normal_(fcn.weight, nonlinearity='sigmoid') # kaiming with sigoid is equivalent to lecun_normal in keras
        

        self.sfeb = nn.Sequential(
            #Start: Filter bank
            conv1, bn1, nn.ReLU(),\
            conv2, bn2, nn.ReLU(),\
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        );
        
        tfeb_modules = [];
        self.tfeb_width = int(((self.input_length / sr)*1000)/10); # 10ms frames of audio length in seconds
        tfeb_pool_sizes = self.get_tfeb_pool_sizes(self.ch_config[1], self.tfeb_width);
        p_index = 0;
        for i in [3,4,6,8,10]:
            # tfeb_modules.extend([eval('conv{}'.format(i)), eval('bn{}'.format(i)), nn.ReLU()]);

            if i == 3:
                tfeb_modules.extend([eval('conv{}'.format(i)), eval('bn{}'.format(i)), nn.ReLU()]);
            else:
                tfeb_modules.extend([eval('conv{}'.format(i)),eval('se{}'.format(i)), eval('bn{}'.format(i)), nn.ReLU()]);
            
            if i != 3:
                tfeb_modules.extend([eval('conv{}'.format(i+1)), eval('bn{}'.format(i+1)), nn.ReLU()]);

            h, w = tfeb_pool_sizes[p_index];
            if h>1 or w>1:
                tfeb_modules.append(nn.MaxPool2d(kernel_size = (h,w)));
            p_index += 1;

        tfeb_modules.append(nn.Dropout(0.2));
        # tfeb_modules.extend([conv12, bn12, nn.ReLU()]);
        h, w = tfeb_pool_sizes[-1];
        # if h>1 or w>1:
        #     tfeb_modules.append(nn.AvgPool2d(kernel_size = (h,w-1)));
        tfeb_modules.extend([GAP, nn.Flatten()]);
        tfeb_modules.extend([fcn_1]);

        tfeb_modules.extend([fcn_2])
        tfeb_modules.extend([fcn])
        self.tfeb = nn.Sequential(*tfeb_modules);

        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        );

    def forward(self, x):
        x = self.sfeb(x);
        #swapaxes
        x = x.permute((0, 2, 1, 3));
        x = self.tfeb(x)
        x = self.output[0](x);
        return x;

    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels);
        return conv, bn;
    # def make_se_block(self,in_channels,ratio):

    #     squeeze = nn.AdaptiveAvgPool2d(1)
    #     excitation = nn.Sequential(
    #         nn.Linear(in_channels,in_channels//ratio,bias = False)
    #         nn.ReLU(inplace=True)
    #         nn.Linear(in_channels // ratio,in_channels,bias = False)
    #     )
    def get_tfeb_pool_sizes(self, con2_ch, width):
        h = self.get_tfeb_pool_size_component(con2_ch);
        w = self.get_tfeb_pool_size_component(width);
        # print(w);
        pool_size = [];
        for  (h1, w1) in zip(h, w):
            pool_size.append((h1, w1));
        return pool_size;

    def get_tfeb_pool_size_component(self, length):
        # print(length);
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

def GetSEACDNetModel(input_len=22050, nclass=50, sr=22050, channel_config=None):
    net = SEACDNetV2(input_len, nclass, sr, ch_conf=channel_config);
    
    if torch.cuda.is_available():
        net.cuda()
    summary(net,input_size = (1,1,input_len))
    # quit()
    return net;

class SE_Block1d(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=True),
            nn.ReLU(),
            nn.Linear(c // r, c, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.ch_config = [32,64,96]
        conv1 = nn.Conv1d(in_channels = 1,out_channels = 32,kernel_size = 7,stride = 1)
        bn1   = nn.BatchNorm1d(32)
        conv2 = nn.Conv1d(in_channels = 32,out_channels = 64,kernel_size = 7,stride = 1)
        bn2   = nn.BatchNorm1d(64)
        conv3 = nn.Conv1d(in_channels = 64,out_channels = 128,kernel_size = 7,stride = 1) 
        bn3   = nn.BatchNorm1d(128)  
        conv4 = nn.Conv1d(in_channels = 128,out_channels = 256,kernel_size = 7,stride = 1) 
        bn4   = nn.BatchNorm1d(256)  
        maxpool = nn.MaxPool1d(kernel_size = 4)
        global_avg_pool = nn.AvgPool1d(256)
        se_1 = SE_Block1d(128,8)
        se_2 = SE_Block1d(256,16)
        relu = nn.ReLU()
        dropout = nn.Dropout(0.2)
        dense_1 = nn.Linear(256,128)
        dense_2 = nn.Linear(128,64)
        classifier = nn.Linear(64,50)
        self.cpd1 = nn.Sequential(
            conv1,
            bn1,
            relu,
            maxpool,
        )
        self.cpd2 = nn.Sequential(
            conv2,
            bn2,
            relu,
            maxpool,
            
        )
        self.cpd3 = nn.Sequential(
            conv3,
            bn3,
            relu,
            se_1,
            maxpool,
        )
        self.cpd4 = nn.Sequential(
            conv4,
            bn4,
            relu,
            se_2,
            maxpool,
            
        )
        self.fc = nn.Sequential(           
            dense_1,
            relu,
            dense_2,
            relu
        )
        self.cls = nn.Sequential(
            classifier
        )
        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        );
    def forward(self,x):
        x = self.cpd1(x)
        x = self.cpd2(x)
        x = self.cpd3(x)
        x = self.cpd4(x)
        x = nn.AdaptiveAvgPool1d(1)(x)
        x = torch.reshape(x,(x.shape[0],x.shape[1]))
        
        # x = nn.Flatten()(x)
        x = self.fc(x)
        x= self.cls(x)
        y = self.output[0](x)
        return y 

def GetCNN1DModel():
    net = CNN1D();
    if torch.cuda.is_available():
        net.cuda()
    # summary(net,input_size = (-1,1,22050))
    return net 


class Autoencoder(nn.Module):
    def __init__(self):
      super(Autoencoder,self).__init__()
      conv1 = nn.Conv1d(in_channels = 1,out_channels = 32,kernel_size = 7,stride = 1)
      conv2 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 7, stride = 1)
      conv3 = nn.Conv1d(in_channels = 64, out_channels = 96, kernel_size = 7, stride = 1)
      convt1 = nn.ConvTranspose1d(in_channels = 1, out_channels = 96, kernel_size = 7, stride = 2,output_padding = 1)
      convt2 = nn.ConvTranspose1d(in_channels = 96, out_channels = 64, kernel_size = 7, stride = 2,output_padding = 1)
      convt3 = nn.ConvTranspose1d(in_channels = 64, out_channels = 32, kernel_size = 7, stride = 2,output_padding = 1)
      se = SE_Block1d(96,8)
      tanh = nn.Tanh() 
      dropout = nn.Dropout(0.1)
      avg = nn.AvgPool1d(kernel_size = 2)
      dense = nn.Conv1d(in_channels = 96,out_channels = 1,kernel_size = 1,stride = 1)
      final = nn.Conv1d(in_channels = 1,out_channels = 1, kernel_size = 1, stride = 1)
      
      self.mod1 = nn.Sequential(
          conv1,
          tanh,
          avg
      )
      self.mod2 = nn.Sequential(
          conv2,
          tanh,
          avg
          
      )
      self.mod3 = nn.Sequential(
          conv3,
          tanh,
          se,
          avg
          
      )
      self.bottleneck = nn.Sequential(
          dense,
          tanh
      )
      self.modt1 = nn.Sequential(
          convt1,
          se,
          tanh
          
          
      )
      self.modt2 = nn.Sequential(
          convt2,
          tanh,
          
      )
      self.modt3 = nn.Sequential(
          convt3,
          tanh,
          
      )
      self.output = nn.Sequential(
          final,
          tanh
          
      )

    def forward(self,x):
        x = self.mod1(x)
        x = self.mod2(x)
        x = self.mod3(x)
        x = self.bottleneck(x)

        # Comment below 4 lines to output Compressed Representation
        x = self.modt1(x)
        x = self.modt2(x)
        x = self.modt3(x)
        x = self.output(x)
        return x

def get_ae(path):
    torch_model = Autoencoder().double()

    kmodel = keras.models.load_model(path,compile = False)
    weights = kmodel.get_weights()

    (torch_model.mod1[0].weight.data) = torch.from_numpy(np.transpose(weights[0]))
    (torch_model.mod1[0].bias.data) = torch.from_numpy(weights[1])

    torch_model.mod2[0].weight.data = torch.from_numpy(np.transpose(weights[2]))
    torch_model.mod2[0].bias.data = torch.from_numpy(weights[3])

    torch_model.mod3[0].weight.data = torch.from_numpy(np.transpose(weights[4]))
    torch_model.mod3[0].bias.data = torch.from_numpy(weights[5])

    # Comment below 4 lines to skip Squeeze-and-Excitation Network in AutoEncoder, adapt index in weights list

    torch_model.mod3[2].excitation[0].weight.data = torch.from_numpy(np.transpose(weights[6]))
    torch_model.mod3[2].excitation[0].bias.data = torch.from_numpy(weights[7])

    torch_model.mod3[2].excitation[2].weight.data = torch.from_numpy(np.transpose(weights[8]))
    torch_model.mod3[2].excitation[2].bias.data = torch.from_numpy(weights[9])

    torch_model.bottleneck[0].weight.data = torch.from_numpy((np.expand_dims(weights[10],0)))
    torch_model.bottleneck[0].bias.data = torch.from_numpy(weights[11])

    torch_model.modt1[0].weight.data = torch.from_numpy(np.transpose(weights[12]))
    torch_model.modt1[0].bias.data = torch.from_numpy((weights[13]))

    # Comment below 4 lines to skip Squeeze-and-Excitation Network in AutoEncoder, adapt index in weights list

    torch_model.modt1[1].excitation[0].weight.data = torch.from_numpy(np.transpose(weights[14]))
    torch_model.modt1[1].excitation[0].bias.data = torch.from_numpy((weights[15]))

    torch_model.modt1[1].excitation[2].weight.data = torch.from_numpy(np.transpose(weights[16]))
    torch_model.modt1[1].excitation[2].bias.data = torch.from_numpy((weights[17]))

    torch_model.modt2[0].weight.data = torch.from_numpy(np.transpose(weights[18]))
    torch_model.modt2[0].bias.data = torch.from_numpy((weights[19]))

    torch_model.modt3[0].weight.data = torch.from_numpy(np.transpose(weights[20]))
    torch_model.modt3[0].bias.data = torch.from_numpy((weights[21]))

    torch_model.output[0].weight.data = torch.from_numpy(np.transpose(weights[22]))
    torch_model.output[0].bias.data = torch.from_numpy((weights[23]))

    if torch.cuda.is_available():
        torch_model.cuda()
    summary(torch_model,input_size = (1,22050))
    return torch_model


class ACDNetV2(nn.Module):
    def __init__(self, input_length, n_class, sr, ch_conf=None):
        super(ACDNetV2, self).__init__();
        self.input_length = input_length;
        self.ch_config = ch_conf;

        stride1 = 2;
        stride2 = 2;
        channels = 8;
        k_size = (3, 3);
        n_frames = (sr/1000)*10; #No of frames per 10ms

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
        fcn = nn.Linear(fcn_no_of_inputs, n_class);
        nn.init.kaiming_normal_(fcn.weight, nonlinearity='sigmoid') # kaiming with sigoid is equivalent to lecun_normal in keras

        self.sfeb = nn.Sequential(
            #Start: Filter bank
            conv1, bn1, nn.ReLU(),\
            conv2, bn2, nn.ReLU(),\
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        );

        tfeb_modules = [];
        self.tfeb_width = int(((self.input_length / sr)*1000)/10); # 10ms frames of audio length in seconds
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
            tfeb_modules.append(nn.AvgPool2d(kernel_size = (h,w-1)));
        tfeb_modules.extend([nn.Flatten(), fcn]);

        self.tfeb = nn.Sequential(*tfeb_modules);

        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        );

    def forward(self, x):
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
        # print(length);
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

def GetACDNetModel(input_len=22050, nclass=50, sr=22050, channel_config=None):
    net = ACDNetV2(input_len, nclass, sr, ch_conf=channel_config);
    return net;




