import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision import models
import numpy as np
import math

class CBN2d(nn.Module):
    def __init__(self, in_channel, n_condition=128):
        super(CBN2d, self).__init__()
        self.in_channel = in_channel
        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.embed = nn.Linear(n_condition, in_channel*2) # generate the affine parameters

        self._initialize()

    def _initialize(self):
        self.embed.weight.data[:, :self.in_channel] = 1 # init gamma as 1
        self.embed.weight.data[:, self.in_channel:] = 0 # init beta as 0

    def forward(self, h, y):
        gamma, beta = self.embed(y).chunk(2, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        out = gamma * self.bn(h) + beta
        return out


class GBlock(nn.Module):
    """Convolution blocks for the generator"""
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
        super(GBlock, self).__init__()
        hidden_channel = out_channel 
        
        # depthwise seperable
        self.dw_conv1 = nn.Conv2d(in_channel, in_channel,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=int(in_channel))

        self.dw_conv2 = nn.Conv2d(hidden_channel, hidden_channel, 
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=int(hidden_channel))
        
        self.pw_conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1)
        self.pw_conv2 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)

        self.c_sc = nn.Conv2d(in_channel, out_channel, kernel_size=1)

        self.cbn0 = CBN2d(in_channel)
        self.cbn1 = CBN2d(hidden_channel)
        
        self._initialize()
        
    def _initialize(self):
        nn.init.xavier_uniform_(self.dw_conv1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.dw_conv2.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.pw_conv1.weight, gain=1)
        nn.init.xavier_uniform_(self.pw_conv2.weight, gain=1)
        nn.init.xavier_uniform_(self.c_sc.weight, gain=1)


    def _upsample(self, x):
        h, w = x.size()[2:]
        return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')

    def shortcut(self, x):
        h = self._upsample(x)
        h = self.c_sc(h)
        return h

    def forward(self, x, y):
        out = self.cbn0(x, y)
        out = F.relu(out)
        out = self._upsample(out)
        out = self.pw_conv1(self.dw_conv1(out))
        out = self.cbn1(out, y)
        out = F.relu(out)
        out = self.pw_conv2(self.dw_conv2(out))
        return out + self.shortcut(x)  # residual


class Generator(nn.Module):
    def __init__(self, image_size=256, conv_dim=64, z_dim=128, c_dim=128, repeat_num=5):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.repeat_num = repeat_num
        self.nfilter0 = np.power(2, repeat_num-1)*self.conv_dim
        self.W0 = image_size // np.power(2, repeat_num)
        
        weight = torch.FloatTensor(np.load('cls_weight_reduce.npy'))
        self.embeding = nn.Embedding.from_pretrained(weight, freeze=False)

        
        self.fc = nn.Linear(z_dim, self.nfilter0*self.W0*self.W0)
        # after reshape: (N, self.nfilter0, self.W0, self.W0) = (N, 1024, 4, 4)
        nfilter = self.nfilter0
        blocks = []
        blocks.append(GBlock(nfilter, nfilter, kernel_size=3))
        for i in range(1, repeat_num):
            blocks.append(GBlock(nfilter, nfilter//2))
            nfilter = nfilter // 2
        self.blocks = nn.Sequential(*blocks)
        
        self.bn = nn.BatchNorm2d(nfilter)
        self.colorize = nn.Conv2d(conv_dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, noise, label):
        h = self.fc(noise).view(-1, self.nfilter0, self.W0, self.W0)
        y_emb = self.embeding(label)

        for i in range(self.repeat_num):
            h = self.blocks[i](h, y_emb)
        h = F.relu(self.bn(h))
         
        out = F.tanh(self.colorize(h)) # (batch_size, 3, image_size, image_size)
        
        return out

    def interpolate(self, noise, y_emb):
        h = self.fc(noise).view(-1, self.nfilter0, self.W0, self.W0)
        
        for i in range(self.repeat_num):
            h = self.blocks[i](h, y_emb)
        h = F.relu(self.bn(h))
         
        out = F.tanh(self.colorize(h)) # (batch_size, 3, image_size, image_size)
        
        return out