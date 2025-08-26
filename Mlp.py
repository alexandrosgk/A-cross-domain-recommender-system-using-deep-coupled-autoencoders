
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class mlp_network(nn.Module):
    def __init__(self, input_n, layer1_n):
        super(mlp_network, self).__init__()
        self.input_n = input_n
        self.layer1_n = layer1_n
        self.layer2_n = layer1_n
        self.dropout = nn.Dropout(0.1)


        self.layer1 =  (nn.Linear(in_features=input_n,  out_features=layer1_n))
        #self.layer2 =  (nn.Linear(in_features=layer1_n, out_features=layer1_n))
        self.layer2 =  (nn.Linear(in_features=layer1_n, out_features=input_n))

    def forward(self, x):

        x = F.leaky_relu(self.layer1(x))
        x = (self.dropout(x))

        x = F.leaky_relu(self.layer2(x))

        return x