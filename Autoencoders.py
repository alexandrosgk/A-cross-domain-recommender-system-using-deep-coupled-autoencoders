import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class encoder(nn.Module):
    def __init__(self, input_n, layer1_n, layer2_n, latent_n):
        super(encoder, self).__init__()

        self.input_n = input_n
        self.layer1_n = layer1_n
        self.layer2_n = layer2_n
        self.latent_n = latent_n
        self.dropout = nn.Dropout(0.2)

        #Encoder
        self.enc1 =  (nn.Linear(in_features=input_n, out_features=layer1_n))
        self.enc2 =  (nn.Linear(in_features=layer1_n, out_features=layer2_n))
        self.enc3 =  (nn.Linear(in_features=layer2_n, out_features=latent_n))

    def forward(self, x):
        x = F.leaky_relu(self.enc1(x))
        x = (self.dropout(x))

        x = F.leaky_relu(self.enc2(x))
        x = (self.dropout(x))
        x = F.leaky_relu(self.enc3(x))

        return x

class decoder(nn.Module):
    def __init__(self,input_n,layer1_n,layer2_n,output_n):
        super(decoder, self).__init__()

        self.input_n = input_n
        self.layer1_n = layer1_n
        self.layer2_n = layer2_n
        self.output_n = output_n

        #Decoder
        self.dec1 = (nn.Linear(in_features=input_n, out_features=layer1_n))
        self.dec2 = (nn.Linear(in_features=layer1_n, out_features=layer2_n))
        self.dec3 = (nn.Linear(in_features=layer2_n, out_features=output_n))

    def forward(self, x):
        x = F.leaky_relu(self.dec1(x))
        x = F.leaky_relu(self.dec2(x))
        x =  (self.dec3(x))
        return x