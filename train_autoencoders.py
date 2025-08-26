
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pandas as pd



def train_autoencoders(model_encoder, model_decoder, num_epochs, learning_rate, dataloader_train, device, l2_w, noise_level):
    criterion = MSEloss_with_Mask()
    criterion = criterion.cuda()
    model_encoder.train()
    model_decoder.train()
    optimizer = torch.optim.Adam([
         {'params': model_encoder.parameters()},
         {'params': model_decoder.parameters()}], lr=learning_rate, weight_decay=l2_w)
    for epoch in range(num_epochs):
       for data in dataloader_train:
           R_movies = data.to(device)
           noisy_input = R_movies
           # ===================forward=====================
           latent = model_encoder(noisy_input.float())
           output = model_decoder(latent)

           loss= torch.sqrt(criterion(output, R_movies.float()))
           # ===================backward====================
           optimizer.zero_grad()
           loss.backward(retain_graph=True)
           optimizer.step()
           





def train_latentfactors_autoencoders(model_encoder_m, model_decoder_m, model_encoder_u, model_decoder_u, num_epochs,
                                     learning_rate, dataloader_movies, dataloader_users, device, l2_w,  batch_size_u):
     criterion = MSEloss_with_Mask()
     criterion = criterion.cuda()
     optimizer = torch.optim.Adam([
         {'params': model_encoder_m.parameters()},
         {'params': model_decoder_m.parameters()},
         {'params': model_encoder_u.parameters()},
         {'params': model_decoder_u.parameters()},], lr=learning_rate, weight_decay=l2_w)

     for epoch in range(num_epochs):


        #for data_m in (dataloader_movies):
        for cnt_i,data_m in enumerate(dataloader_movies):

            cnt_i=0
            cnt_j=batch_size_u

            for data_u in (dataloader_users):

               #a=data_m.numpy()
               Rs_movies = data_m.to(device)
               Rs_users  = data_u.to(device)

               # ===================forward=====================
               latent_movies = model_encoder_m(Rs_movies.float())
               output_movies = model_decoder_m(latent_movies)

               latent_users = model_encoder_u(Rs_users.float())
               output_users = model_decoder_u(latent_users)


               rating = Rs_movies[0:latent_movies.size()[0], cnt_i:cnt_j]
               prod_factors = torch.matmul(latent_movies, latent_users.T)

               cnt_j = cnt_j + latent_users.size()[0]
               cnt_i = cnt_i + latent_users.size()[0]


               loss_movies = torch.sqrt(criterion(output_movies, Rs_movies.float()))
               loss_users = torch.sqrt(criterion(output_users, Rs_users.float()))
               loss_factors = torch.sqrt(criterion(prod_factors,rating.float()))

               loss = 0.5*loss_movies + 0.5*loss_users + loss_factors
              # ===================backward====================
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
   
       
       
       
       
       
       
       
       
       
       
          

class MSEloss_with_Mask(nn.Module):
  def __init__(self):
    super(MSEloss_with_Mask,self).__init__()

  def forward(self, inputs, targets):
    # Masking into a vector of 1's and 0's.
    mask = (targets!=0)
    mask = mask.float()

    # Actual number of ratings.
    # Take max to avoid division by zero while calculating loss.
    other = torch.Tensor([1.0])
    other = other.cuda()
    number_ratings = torch.max(torch.sum(mask),other)
    error = torch.sum(torch.mul(mask,torch.mul((targets-inputs),(targets-inputs))))
    loss = error.div(number_ratings)
    return loss[0]
