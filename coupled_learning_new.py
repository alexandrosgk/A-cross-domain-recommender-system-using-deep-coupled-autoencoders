import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pandas as pd

from Testing_function import *


def train_latentfactors_autoencoders_new(model_encoder_m, model_decoder_m, model_encoder_u, model_decoder_u,
                                     num_epochs, learning_rate, dataloader_movies, dataloader_users,
                                     ratings_matrix, device, l2_w, weight_loss_movies = 0.5, weight_loss_users = 0.5, weight_loss_latent = 0.5):
    criterion = MSEloss_with_Mask().to(device)
    optimizer = torch.optim.Adam([
        {'params': model_encoder_m.parameters()},
        {'params': model_decoder_m.parameters()},
        {'params': model_encoder_u.parameters()},
        {'params': model_decoder_u.parameters()}
    ], lr=learning_rate, weight_decay=l2_w)

    import itertools
    user_loader_cycle = itertools.cycle(dataloader_users)
 
    for epoch in range(num_epochs):
        for (idx_m, data_m) in dataloader_movies:
            (idx_u, data_u) = next(user_loader_cycle)
            data_m, data_u = data_m.to(device).float(), data_u.to(device).float()

            # Forward pass through item autoencoder
            latent_movies = model_encoder_m(data_m)
            output_movies = model_decoder_m(latent_movies)

            # Forward pass through user autoencoder
            latent_users = model_encoder_u(data_u)
            output_users = model_decoder_u(latent_users)
            
            # Find common users
            rating = ratings_matrix[idx_m, :][:, idx_u].to(device).float()
            prod_factors = torch.matmul(latent_movies, latent_users.T)

            # Compute losses
            loss_movies = torch.sqrt(criterion(output_movies, data_m))
            loss_users = torch.sqrt(criterion(output_users, data_u))
            loss_factors = torch.sqrt(criterion(prod_factors, rating))

            # Combined loss
            loss = weight_loss_movies * loss_movies + weight_loss_users * loss_users + weight_loss_latent*loss_factors

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model_encoder_m, model_decoder_m, model_encoder_u, model_decoder_u






def train_Coupled_model_Matrix_factorization_new_version(mapping_model, model_encoder_m_s, model_encoder_u_t, num_epochs, learning_rate, dataloader_movies,
                        dataloader_users, device, Movies_s_test, Movies_t_test, l2_w, Users_t_train, Movies_t_train):

    criterion = MSEloss_with_Mask()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam([
        {'params': model_encoder_m_s.parameters()},
        {'params': mapping_model.parameters()},
        {'params': model_encoder_u_t.parameters()},],
                                          lr=learning_rate, weight_decay=l2_w)
    rmsee_loss=[]
    mae_loss=[]
    prec_loss = []
    recall_loss = []
    Movies_t_train = torch.tensor(Movies_t_train)
    
    import itertools
    user_loader_cycle = itertools.cycle(dataloader_users)
 
    for epoch in range(num_epochs):
        for (idx_m, data_m) in dataloader_movies:
            (idx_u, data_u) = next(user_loader_cycle)
            data_m, data_u = data_m.to(device).float(), data_u.to(device).float()
    
            # ===================forward=====================
            latent_source = model_encoder_m_s(data_m)
            output_mlp    = mapping_model(latent_source)
            latent_users  = model_encoder_u_t(data_u)

            output_target = torch.matmul(output_mlp, latent_users.T)
            
            rating = Movies_t_train[idx_m, :][:, idx_u].to(device).float()

            loss = torch.sqrt(criterion(output_target, rating.float()))
            # ===================backward===============================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

        r3, m3, prec, recall = test_data_Matrix_factorization(model_encoder_m_s, mapping_model, model_encoder_u_t,
                        Movies_s_test, Movies_t_test, Users_t_train)
        rmsee_loss.append(r3)
        mae_loss.append(m3)
        prec_loss.append(prec)
        recall_loss.append(recall)

    return min(rmsee_loss),min(mae_loss), min(prec_loss),min(recall_loss)


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
