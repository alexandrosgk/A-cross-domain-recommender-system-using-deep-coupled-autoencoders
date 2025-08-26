
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pandas as pd

from Testing_function import *

def train_Coupled_model(mapping_model, model_encoder_s, model_decoder_t, num_epochs, learning_rate,
                        dataloader_Coupled, device, Movies_s_test, Movies_t_test, l2_w, noise_level):

    criterion = MSEloss_with_Mask()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam([
        {'params': model_encoder_s.parameters()},
        {'params': mapping_model.parameters()},
        {'params': model_decoder_t.parameters()},],
                                          lr=learning_rate, weight_decay=l2_w)
    rmsee_loss=[]
    mae_loss=[]
    prec_loss = []
    recall_loss = []
    for epoch in range(num_epochs):
        for data_s, data_t in dataloader_Coupled:
            Rs_movies = data_s.to(device)
            Rt_movies = data_t.to(device)
            noisy_input = Rs_movies+((noise_level**0.5)*torch.randn(Rs_movies.shape[0],Rs_movies.shape[1])).to(device)
            # ===================forward=====================
            latent_source = model_encoder_s(noisy_input.float())
            output_mlp    = mapping_model(latent_source)
            output_target = model_decoder_t(output_mlp)


            loss_target = torch.sqrt(criterion(output_target, Rt_movies.float()))
            loss = loss_target
           # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    r3,m3, precision, recall= test_data(model_encoder_s, mapping_model, model_decoder_t, Movies_s_test, Movies_t_test, noise_level)
    # print(r3,m3)
    rmsee_loss.append(r3)
    mae_loss.append(m3)
    prec_loss.append(precision)
    recall_loss.append(recall)
        
        
    min_rmse = min(rmsee_loss)
    min_epoch_rmse = rmsee_loss.index(min_rmse)

    # Find minimum MAE
    min_mae = min(mae_loss)
    min_epoch_mae = mae_loss.index(min_mae)

    # Find maximum Precision
    max_prec = max(prec_loss)
    max_epoch_prec = prec_loss.index(max_prec)

    # Find maximum Recall
    max_recall = max(recall_loss)
    max_epoch_recall = recall_loss.index(max_recall)

    # Print the results
    print(f"Lowest RMSE: {min_rmse:.4f} at epoch {min_epoch_rmse}")
    print(f"Lowest MAE: {min_mae:.4f} at epoch {min_epoch_mae}")
    print(f"Highest Precision: {max_prec:.4f} at epoch {max_epoch_prec}")
    print(f"Highest Recall: {max_recall:.4f} at epoch {max_epoch_recall}")    


    return  min(rmsee_loss), min(mae_loss), min(prec_loss), min(recall_loss)



def train_Coupled_model_Matrix_factorization(mapping_model, model_encoder_m_s, model_encoder_u_t, num_epochs, learning_rate, dataloader_movies,
                        dataloader_users, device, Movies_s_test, Movies_t_test, l2_w, Users_t_train, Movies_t_train, batch_size_m, batch_size_u):

    criterion = MSEloss_with_Mask()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam([
        {'params': model_encoder_m_s.parameters()},
        {'params': mapping_model.parameters()},
        {'params': model_encoder_u_t.parameters()},],
                                          lr=learning_rate, weight_decay=l2_w)
    rmsee_loss=[]
    mae_loss=[]
    prec_loss=[]
    recall_loss=[]
    for epoch in range(num_epochs):
        for data_m_s, data_m_t in (dataloader_movies):
            Rs_movies = data_m_s.to(device)
            Rt_movies = data_m_t.to(device)
            cnt_i = 0
            cnt_j = batch_size_u

            for data_u_t in (dataloader_users):
               Rt_users  = data_u_t.to(device)

               # ===================forward=====================
               latent_source = model_encoder_m_s(Rs_movies.float())
               output_mlp    = mapping_model(latent_source)
               latent_users  = model_encoder_u_t(Rt_users.float())

               output_target = torch.matmul(output_mlp, latent_users.T)
               rating = Rt_movies[0:latent_source.size()[0],cnt_i:cnt_j]

               cnt_j = cnt_j + latent_users.size()[0]
               cnt_i = cnt_i + latent_users.size()[0]

               loss = torch.sqrt(criterion(output_target, rating.float()))
               # ===================backward====================
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

    r3, m3, prec, recall = test_data_Matrix_factorization(model_encoder_m_s, mapping_model, model_encoder_u_t,
                    Movies_s_test, Movies_t_test, Users_t_train)
    rmsee_loss.append(r3)
    mae_loss.append(m3)
    prec_loss.append(prec)
    recall_loss.append(recall)

    return min(rmsee_loss), min(mae_loss), min(prec_loss), min(recall_loss)


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