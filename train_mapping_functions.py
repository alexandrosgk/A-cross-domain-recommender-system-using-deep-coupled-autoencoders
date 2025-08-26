import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pandas as pd


def train_MLP(mlp_model, model_encoder_s, model_encoder_t, num_epochs, learning_rate, dataloader_Coupled, device, l2_w):

    criterion = nn.MSELoss()
    mlp_model.train()
    optimizer = torch.optim.Adam(
         mlp_model.parameters(), lr=learning_rate, weight_decay=l2_w)

    for epoch in range(num_epochs):
       for data_s, data_t in dataloader_Coupled:
           Rs_movies = data_s.to(device)
           Rt_movies = data_t.to(device)

           noisy_input_s = Rs_movies
           noisy_input_t = Rt_movies
           # ===================forward=====================
           latent_s = model_encoder_s((noisy_input_s.float()))
           latent_t = model_encoder_t((noisy_input_t.float()))
           pred = mlp_model(latent_s)
           loss = torch.sqrt(criterion(latent_t, pred))
           # ===================backward====================
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

#GAN


def train_GAN(generator, discriminator, model_encoder_s, model_encoder_t, dataloader, device, epochs, learning_rate):
    combined_optimizer = torch.optim.Adam(
        list(generator.parameters()) + list(discriminator.parameters()), lr=learning_rate
    )
    mse_loss = nn.MSELoss()
    ce_loss = nn.BCEWithLogitsLoss()

    real_label = 1
    fake_label = 0

    for epoch in range(epochs):
        for data_s, data_t in dataloader:
            # Assuming data_s and data_t are tensors
            data_s = data_s.float().to(device)  # Ensure data is float type and moved to device
            data_t = data_t.float().to(device)

            # Generate embeddings
            real_embeddings = model_encoder_t(data_t.float())
            source_embeddings = model_encoder_s(data_s.float())
            fake_embeddings = generator(source_embeddings.detach())

            # Discriminator training for real and fake data
            discriminator.zero_grad()
            real_preds = discriminator(real_embeddings)
            real_labels = torch.ones(real_embeddings.size(0), 1, device=device)
            real_loss = ce_loss(real_preds, real_labels)

            fake_preds = discriminator(fake_embeddings.detach())
            fake_labels = torch.zeros(fake_embeddings.size(0), 1, device=device)
            fake_loss = ce_loss(fake_preds, fake_labels)

            # Generator loss
            generator.zero_grad()
            tricked_preds = discriminator(fake_embeddings)
            g_loss = torch.sqrt(mse_loss(fake_embeddings, real_embeddings))

            # Combine all losses
            combined_loss = real_loss + fake_loss + g_loss
            combined_loss.backward()
            combined_optimizer.step()


