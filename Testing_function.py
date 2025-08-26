import os
import numpy as np
import torch
import copy
import torch.nn as nn

device = torch.device("cuda")

def test_data(model_encoder_s, mlp_model, model_decoder_t, Movies_s_test, Movies_t_test, noise_level):
    threshold = 0.5
    
    criterion = nn.MSELoss()
    c2 = nn.L1Loss()

    model_encoder_s.eval()
    model_decoder_t.eval()
    mlp_model.eval()

    data_t = torch.from_numpy(Movies_t_test)
    data_s = torch.from_numpy(Movies_s_test)

    Rs_movies = data_s.to(device)
    Rt_movies = data_t.to(device)
    noisy_input = Rs_movies

    # ===================forward=====================
    latent_source = model_encoder_s(noisy_input.float())
    output_mlp    = mlp_model(latent_source)
    output_target_ZEROS = model_decoder_t(output_mlp)

    non_zero_indices = Rt_movies != 0
    output_target = output_target_ZEROS[non_zero_indices]
    Rt_movies_NON_ZEROS = Rt_movies[non_zero_indices]
    

    loss_target = torch.sqrt(criterion(output_target, Rt_movies_NON_ZEROS.float()))
    mae = c2(output_target, Rt_movies_NON_ZEROS.float())



    # Convert to numpy arrays
    # Convert to numpy arrays
    output_target_ZEROS[Rt_movies == 0] = 0
    output_target = output_target_ZEROS.cpu().detach().numpy()
    Rt_movies = Rt_movies.cpu().detach().numpy()

    top_k = 5
    precision_list = []
    recall_list = []

    # Calculate precision and recall for each user based on top-K recommendations
    for user_index in range(Movies_t_test.shape[0]):
        true_ratings = Rt_movies[user_index]
        predicted_ratings = output_target[user_index]

        # Sort indices based on predicted ratings in descending order and get top-K
        top_k_indices = np.argsort(predicted_ratings)[-top_k:][::-1]
        
        top_k_true_ratings = true_ratings[top_k_indices]
        top_k_predicted_ratings = predicted_ratings[top_k_indices]

        # Filter out unrated items in the top-K
        valid_indices = top_k_true_ratings != 0
        top_k_true_ratings = top_k_true_ratings[valid_indices]
        top_k_predicted_ratings = top_k_predicted_ratings[valid_indices]

        if len(top_k_true_ratings) == 0:
            continue

        n_rel = np.sum(top_k_true_ratings >= threshold)
        n_rec_k = np.sum(top_k_predicted_ratings >= threshold)
        n_rel_and_rec_k = np.sum((top_k_true_ratings >= threshold) & (top_k_predicted_ratings >= threshold))

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    precision = np.mean(precision_list) if precision_list else 0
    recall = np.mean(recall_list) if recall_list else 0

    return loss_target.item(), mae.item(), precision, recall

    # return loss_target.item(), mae.item()



def test_data_Matrix_factorization(model_encoder_m_s,mlp_model,model_encoder_u_t,Movies_s_test,Movies_t_test,Users_t_test):
    criterion = nn.MSELoss()
    c2=nn.L1Loss()
    threshold = 0.5
    users_t = torch.from_numpy(Users_t_test)
    movies_t= torch.from_numpy(Movies_t_test)
    data_s=torch.from_numpy(Movies_s_test)
    Rs_movies= data_s.to(device)
    Rt_users= users_t.to(device)
    Rt_movies= movies_t.to(device)

    model_encoder_m_s.eval()
    model_encoder_u_t.eval()
    mlp_model.eval()

    # ===================forward=====================
    latent_movies = model_encoder_m_s(Rs_movies.float())
    output_mlp    = mlp_model(latent_movies)

    latent_users  = model_encoder_u_t(Rt_users.float())
    output_target_ZEROS = torch.matmul(output_mlp,latent_users.T)

    non_zero_indices = Rt_movies != 0
    output_target = output_target_ZEROS[non_zero_indices]
    Rt_movies_NON_ZEROS = Rt_movies[non_zero_indices]
    

    loss_target = torch.sqrt(criterion(output_target, Rt_movies_NON_ZEROS.float()))
    mae = c2(output_target, Rt_movies_NON_ZEROS.float())



    # Convert to numpy arrays
    
    output_target_ZEROS[Rt_movies == 0] = 0
    output_target = output_target_ZEROS.cpu().detach().numpy()
    Rt_movies = Rt_movies.cpu().detach().numpy()

    top_k = 5
    precision_list = []
    recall_list = []

    # Calculate precision and recall for each user based on top-K recommendations
    for user_index in range(Movies_t_test.shape[0]):
        true_ratings = Rt_movies[user_index]
        predicted_ratings = output_target[user_index]

        # Sort indices based on predicted ratings in descending order and get top-K
        top_k_indices = np.argsort(predicted_ratings)[-top_k:][::-1]
        
        top_k_true_ratings = true_ratings[top_k_indices]
        top_k_predicted_ratings = predicted_ratings[top_k_indices]

        # Filter out unrated items in the top-K
        valid_indices = top_k_true_ratings != 0
        top_k_true_ratings = top_k_true_ratings[valid_indices]
        top_k_predicted_ratings = top_k_predicted_ratings[valid_indices]

        if len(top_k_true_ratings) == 0:
            continue

        n_rel = np.sum(top_k_true_ratings >= threshold)
        n_rec_k = np.sum(top_k_predicted_ratings >= threshold)
        n_rel_and_rec_k = np.sum((top_k_true_ratings >= threshold) & (top_k_predicted_ratings >= threshold))

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    precision = np.mean(precision_list) if precision_list else 0
    recall = np.mean(recall_list) if recall_list else 0

    return loss_target.item(), mae.item(), precision, recall
