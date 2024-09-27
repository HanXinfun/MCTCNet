import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimilarLossCalculator(nn.Module):

    def __init__(self, beta, tau, device):
        super(SimilarLossCalculator, self).__init__()
        self.beta = beta
        self.tau = tau
        self.device = device

        self.U = nn.Parameter(torch.randn(128, 64))
        self.V = nn.Parameter(torch.randn(128, 64))
        # Linear
        self.line_layers = torch.nn.Linear(1152, 128)

    def retrieve_embeddings(self, x):
        U_embeddings = []
        V_embeddings = []

        for i in range(x.size(0)):
            U_x_i = self.U @ F.softmax(self.beta * (self.U.T @ x[i]), dim=-1)
            V_x_i = self.V @ F.softmax(self.beta * (self.V.T @ x[i]), dim=-1)
            U_embeddings.append(U_x_i)
            V_embeddings.append(V_x_i)

        U_embeddings = torch.stack(U_embeddings)
        V_embeddings = torch.stack(V_embeddings)

        return U_embeddings, V_embeddings


    def compute_loss(self, U_embeddings, V_embeddings, N):
        total_loss = 0
        num_embeddings = len(U_embeddings)
        
        for U_x in U_embeddings:
            for U_z in U_embeddings:
                if not torch.equal(U_x, U_z):

                    similarity_matrix = torch.mm(U_x, U_z.T) / self.tau
                    
                    mask = torch.eye(N, dtype=torch.bool).to(similarity_matrix.device)
                    positives = similarity_matrix[mask].view(N, -1)
                    negatives = similarity_matrix[~mask].view(N, N - 1)

                    loss = -torch.log(positives.exp() / negatives.exp().sum(dim=-1)).mean()

                    total_loss += loss

        for V_x in V_embeddings:
            for V_z in V_embeddings:
                if not torch.equal(V_x, V_z):

                    similarity_matrix = torch.mm(V_x, V_z.T) / self.tau

                    mask = torch.eye(N, dtype=torch.bool).to(similarity_matrix.device)
                    positives = similarity_matrix[mask].view(N, -1)
                    negatives = similarity_matrix[~mask].view(N, N - 1)

                    loss = -torch.log(positives.exp() / negatives.exp().sum(dim=-1)).mean()
                    total_loss += loss
        
        return total_loss / (2 * num_embeddings * (num_embeddings - 1))


    def calculate_similar_loss(self, x_all):
        N = x_all[0].size(0)
        
        U_embeddings_list = []
        V_embeddings_list = []

        for x in x_all:
            x = x.reshape(x.shape[0], -1)  # (32, 1152)
            x = self.line_layers(x)  # (32, 128)

            U_emb, V_emb = self.retrieve_embeddings(x)
            U_embeddings_list.append(U_emb)
            V_embeddings_list.append(V_emb)

        loss = self.compute_loss(U_embeddings_list, V_embeddings_list, N)
        
        return loss
