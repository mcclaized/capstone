import csv
import json
import numpy as np
import pandas as pd
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class BPRModel(nn.Module):
    """
    This class implements a PyTorch model for Bayesian Personalized Recommendation
    Per https://arxiv.org/abs/1205.2618
    """
    def __init__(self, num_bonds, num_factors, regularization_lambda=0.0000001):
        super().__init__()
        
        self.regularization_lambda = regularization_lambda
        self.bond_embedding = nn.Embedding(num_bonds, num_factors)
    
    def forward(self, rankings):
        # Input is a batch of (bond, better recommendation, worse recommendation) triplets
        bonds = rankings[:,0]
        better_recommendations = rankings[:,1]
        worse_recommendations = rankings[:,2]
        
        # Fetch our existing latent vector representation of each bond
        bond_embeddings = self.bond_embedding(bonds)
        better_recommendation_embeddings = self.bond_embedding(better_recommendations)
        worse_recommendation_embeddings = self.bond_embedding(worse_recommendations)
        
        # Compute the dot product of the latent embedding
        # We equate large dot products with high bond similarity
        dot_better = torch.sum(bond_embeddings * better_recommendation_embeddings, dim=1)
        dot_worse = torch.sum(bond_embeddings * worse_recommendation_embeddings, dim=1)
        dot_diff = dot_better - dot_worse
        
        log_likelihood = torch.mean(F.logsigmoid(dot_diff))
        
        # useful to track how many the model "got right", i.e. agrees that better ones are better
        auc = torch.mean((dot_diff > 0).float())
        
        # Recall that a guassian prior is equivalent to l2 regularization
        # http://bjlkeng.github.io/posts/probabilistic-interpretation-of-regularization/
        prior = sum(
            [
                self.regularization_lambda * torch.sum(bond_embeddings * bond_embeddings),
                self.regularization_lambda * torch.sum(better_recommendation_embeddings * better_recommendation_embeddings),
                self.regularization_lambda * torch.sum(worse_recommendation_embeddings * worse_recommendation_embeddings),
            ]
        )
        
        return log_likelihood, prior, auc        


class ModelHelper(object):
    def __init__(self, model, isin_to_index_mapping, metadata):
        
        self.isin_to_index = isin_to_index_mapping
        self.index_to_isin = {idx: isin for isin, idx in self.isin_to_index.items()}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.metadata = metadata
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        
    def predict(self, isin, n=10):
        with torch.no_grad():
            bond_idx = self.isin_to_index[isin]
            bond = self.model.bond_embedding(torch.tensor(bond_idx, device=self.device))
            dots = torch.sum(bond * self.model.bond_embedding.weight.data, dim=1)
            bond_indices = torch.argsort(dots, descending=True)[:n]
            isins = [self.index_to_isin[i] for i in bond_indices.cpu().numpy().tolist()]
            return isins
    
    def process_feedback(self, feedback):
        # Save a list of [(bond, better recommendation, worse recommendation), ...]
        feedback = [[self.isin_to_index[isin] for isin in isin_triplet] for isin_triplet in feedback]
        feedback = torch.tensor(feedback, device=self.device)
        likelihood, prior, auc = self.model(feedback)
        loss = -likelihood + prior
        loss.backward()
        self.optimizer.step()
    
    def display(self, isins, display_cols=None):
        if display_cols is None:
            display_cols = ['BCLASS3', 'Ticker', 'Country', 'Bid Spread', 'Cur Yld', 'OAS', 'OAD', 'Cpn']
        display_bonds = (
            self.metadata.get_bonds(isins)
            .reset_index()
            [display_cols]
        )
        return display_bonds

    def display_table(self, isins, display_cols=None):
        return tabulate(
            self.display(isins, display_cols=display_cols),
            headers=display_cols, 
            #showindex="never"
        )

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        
