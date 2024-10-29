import torch
from kwisatzHaderach import KwisatzHaderach
from Kaisarion import Kaisarion
from Transformer import ContConvTransformer
from Simple_model import ContinuousConvolutionModel
import json
import os
import tqdm
import numpy as np
from datagen import *
from trainer import Trainer

def euclidean_distance(a, b):
    return torch.sqrt(torch.sum((a - b)**2, dim=-1) + 1e-12)

def mean_distance(a, b):
    return torch.mean(euclidean_distance(a, b))

def loss_fn(a, b, num_neighbors=None):
    if num_neighbors is not None:
        importances = torch.exp(num_neighbors/num_neighbors.max()) 
        dists = euclidean_distance(a, b)
        loss = torch.mean(importances * dists)
        return loss
    return torch.mean(euclidean_distance(a, b))

model = ContinuousConvolutionModel(kernel_size=[7,7,7], num_features = 1, calc_neighbors=True, radius=2.0, use_dense_in_conv=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7, verbose=True)
scheduler = None

trainer = Trainer(loss_fn=loss_fn, batch_size=16, device='cuda', mode='present')

trainer.train(model=model, optimizer=optimizer, scheduler=scheduler, rounds=500, epochs_per_dataset=1, save_after=10,scenes_per_dataset=25)