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

model = ContinuousConvolutionModel(kernel_size=[7,7,7], num_features = 4)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7, verbose=True)
scheduler = None

trainer = Trainer(loss_fn=mean_distance, batch_size=32, device='cpu', mode='present')

trainer.train(model=model, optimizer=optimizer, scheduler=scheduler, rounds=500, epochs_per_dataset=1, save_after=10,scenes_per_dataset=2)