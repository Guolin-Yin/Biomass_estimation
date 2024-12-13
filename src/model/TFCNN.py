from einops import repeat
from .backbone.TSViT.TSViTdense import Transformer
from .backbone.RegressionHead import RegressionHead
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from einops.layers.torch import Rearrange
from torch import optim
from matplotlib.patches import Rectangle
from torch.optim import lr_scheduler
from pathlib import Path

import os
class TFCNN(nn.Module):
    def __init__(self, 
                model_params,
                train_params = None
    ):
        super().__init__()


        self.model_params = model_params
        self.train_params = train_params

        self._init_model_params()
        
    def _init_model_params(self):
        dim, depth, heads, dim_head, mlp_dim = self.model_params['dim'], self.model_params['depth'], self.model_params['heads'], self.model_params['dim_head'], self.model_params['mlp_dim']

        self.tem_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.to_temporal_embedding_input = nn.Linear(366, dim)

        # CNN architecture adjusted for input size (16, 12x12)
        # self.features = nn.Sequential(
        #     nn.Conv2d(6, 32, kernel_size=(2,1), ),  # First layer adjusts for 16 input channels
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2,1), stride=2),  # Output size: (32, 6, 6)
        #     nn.Conv2d(32, 64, kernel_size=(2,1)),  # Second convolution layer
        #     nn.ReLU(),
        #     # nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (64, 3, 3)
        # )
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2,1)),  # Input: (B, 6, 1, 1) -> Output: (B, 32, 1, 1)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # Output: (B, 32, 1, 1)
            # nn.Conv2d(32, 64, kernel_size=(1,1)),  # Output: (B, 64, 1, 1)
            # nn.ReLU(),
        )
        # self.regression_head = nn.Linear(64, 1)
        self.regression_head = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
    def _get_temporal_position_embeddings(self, t):
        t = (t * 365.0001).to(torch.int64)
        t = F.one_hot(t, num_classes=366).to(torch.float32)
        t = t.reshape(-1, 366)
        return self.to_temporal_embedding_input(t)
    def forward(self, x):
        # temporal_pos_embedding = self._get_temporal_position_embeddings(time).unsqueeze(dim=0) # time -> (1, time), temporal_pos_embedding -> (1, time, channel)

        # x += temporal_pos_embedding
        x = x.unsqueeze(1) # x -> (1, B <No. Pixels>, channel)
        x = self.tem_transformer(x) # x -> (1, B <No. Pixels>, channel)
        x = x.permute(0, 2, 1) # x -> (B <No. Pixels>, channel, 1)
        # x = x.permute(0, 2, 1) # x -> (1, channel, B <No. Pixels>)
        # x = self.regression_head(x)
        # x = x.squeeze(0)
        x = self.features(x.unsqueeze(1))
        x = self.regression_head(x.view(x.size(0), -1))

        return x
   