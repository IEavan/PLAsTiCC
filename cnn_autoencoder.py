import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable 

# Define Model
class CNNAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(CNNAutoEncoder, self).__init__()

        # Conv Layers (init dim (50), passband as channels)
        self.conv1 = torch.nn.Conv1d(6, 8, 5) # 46
        self.conv2 = torch.nn.Conv1d(8, 16, 3, stride=2)
        self.conv3 = torch.nn.Conv1d(16, 16, 3, stride=2)
        self.conv4 = torch.nn.Conv1d(16, 16, 3, stride=2)

        # Batchnorms
        self.batchnorm1 = torch.nn.BatchNorm1d(50)
        self.batchnorm2 = torch.nn.BatchNorm1d(23)

        self.relu = torch.nn.LeakyReLU()
        self.fully_conected = torch.nn.Linear(2 * 16, 1)

    def forward(self, fluxes, dropout_p=0.25):
        dropout = torch.nn.Dropout(dropout_p)

        fluxes = self.batchnorm1(fluxes)
        fluxes = self.conv1(fluxes)
        fluxes = self.relu(fluxes)
        fluxes = dropout(fluxes)

        fluxes = self.conv2(fluxes)
        fluxes = self.relu(fluxes)
        fluxes = dropout(fluxes)

        
