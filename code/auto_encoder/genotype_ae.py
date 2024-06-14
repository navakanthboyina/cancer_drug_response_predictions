import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Creating a PyTorch class
# 2048 ==> 1024 ==> 512 ==> 128 ==> 64 ==> 32 ==> 10
class AeGene(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 2048 ==> 10
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(3008, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(p=0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(p=0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
            #torch.nn.ReLU(),
            #torch.nn.Linear(64, 32)

        )

        # 10 ==> 2048
        self.decoder = torch.nn.Sequential(
            #torch.nn.Linear(32, 64),
            #torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(1024, 3008)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded
