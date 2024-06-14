from numpy import dtype
import torch
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from util import CustomDataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# Creating a PyTorch class
# 2048 ==> 1024 ==> 512 ==> 128 ==> 64 ==> 32 ==> 10

class AeDrug(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 2048 ==> 10
        self.encoder = torch.nn.Sequential(

            torch.nn.Linear(2048, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
            #torch.nn.ReLU(),
            #torch.nn.Linear(128, 64)

        )

        # 10 ==> 2048
        self.decoder = torch.nn.Sequential(
            #torch.nn.Linear(64, 128),
            #torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048)
        )

    def forward(self, x):
        #print(dtype(x))
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.type(torch.float32)
        print(x.dtype)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

