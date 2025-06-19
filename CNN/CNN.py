import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self, canale_intrare=2, canale_iesire=2):
        super().__init__()
        # tensorul de intrare are (dim_batch, nr_canale, frecvente, timp), nr_canale=1 sau 2
        self.strat_conv1 = nn.Conv2d(in_channels=canale_intrare, out_channels=32, kernel_size=3, padding=1)
        self.strat_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.strat_conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.strat_conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.strat_conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.strat_conv6 = nn.Conv2d(in_channels=32, out_channels=canale_iesire, kernel_size=3, padding=1)

        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU()
        self.normalizare1 = nn.BatchNorm2d(32)
        self.normalizare2 = nn.BatchNorm2d(64)
        self.normalizare3 = nn.BatchNorm2d(128)
        self.normalizare4 = nn.BatchNorm2d(64)
        self.normalizare5 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, strat):
        strat = self.strat_conv1(strat)
        strat = self.normalizare1(strat)
        strat = self.relu(strat) # (b, 32, f, t)

        strat = self.pooling(strat) # (b, 32, f/2, t/2)

        strat = self.strat_conv2(strat)
        strat = self.normalizare2(strat)
        strat = self.relu(strat)  # (b, 64, f/2, t/2)
        strat = self.pooling(strat)  # (b, 64, f/4, t/4)
        strat = self.dropout(strat)

        strat = self.strat_conv3(strat)
        strat = self.normalizare3(strat)
        strat = self.relu(strat) # (b, 128, f/4, t/4)

        strat = self.upsample(strat) # (b, 128, f/2, t/2)

        strat = self.strat_conv4(strat)  # (b, 64, f/2, t/2)
        strat = self.normalizare4(strat)
        strat = self.relu(strat)

        strat = self.upsample(strat) # (b, 64, f, t)

        strat = self.strat_conv5(strat)  # (b, 32, f, t)
        strat = self.normalizare5(strat)
        strat = self.relu(strat)

        strat = self.strat_conv6(strat) # (b, 2, f, t)
        strat = self.sigmoid(strat)
        return strat
