import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, timp, dim_embedding):
        super().__init__()
        factori_scalare = np.array(
            [1 / (10000 ** (2 * (pozitie_embedding // 2) / dim_embedding)) for pozitie_embedding in
             range(dim_embedding)])  # (1, dim_embedding)
        pozitii_initiale = np.array([[p] for p in range(timp)])  # (timp, 1)
        valori = pozitii_initiale * factori_scalare  # (timp, dim_embedding)
        valori = torch.tensor(valori, dtype=torch.float32)
        rezultat = np.zeros((timp, dim_embedding))
        rezultat[:, 0::2] = torch.sin(valori[:, 0::2])
        rezultat[:, 1::2] = torch.cos(valori[:, 1::2])

        rezultat_tensor = torch.tensor(rezultat, dtype=torch.float32) # (timp, dim_embedding)
        rezultat_tensor = rezultat_tensor.unsqueeze(0) # (1, timp, emb)
        self.register_buffer('encoding', rezultat_tensor)

    def forward(self, strat):
        # strat e de forma (b, timp, emb)
        return strat + self.encoding[:, :strat.size(1)].to(strat.device)

class Atentie(nn.Module):
    def __init__(self, dim_embedding):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.Q_dense = nn.Linear(dim_embedding, dim_embedding)
        self.K_dense = nn.Linear(dim_embedding, dim_embedding)
        self.V_dense = nn.Linear(dim_embedding, dim_embedding)

    def forward(self, strat):
        Q = self.Q_dense(strat)
        K = self.K_dense(strat)
        V = self.V_dense(strat)

        scoruri_atentie = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_embedding ** 0.5) # (t, e) * (e, t) = (t, t)
        ponderi_atentie = torch.softmax(scoruri_atentie, dim=-1)
        rezultat = torch.matmul(ponderi_atentie, V)
        return rezultat

class TransformerEncoder(nn.Module):
    def __init__(self, dim_embedding, dim_ff):
        super().__init__()
        self.atentie = Atentie(dim_embedding)
        self.norm1 = nn.LayerNorm(dim_embedding)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_embedding, dim_ff), nn.ReLU(),
            nn.Linear(dim_ff, dim_embedding)
        )
        self.norm2 = nn.LayerNorm(dim_embedding)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, strat):
        strat_atentie = self.atentie(strat)
        strat_atentie = self.norm1(strat + self.dropout(strat_atentie))
        strat_ff = self.feed_forward(strat_atentie)
        strat = self.norm2(strat_atentie + self.dropout(strat_ff))
        return strat

class WaveUNet(nn.Module):
    def __init__(self, canale_intrare = 2, canale_iesire = 2, strat_atentie = False, nr_canale=[16, 32, 64, 128, 256, 512, 1024], timp = 64000):
        super().__init__()
        self.strat_atentie = strat_atentie
        self.canale_intrare = canale_intrare
        self.canale_iesire = canale_iesire
        self.nr_canale = nr_canale
        self.encoder_conv = nn.ModuleList()
        self.encoder_norm = nn.ModuleList()
        self.strat_atentie = strat_atentie

        # encoderul
        intrare = canale_intrare
        for i in range(len(nr_canale) - 1):
            strat_conv = nn.Conv1d(intrare, nr_canale[i], kernel_size=15, padding=7)
            strat_norm = nn.BatchNorm1d(nr_canale[i])
            self.encoder_conv.append(strat_conv)
            self.encoder_norm.append(strat_norm)
            intrare = nr_canale[i]

        self.bottleneck = nn.Conv1d(nr_canale[-2], nr_canale[-1], kernel_size=15, padding=7)
        self.bottleneck_norm = nn.BatchNorm1d(nr_canale[-1])

        # decoderul:
        self.decoder_conv = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.decoder_prelu = nn.ModuleList()

        for i in range(len(nr_canale) - 2, -1, -1):
            intrare = nr_canale[i] + nr_canale[i + 1]
            iesire = nr_canale[i]
            strat_conv = nn.Conv1d(intrare, iesire, kernel_size=15, padding=7)
            strat_norm = nn.InstanceNorm1d(iesire)
            self.decoder_conv.append(strat_conv)
            self.decoder_norm.append(strat_norm)
            self.decoder_prelu.append(nn.PReLU())

        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.final = nn.Conv1d(nr_canale[0], canale_iesire, kernel_size=1)
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.1)
        self.pooling = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.2)

        if self.strat_atentie:
            self.pos_enc = PositionalEncoding(timp=timp, dim_embedding=nr_canale[-1])
            self.transformer_enc = nn.ModuleList()
            for _ in range(5):
                self.transformer_enc.append(TransformerEncoder(dim_embedding=nr_canale[-1], dim_ff=nr_canale[-1]*4))

    def forward(self, strat):
        straturi_encoder = []
        # encoder
        for i in range(len(self.nr_canale) - 1):
            strat = self.leakyRelu(self.encoder_norm[i](self.encoder_conv[i](strat)))
            straturi_encoder.append(strat)
            strat = self.pooling(strat)

        strat = self.bottleneck(strat)
        strat = self.relu(self.bottleneck_norm(strat))

        if self.strat_atentie:
            strat = strat.transpose(1, 2) # (b, c, t) devine (b, t, c)
            strat = self.pos_enc(strat)
            for encoder in self.transformer_enc:
                strat = encoder(strat)
            strat = strat.transpose(1, 2)

        #decoder
        straturi_encoder = list(reversed(straturi_encoder))
        for i in range(len(self.nr_canale) - 1):
            strat = self.upsample(strat)
            strat = torch.cat([strat, straturi_encoder[i]], dim=1)
            strat = self.decoder_prelu[i](self.decoder_norm[i](self.decoder_conv[i](strat)))
            strat = self.dropout(strat)

        strat = torch.tanh(self.final(strat))
        return strat


class WaveUNetVarianta1(nn.Module):
    def __init__(self, canale_intrare = 2, canale_iesire = 2):
        super().__init__()
        self.canale_intrare = canale_intrare
        self.canale_iesire = canale_iesire

        self.strat_conv1 = nn.Conv1d(canale_intrare, 16, kernel_size=15, padding=7)
        self.strat_conv2 = nn.Conv1d(16, 32, kernel_size=15, padding=7)
        self.strat_conv3 = nn.Conv1d(32, 64, kernel_size=15, padding=7)

        self.pooling = nn.MaxPool1d(kernel_size=2)
        self.bottleneck = nn.Conv1d(64, 128, kernel_size=15, padding=7)

        self.strat_conv4 = nn.Conv1d(128 + 64, 64, kernel_size=15, padding=7)
        self.strat_conv5 = nn.Conv1d(64 + 32, 32, kernel_size=15, padding=7)
        self.strat_conv6 = nn.Conv1d(32 + 16, 16, kernel_size=15, padding=7)

        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.final = nn.Conv1d(16, canale_iesire, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, strat):
        # encoderul
        strat = self.strat_conv1(strat)
        strat1 = self.relu(strat) # (b, 16, t)
        strat2 = self.pooling(strat1) # (b, 16, t/2)

        strat2 = self.strat_conv2(strat2) # (b, 32, t/2)
        strat2 = self.relu(strat2)
        strat3 = self.pooling(strat2) # (b, 32, t/4)

        strat3 = self.strat_conv3(strat3)  # (b, 64, t/4)
        strat3 = self.relu(strat3)
        strat4 = self.pooling(strat3)  # (b, 64, t/8)

        # bottleneck
        strat4 = self.bottleneck(strat4)

        # decoder
        strat4 = self.upsample(strat4)
        strat = torch.cat([strat4, strat3], dim=1) # (b, 128 + 64, t/4), concatenare pe dimensiunea canalelor
        strat = self.relu(self.strat_conv4(strat)) # (b, 64, t/4)

        strat = self.upsample(strat) # (b, 64, t/2)
        strat = torch.cat([strat, strat2], dim=1)
        strat = self.relu(self.strat_conv5(strat)) # (b, 32, t/2)

        strat = self.upsample(strat)  # (b, 32, t)
        strat = torch.cat([strat, strat1], dim=1)
        strat = self.relu(self.strat_conv6(strat))  # (b, 16, t)

        strat = torch.tanh(self.final(strat))

        return strat
