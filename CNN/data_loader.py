import torch
from torch.utils.data import Dataset
import soundfile as sf
import os
import numpy as np
import torchaudio.transforms as T

class SetDate(Dataset):
    def __init__(self, folder_base, caz="stereo", nr_exemple=3000):
        self.caz = caz
        self.folder_s1 = os.path.join(folder_base, "s1")
        self.folder_s2 = os.path.join(folder_base, "s2")
        self.folder_mic1 = os.path.join(folder_base, "mic1")
        self.folder_mic2 = os.path.join(folder_base, "mic2")
        self.folder_mono = os.path.join(folder_base, "mix_clean")
        self.nume_fisiere = sorted(os.listdir(self.folder_s1))[:nr_exemple]
        self.spectrograma = T.Spectrogram(n_fft=1024, hop_length=512, power=None, return_complex=True)

    def __len__(self):
        return len(self.nume_fisiere)

    def __getitem__(self, index):
        nume_fisier = self.nume_fisiere[index]
        s1, _ = sf.read(os.path.join(self.folder_s1, nume_fisier))
        s2, _ = sf.read(os.path.join(self.folder_s2, nume_fisier))
        surse = torch.tensor(np.array([s1, s2]), dtype=torch.float32)

        if self.caz == "mono":
            mono, _ = sf.read(os.path.join(self.folder_mono, nume_fisier))
            intrare = torch.tensor(np.array([mono]), dtype=torch.float32)
        else:
            mic1, _ = sf.read(os.path.join(self.folder_mic1, nume_fisier))
            mic2, _ = sf.read(os.path.join(self.folder_mic2, nume_fisier))
            intrare = torch.tensor(np.array([mic1, mic2]), dtype=torch.float32)

        return intrare, surse
