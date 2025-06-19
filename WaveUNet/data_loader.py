import torch
from torch.utils.data import Dataset
import soundfile as sf
import os
import numpy as np

class SetDate(Dataset):
    def __init__(self, path, caz="stereo", nr_exemple=3000):
        self.caz = caz
        self.folder_s1 = os.path.join(path, "s1")
        self.folder_s2 = os.path.join(path, "s2")
        self.folder_mic1 = os.path.join(path, "mic1")
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
