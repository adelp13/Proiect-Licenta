import os
import glob
import librosa
import matplotlib.pyplot as plt

path = r"E:\storage\Libri2Mix\wav16k\min\train-100"
folder_canal1 = os.path.join(path, "mic1")
folder_canal2 = os.path.join(path, "mic2")

files_mic1 = sorted(glob.glob(os.path.join(folder_canal1, "*.wav")))

for fpath1 in files_mic1[:10]:
    fname = os.path.basename(fpath1)
    fpath2 = os.path.join(folder_canal2, fname)

    s1, _ = librosa.load(fpath1, sr=None)
    s2, _ = librosa.load(fpath2, sr=None)

    plt.figure(figsize=(12, 4))
    plt.plot(s1, label='Mic 1', alpha=0.7)
    plt.plot(s2, label='Mic 2', alpha=0.7)
    plt.title(f"comparatie canale - {fname}")
    plt.legend()
    plt.show()
