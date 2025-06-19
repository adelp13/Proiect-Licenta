import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def medie_nula(semnal):
    medie = np.mean(semnal) # pt inversare
    return semnal - medie, medie

def dev_standard1(semnal):
    std = np.std(semnal)
    return semnal / std, std

def plot_spectrograma(signal, i, sr=16000, n_fft=1024, hop_length=512, nume="spectrograma"):
    plt.figure(figsize=(10, 4))
    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(abs(S), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(nume.upper(), fontsize=16, fontweight='bold')  # titlul mare și bold
    plt.tight_layout()
    plt.savefig(f"s_{nume}_{i}.pdf")
    plt.show()
    plt.close()

