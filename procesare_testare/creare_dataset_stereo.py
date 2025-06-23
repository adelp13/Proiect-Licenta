import os
import numpy as np
import soundfile as sf

def intarziere(semnal, esantioane):
    if esantioane > 0:
        return np.concatenate((np.zeros(esantioane), semnal[:-esantioane]))
    else:
        return np.concatenate((semnal[-esantioane:], np.zeros(-esantioane)))

def amestec(surse, intarzieri, ponderi):
    mix = np.zeros((2, len(surse[0])))

    canal1_s1 = intarziere(surse[0], intarzieri[0, 0])
    canal1_s2 = intarziere(surse[1], intarzieri[0, 1])
    mix[0, :len(surse[0])] = ponderi[0, 0] * canal1_s1[:len(surse[0])] + ponderi[0, 1] * canal1_s2[:len(surse[0])]

    canal2_s1 = intarziere(surse[0], intarzieri[1, 0])
    canal2_s2 = intarziere(surse[1], intarzieri[1, 1])
    mix[1, :len(surse[0])] = ponderi[1, 0] * canal2_s1[:len(surse[0])] + ponderi[1, 1] * canal2_s2[:len(surse[0])]

    return mix

path = r"E:\storage\Libri2Mix\wav16k\min\train-100"
folder_s1 = os.path.join(path, "s1")
folder_s2 = os.path.join(path, "s2")
folder_mic1 = os.path.join(path, "mic1")
folder_mic2 = os.path.join(path, "mic2")

os.makedirs(folder_mic1, exist_ok=True)
os.makedirs(folder_mic2, exist_ok=True)

files_s1 = [f for f in os.listdir(folder_s1) if f.endswith('.wav')]

for fname in files_s1:
    path_s1 = os.path.join(folder_s1, fname)
    path_s2 = os.path.join(folder_s2, fname)

    s1, _ = sf.read(path_s1)
    s2, _ = sf.read(path_s2)
    surse = [s1, s2]

    intarziere_max = 240

    intarzieri = np.random.randint(-intarziere_max, intarziere_max, size=(2, 2))
    ponderi = np.random.uniform(0.2, 1.3, size=(2, 2))

    mix = amestec(surse, intarzieri, ponderi)

    max_val = np.max(np.abs(mix))
    if max_val > 1.0:
        mix = mix / max_val

    sf.write(os.path.join(folder_mic1, fname), mix[0], 16000)
    sf.write(os.path.join(folder_mic2, fname), mix[1], 16000)
    # print(intarzieri)
    # print(ponderi)
