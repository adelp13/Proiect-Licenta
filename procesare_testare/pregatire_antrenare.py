import os
import glob
import librosa
import soundfile as sf

# pt a crea seturi de date de diferite dimensiuni
path = r"E:\storage\Libri2Mix\wav16k\min\train-100"
folder_sursa1 = os.path.join(path, "s1")
folder_sursa2 = os.path.join(path, "s2")
folder_canal1 = os.path.join(path, "mic1")
folder_canal2 = os.path.join(path, "mic2")
folder_mono = os.path.join(path, "mix_clean")

output_base_20000 = path + "_8000samples"
os.makedirs(output_base_20000, exist_ok=True)
output_folders_20000 = {
    "s1": os.path.join(output_base_20000, "s1"),
    "s2": os.path.join(output_base_20000, "s2"),
    "mic1": os.path.join(output_base_20000, "mic1"),
    "mic2": os.path.join(output_base_20000, "mic2"),
    "mix_clean": os.path.join(output_base_20000, "mix_clean"),
}

for f in output_folders_20000.values():
    os.makedirs(f, exist_ok=True)

nr_exemple = 8000

nume_fisiere = glob.glob(os.path.join(folder_sursa1, "*.wav"))
nume_fisiere = [os.path.basename(f) for f in nume_fisiere]
nume_fisiere = nume_fisiere[:nr_exemple]

for nume in nume_fisiere:
    paths = {
        "s1": os.path.join(folder_sursa1, nume),
        "s2": os.path.join(folder_sursa2, nume),
        "mic1": os.path.join(folder_canal1, nume),
        "mic2": os.path.join(folder_canal2, nume),
        "mix_clean": os.path.join(folder_mono, nume),
    }
    semnale = {k: librosa.load(v, sr=None)[0] for k, v in paths.items()}

    for j in semnale:
        out_path = os.path.join(output_folders_20000[j], nume)
        sf.write(out_path, semnale[j], samplerate=16000)

