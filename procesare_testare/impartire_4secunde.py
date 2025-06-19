import os
import glob
import librosa
import soundfile as sf

path = r"E:\storage\Libri2Mix\wav16k\min\train-100"
folder_sursa1 = os.path.join(path, "s1")
folder_sursa2 = os.path.join(path, "s2")
folder_canal1 = os.path.join(path, "mic1")
folder_canal2 = os.path.join(path, "mic2")
folder_mono = os.path.join(path, "mix_clean")

output_base = path + "_4s"
os.makedirs(output_base, exist_ok=True)
output_folders = {
    "s1": os.path.join(output_base, "s1"),
    "s2": os.path.join(output_base, "s2"),
    "mic1": os.path.join(output_base, "mic1"),
    "mic2": os.path.join(output_base, "mic2"),
    "mix_clean": os.path.join(output_base, "mix_clean"),
}
for f in output_folders.values():
    os.makedirs(f, exist_ok=True)

durata_sec = 4
rata_esantionare = 16000
esantioane_per_vocal = durata_sec * rata_esantionare

nume_fisiere = glob.glob(os.path.join(folder_sursa1, "*.wav"))
nume_fisiere = [os.path.basename(f) for f in nume_fisiere]

for nume in nume_fisiere:
    paths = {
        "s1": os.path.join(folder_sursa1, nume),
        "s2": os.path.join(folder_sursa2, nume),
        "mic1": os.path.join(folder_canal1, nume),
        "mic2": os.path.join(folder_canal2, nume),
        "mix_clean": os.path.join(folder_mono, nume),
    }
    semnale = {nume: librosa.load(path, sr=None)[0] for nume, path in paths.items()}

    total_len = len(semnale["s1"])
    nr_bucati = total_len // esantioane_per_vocal

    for i in range(nr_bucati):
        start = i * esantioane_per_vocal
        end = start + esantioane_per_vocal

        for j in semnale:
            bucata = semnale[j][start:end]
            nume_nou = os.path.splitext(nume)[0] + f"_{i+1}.wav"
            out_path = os.path.join(output_folders[j], nume_nou)
            sf.write(out_path, bucata, rata_esantionare)