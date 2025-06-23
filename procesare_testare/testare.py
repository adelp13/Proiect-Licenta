import os
import glob
import librosa
from ICA.ICA import ICA
from DUET.DUET import DUET
from NMF.NMF import NMF
from WaveUNet.WaveUNet import WaveUNet
from CNN.CNN import CNN
import numpy as np
import mir_eval
import time
import soundfile as sf
from utilities import medie_nula, dev_standard1, plot_spectrograma
import matplotlib.pyplot as plt
import torch

caz = "stereo"
model = "ica" # ica, nmf, duet, cnn, wave, wave_transf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = r"E:\storage\Libri2Mix\wav16k\min\train-100_4s"
folder_sursa1 = os.path.join(path, "s1")
folder_sursa2 = os.path.join(path, "s2")
folder_canal1 = os.path.join(path, "mic1")
folder_canal2 = os.path.join(path, "mic2")
folder_mono = os.path.join(path, "mix_clean")

nume_fisiere = sorted(glob.glob(os.path.join(folder_sursa1, "*.wav")))[10000:10020]
nume_fisiere = [os.path.basename(f) for f in nume_fisiere]
timpi = []
sdrs1 = []
sdrs2 = []
sirs1 = []
sirs2 = []
sars1 = []
sars2 = []

def aranjeaza_permutare(surse_originale, surse_separate):
    sdr, sar, sir, perm = mir_eval.separation.bss_eval_sources(surse_originale, surse_separate)
    surse_separate_corectate = np.zeros_like(surse_separate)
    for i, p in enumerate(perm):
        surse_separate_corectate[p] = surse_separate[i]
    sdr_corectat = sdr[perm.argsort()]
    sar_corectat = sar[perm.argsort()]
    sir_corectat = sir[perm.argsort()]
    return surse_separate_corectate, sdr_corectat, sir_corectat, sar_corectat

surse_separate = []
def testare(caz="mono", model="nmf"):
    for nume in nume_fisiere:
        cale_sursa1 = os.path.join(folder_sursa1, nume)
        cale_sursa2 = os.path.join(folder_sursa2, nume)
        cale_canal1 = os.path.join(folder_canal1, nume)
        cale_canal2 = os.path.join(folder_canal2, nume)
        cale_mono = os.path.join(folder_mono, nume)
        sursa1, _ = librosa.load(cale_sursa1, sr=None)
        sursa2, _ = librosa.load(cale_sursa2, sr=None)
        canal1, _ = librosa.load(cale_canal1, sr=None)
        canal2, _ = librosa.load(cale_canal2, sr=None)
        mono, _ = librosa.load(cale_mono, sr=None)

        start = time.time()
        medie_mono, dev_mono, medie1_stereo, medie2_stereo, dev1_stereo, dev2_stereo = 0, 0, 0, 0, 0, 0

        if caz == "mono":
            mono, medie_mono = medie_nula(mono)
            mono, dev_mono = dev_standard1(mono)
            if model == "nmf":
                nmf = NMF(mono)
                nmf.separare()
                surse_separate = np.array(nmf.surse_separate)
            elif model == "wave":
                model_wave_1 = WaveUNet(canale_intrare=1, strat_atentie=False)
                model = WaveUNet(canale_intrare=1, strat_atentie=False)  # sau True, vezi cum ai salvat!
                model_wave_1.load_state_dict(torch.load("../WaveUNet/model_wave_1.pth", map_location=device))
                model_wave_1.to(device)
                model_wave_1.eval()
                mono = torch.tensor(mono, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, t) #pt mono de 2 ori unsqueeze, la stereo era deja (2, t), nu doar (t)
                iesire = model_wave_1(mono).squeeze(0).cpu().detach().numpy()  # (2, t)
                surse_separate = iesire

            elif model == "wave_transf":
                model_wave_transf_1 = WaveUNet(canale_intrare=1, strat_atentie=True)
                model_wave_transf_1.load_state_dict(torch.load("../WaveUNet/model_wave_transf_1.pth", map_location=device))
                model_wave_transf_1.to(device)
                model_wave_transf_1.eval()
                mono = torch.tensor(mono, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                iesire = model_wave_transf_1(mono).squeeze(0).cpu().detach().numpy()
                surse_separate = iesire
        else:
            canal1, medie1_stereo = medie_nula(canal1)
            canal2, medie2_stereo = medie_nula(canal2)
            if model != "ica":
                canal1, dev1_stereo = dev_standard1(canal1)
                canal2, dev2_stereo = dev_standard1(canal2)
            canale = np.vstack([canal1, canal2])
            if model == "ica":
                ica = ICA(canale)
                ica.separare()
                surse_separate = ica.surse_separate
            elif model == "duet":
                duet = DUET(canale)
                duet.separare()
                surse_separate = duet.surse_separate

            elif model == "wave":
                model_wave_2 = WaveUNet(canale_intrare=2, strat_atentie=False)
                model_wave_2.load_state_dict(torch.load("../WaveUNet/model_wave_2.pth", map_location=device))
                model_wave_2.to(device)
                model_wave_2.eval()
                canale = torch.tensor(canale, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 2, t)
                iesire = model_wave_2(canale).squeeze(0).cpu().detach().numpy()  # (2, t)
                surse_separate = iesire
            elif model == "wave_transf":
                model_wave_transf_2 = WaveUNet(canale_intrare=2, strat_atentie=True)
                model_wave_transf_2.load_state_dict(torch.load("../WaveUNet/model_wave_transf_2.pth", map_location=device))
                model_wave_transf_2.to(device)
                model_wave_transf_2.eval()
                canale = torch.tensor(canale, dtype=torch.float32).unsqueeze(0).to(device)
                iesire = model_wave_transf_2(canale).squeeze(0).cpu().detach().numpy()
                surse_separate = iesire

        # sf.write("prima_sursa.wav",surse_separate[0], sr)
        # sf.write("a_doua_sursa.wav", surse_separate[1], sr)
        end = time.time()
        durata = end - start
        timpi.append(durata)

        if caz == "stereo":
            if model != "ica":
                surse_separate[0] = surse_separate[0] * dev1_stereo + medie1_stereo
                surse_separate[1] = surse_separate[1] * dev2_stereo + medie2_stereo
            # surse_separate = surse_separate[:, 240:-240]
            # sursa1 = sursa1[240:-240]
            # sursa2 = sursa2[240:-240]
        else:
            surse_separate[0] = surse_separate[0] * dev_mono + medie_mono
            surse_separate[1] = surse_separate[1] * dev_mono + medie_mono

        surse_separate[0] = surse_separate[0] / np.max(np.abs(surse_separate[0]))
        surse_separate[1] = surse_separate[1] / np.max(np.abs(surse_separate[1]))

        sursa1 = sursa1 / np.max(np.abs(sursa1))
        sursa2 = sursa2 / np.max(np.abs(sursa2))
        surse = np.vstack([sursa1, sursa2])

        surse_separate_corectate, sdr, sir, sar = aranjeaza_permutare(surse, surse_separate)

        print("SDR per sursa:", sdr)
        print("SIR per sursa:", sir)
        print("SAR per sursa:", sar, '\n')

        sirs1.append(sir[0])
        sirs2.append(sir[1])
        sdrs1.append(sdr[0])
        sdrs2.append(sdr[1])
        sars1.append(sar[0])
        sars2.append(sar[1])

    timp_mediu = sum(timpi) / len(timpi)
    sirs1_mediu = sum(sirs1) / len(sirs1)
    sirs2_mediu = sum(sirs2) / len(sirs2)
    sdrs1_mediu = sum(sdrs1) / len(sdrs1)
    sdrs2_mediu = sum(sdrs2) / len(sdrs2)
    sars1_mediu = sum(sars1) / len(sars1)
    sars2_mediu = sum(sars2) / len(sars2)

    sdr_mediu_total = (sdrs1_mediu + sdrs2_mediu) / 2
    sir_mediu_total = (sirs1_mediu + sirs2_mediu) / 2
    sar_mediu_total = (sars1_mediu + sars2_mediu) / 2

    print(f"timpu mediu: {timp_mediu:.4f} secunde")
    print(f"SIR mediu sursa 1: {sirs1_mediu:.2f} dB")
    print(f"SIR mediu sursa 2: {sirs2_mediu:.2f} dB")
    print(f"SDR mediu sursa 1: {sdrs1_mediu:.2f} dB")
    print(f"SDR mediu sursa 2: {sdrs2_mediu:.2f} dB")
    print(f"SAR mediu sursa 1: {sars1_mediu:.2f} dB")
    print(f"SAR mediu sursa 2: {sars2_mediu:.2f} dB")
    print(f"SIR mediu total: {sir_mediu_total:.2f} dB")
    print(f"SAR mediu total: {sar_mediu_total:.2f} dB")
    print(f"SDR mediu total: {sdr_mediu_total:.2f} dB")

testare(caz, model)
# testare("stereo", "wave_transf")

