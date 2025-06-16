import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

class NMF:
    def __init__(self, canal, nr_surse=2, tip='single channel', nr_max_iteratii=3000):
        self.tip = tip
        self.canal = canal
        self.nr_max_iteratii = nr_max_iteratii
        self.nr_surse = nr_surse
        self.surse_separate = []

    def separare(self):
        # calculam spectograma stft, adica fft dar pe ferestre scurte cu suprapunere
        # deci un canal cu surse amestecate devine o matrice (momentele de timp in functie de frecvente)
        # spectograma = componente_spectrale * coeficienti_activare
        esantioane_per_fft = 1024
        pas = 512# deci avem 50% suprapunere
        # cu cat esantioane_per_fft e mai mare, cu atat mai mult ainformatie frecventiala, dar mai putina temporala
        # pas mai mic => mai multa informatie temporala
        spectograma = librosa.stft(self.canal, n_fft=esantioane_per_fft, hop_length=pas, window='hann')
        amplitudini = np.abs(spectograma) # ampltudinea adica cata energie are frecventa
        faze = np.angle(spectograma) # cat de decalate sunt frecventele

        a, b = amplitudini.shape
        M = np.abs(np.random.rand(a, self.nr_surse)) # o sursa pe fiecare coloana, frecventele ei
        N = np.abs(np.random.rand(self.nr_surse, b)) # sursele in spatiul timp, cand e fiecare sursa activa si cat de mult
        eps = 1e-10
        # nmf minimizeaza o fct care masoara cat de bine aproximam S
        # raportul (M.T @ amplitudini) / (M.T @ M @ N) e aproape de 1 pt aproximare buna, daca e subuniar, se reduce N pt ca era prea mare. inmultiri sa pastram valorile poz
        for _ in range(self.nr_max_iteratii):
            N *= (M.T @ amplitudini) / (M.T @ M @ N + 0.0000000001)
            M *= (amplitudini @ N.T) / (M @ N @ N.T + 0.0000000001)
        # # lee and seung
        # vrem ca produsul lor sa fie cat mai aproape de spectograma
        amplitudini_aprox = M @ N
        # (frecvente, timp) = (frecvente, surse) * (surse, timp)


        for i in range(self.nr_surse):
            # ne folosim de coloana i din spatiul frecventa, si de linia i din spatiul timp
            frecvente = M[:, i ]
            timpi = N[i, :]
            sursa = np.outer(frecvente, timpi)  # (frecvente, timpi), spectograma doar pt sursa curenta
            # e**(i*teta)=cos(teta)+i*sin(faza), un nr complex pe cercul unitate
            # sursa_compNMlexa = sursa * np.exp(1j * faze)
            masca = sursa / (amplitudini_aprox + 1e-10)
            sursa_complexa = masca * spectograma # iau doar portiunile care apartin sursei
            sursa_separata = librosa.istft(sursa_complexa, hop_length=pas, window='hann')

            #sursa_separata = librosa.griffinlim(sursa, hop_length=pas)
            sursa_separata = sursa_separata / np.max(np.abs(sursa_separata))
            self.surse_separate.append(sursa_separata)
