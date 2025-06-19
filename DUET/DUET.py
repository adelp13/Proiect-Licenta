import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from KMEANS.KMEANS import kmeans

fs = 16000

class DUET:
    def __init__(self, canale, nr_surse=2, tip='double channel', nr_max_iteratii=5000):
        self.tip = tip
        self.canale = np.array(canale)
        self.nr_max_iteratii = nr_max_iteratii
        self.nr_surse = nr_surse
        self.surse_separate = []
        self.nr_canale, self.nr_esantioane = self.canale.shape

    def separare(self):
        # vocile vor ajunge la micrfoane cu un delay usor diferit si o amplitudine usor diferita
        if self.tip == 'double channel':
            esantioane_per_fft = 1024
            pas = 512  # deci avem 50% suprapunere
            spectograma1 = librosa.stft(self.canale[0, :], n_fft=esantioane_per_fft, hop_length=pas, window='hann')
            amplitudini1 = np.abs(spectograma1)
            faze1 = np.angle(spectograma1)
            spectograma2 = librosa.stft(self.canale[1, :], n_fft=esantioane_per_fft, hop_length=pas, window='hann')
            amplitudini2 = np.abs(spectograma2)
            nr_frecvente, nr_timpi = spectograma2.shape
            faze2 = np.angle(spectograma2)

            raport_amplitudini = np.abs(spectograma2 / spectograma1) # element wise
            frecvente = np.fft.rfftfreq(esantioane_per_fft, d=1/fs)

            frecvente[0] = 0.001
            omega = 2 * np.pi * frecvente
            #diferenta_faze = faze2 - faze1
            # diferenta_faze = -np.angle(spectograma2 / spectograma1) / omega[:, np.newaxis]
            diferenta_faze = -np.angle(spectograma2 / spectograma1) / omega[:, np.newaxis]
            #diferenta_faze = (diferenta_faze + np.pi) % (2 * np.pi) - np.pi

            amplitudine_minima = 0.02
            masca_valida = amplitudini1 > amplitudine_minima
            raport_valid = raport_amplitudini[masca_valida]
            faza_valid = diferenta_faze[masca_valida]

            # daca multe frecvente au aceeasi pereche, se vor grupa in acelasi loc
            # deci acl poate fi o sursa care ajunge la ambele microfoane
            # ne trebuie 2 masti binare (hard) complementare
            # gasim 2 peak-uri sau cate voci avem, si atribui, fiecare punct timp-frecventa sursei de care e cel mai close
            # histograma(b_bins_ampl, n_bins_faze)
            # matricea pt clustering are shape (nr_puncte, nr_dimensiuni=2)

            puncte = np.vstack((raport_valid, faza_valid)).T
            scaler = StandardScaler()
            puncte = scaler.fit_transform(puncte)
            #kmeans = KMeans(n_clusters=2, random_state=0).fit(puncte)
            #etichete = kmeans.labels_ # lungimea numarului de puncte
            _, etichete = kmeans(puncte)
            masca1 = np.zeros((nr_frecvente, nr_timpi))
            masca2 = np.zeros((nr_frecvente, nr_timpi))

            index_punct = 0
            for i in range(nr_frecvente):
                for j in range(nr_timpi):
                    if masca_valida[i, j]:
                        if etichete[index_punct] == 0:
                            masca1[i, j] = 1
                        else:
                            masca2[i, j] = 1
                        index_punct += 1

            # plt.figure(figsize=(12, 5))
            # plt.suptitle('MĂȘTILE', fontsize=16, fontweight='bold')
            #
            # plt.subplot(1, 2, 1)
            # librosa.display.specshow(masca1, sr=fs, hop_length=pas, y_axis='log', x_axis='time', cmap='gray_r')
            # plt.colorbar()
            #
            # plt.subplot(1, 2, 2)
            # librosa.display.specshow(masca2, sr=fs, hop_length=pas, y_axis='log', x_axis='time', cmap='gray_r')
            # plt.colorbar()
            #
            # plt.tight_layout(rect=[0, 0, 1, 0.95])
            # plt.show()

            sursa1 = masca1 * spectograma1
            sursa2 = masca2 * spectograma1

            sursa1 = librosa.istft(sursa1, hop_length=pas, window='hann', length=self.nr_esantioane)
            sursa2 = librosa.istft(sursa2, hop_length=pas, window='hann', length=self.nr_esantioane)

            self.surse_separate = np.vstack((sursa1, sursa2))

