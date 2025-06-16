import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


class ICA:
    def __init__(self, canale, tip='fast ICA', nr_max_iteratii=2000, aplicare_whitening=True):
        self.aplicare_whitening = aplicare_whitening
        self.canale = np.array(canale)
        self.canale = canale - np.mean(canale, axis=1, keepdims=True)
        self.nr_canale, self.nr_esantioane = self.canale.shape
        self.canale_whitened = []
        self.nr_max_iteratii = nr_max_iteratii
        self.surse_separate = []
        self.tip = tip
        self.canale -= self.canale.mean(axis=1, keepdims=True)  # normalizam datele

    def whitening(self):
        matrice_covarianta = (self.canale @ self.canale.T) / self.nr_esantioane
        # vrem ca matricea de covarianta sa devina I,
        # matrice_covarianta = daca canalele sunt independente unu de altul sau au valori care se misca in acelasi sens, cum variaza canalele intre ele
        # facem decompozitia eigen a matricei de covarianta curente, putem pt ca e patrata si simetria
        # o descompunem in  valori proprii si vectori de valori proprii, de forma C = autovectori @ autovalori @ autovectori.T
        # autovectori = directiile principale de variatie a datelor, autovalori = modulul directiilor
        autovalori, autovectori = np.linalg.eigh(matrice_covarianta)
        matrice_whitening = autovectori @ np.diag(1.0 / np.sqrt(autovalori + 0.0000000001)) @ autovectori.T # de la dreapta la stanga, rotim in sistemul de directii, scalam, rotim inapoi
        self.canale_whitened = matrice_whitening @ self.canale

    def functie_cost(self, val):
        return np.tanh(val)

    def derivata(self, val):
        return 1 - np.tanh(val)**2

    def separare(self):
        if self.aplicare_whitening:
            self.whitening()

        # teorema centrala limita, sursa idependenta este mai non-gaussiana
        matrice_separare = np.random.rand(self.nr_canale, self.nr_canale) # matricea de amestecare aceeasi dimensiune avea

        if self.tip == 'fast ICA': # cel mai des folosit
            # cautam directiile in care semnaele au cea mai mare non-gaussianitate, mergem in acele directii
            for i in range(self.nr_canale):
                directie_canal_curent = matrice_separare[i, :].copy()

                for _ in range(self.nr_max_iteratii):
                    proiectie_canal_curent = directie_canal_curent.T @ self.canale_whitened # separarea curenta doar pt un canal
                    # proiecte = directie @ canale
                    g = self.functie_cost(proiectie_canal_curent) # masoara non gaussianitatea pt fiecare esantion
                    # functia g aplatizeaza valorile mari si le evidentiaza pe cele mici
                    # directie = canale @ g(proiectie) - d(proiectie).mean() * directie

                    # non gaussianitatea se masoara prin negentropia aproximata
                    medie_ponderata = (self.canale_whitened * g).mean(axis=1) # fiecare esantion primeste o pondere in fct de cat a contribuit la non-gaussianitate, e componenta de urcare
                    corectie_scalare = self.derivata(proiectie_canal_curent).mean() * directie_canal_curent # normalizare practic
                    directie_canal_curent_nou = medie_ponderata - corectie_scalare
                    # e un fel de urcare pe gradient pt a maximiza non gaussianitatea, sursele independente sunt cele mai non-gaussiene

                    # pasul de gram-schmidt, cele n directii trb sa fie independenti intre ei si ortogonali
                    # daca directiile nu sunt complet perp, sursele rezultate nu vor fi complet independente
                    for j in range(i):
                        # formula: np.dot(v, u) * u = proiectia vectorului v pe vectorul u
                        produs_scalar_ij = np.dot(directie_canal_curent_nou, matrice_separare[j]) # intre o directie finalizata deja si cea curenta
                        # produsul scalar masoara alinierea dintre cele 2 directii, trb sa fie 0 pt a fi independnete
                        directie_canal_curent_nou -= produs_scalar_ij * matrice_separare[j]
                        # daca dintr-un vector scad proiectia lui pe un alt vector, ii fac perpendiculari
                    directie_canal_curent_nou /= np.linalg.norm(directie_canal_curent_nou) # transforma in vector unitate(de norma1), pt a retine doar directia

                    produs_scalar_ii = np.dot(directie_canal_curent_nou, directie_canal_curent)
                    # ele au norma 1, deci prod intre -1 si 1.
                    if np.abs(np.abs(produs_scalar_ii) - 1) < 0.00001:
                        # cand diferenta dintre vechi si nou nu e semnificativa, am gasit o directie potrivita
                        break

                matrice_separare[i, :] = directie_canal_curent_nou

        self.surse_separate = matrice_separare @ self.canale_whitened

        for i in range(self.surse_separate.shape[0]):
            max_abs = np.max(np.abs(self.surse_separate[i]))
            if max_abs > 0:
                self.surse_separate[i] /= max_abs
