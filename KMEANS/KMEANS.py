import numpy as np

# clusters = grupuri care se aseamana
#kmeans e un algortim de clustering
def kmeans(matrice, nr_clustere=2, nr_max_iteratii=100):
    # date(nr_puncte, nr_dimensiuni)
    nr_puncte, nr_dimensiuni = matrice.shape
    np.random.seed(0) # ca sa imi dea mereu aceleasi valori
    indici_clustere = np.random.choice(nr_puncte, nr_clustere, replace=False)
    centroizi = matrice[indici_clustere]
    etichete = []

    for _ in range(nr_max_iteratii):
        distante = np.zeros((nr_puncte, nr_clustere))
        for i in range(nr_clustere):
            distante[:, i] = np.linalg.norm(matrice - centroizi[i], axis=1)

        etichete = np.argmin(distante, axis=1)
        centroizi_vechi = centroizi.copy()
        for i in range(nr_clustere):
            puncte_cluster = matrice[etichete == i]
            if len(puncte_cluster) > 0:
                centroizi[i] = np.mean(puncte_cluster, axis=0)

        if np.linalg.norm(centroizi - centroizi_vechi) < 0.0001:
            break
    return centroizi, etichete