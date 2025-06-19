import numpy as np
import matplotlib.pyplot as plt
from ICA.ICA import ICA
from scipy import signal

fs = 1000
t = np.linspace(0, 1, fs)

s1 = np.sin(8 * np.pi * t)
s2 = np.sign(np.cos(8 * np.pi * t))

canale = np.vstack([s1, s2])
A = np.array([[0.8, 0.7],
              [0.7, 0.8]])

mix = A @ canale
mix = mix / np.max(np.abs(mix))
ica = ICA(mix)
ica.separare()
surse_separate = ica.surse_separate
canale_whitened = ica.canale_whitened

plt.figure(figsize=(12, 5))
plt.subplots_adjust(hspace=0.35)

plt.subplot(4, 1, 1)
plt.plot(t, canale[0])
plt.plot(t, canale[1])
plt.text(-0.04, 0.5, 'SURSELE INIȚIALE', fontsize=28, color='black',
         transform=plt.gca().transAxes, ha='right', va='center', fontweight='bold')

plt.subplot(4, 1, 2)
plt.plot(t, mix[0])
plt.plot(t, mix[1])
plt.text(-0.04, 0.5, 'AMESTECURI', fontsize=28, color='black',
         transform=plt.gca().transAxes, ha='right', va='center', fontweight='bold')

plt.subplot(4, 1, 3)
plt.plot(t, canale_whitened[0])
plt.plot(t, canale_whitened[1])
plt.text(-0.04, 0.5, 'WHITENING', fontsize=28, color='black',
         transform=plt.gca().transAxes, ha='right', va='center', fontweight='bold')

plt.subplot(4, 1, 4)
plt.plot(t, surse_separate[0])
plt.plot(t, surse_separate[1])
plt.text(-0.04, 0.5, 'SURSE SEPARATE', fontsize=28, color='black',
         transform=plt.gca().transAxes, ha='right', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig("toti_pasii2.pdf")
plt.show()
