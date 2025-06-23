from data_loader import SetDate
from torch.utils.data import DataLoader
import torch
from WaveUNet import WaveUNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

canale_intrare = 2
strat_atentie = True
caz = "stereo" if canale_intrare == 2 else "mono"
nr_exemple = 3000
save = True
nr_epoci = 8

folder_antrenare = r"E:\storage\Libri2Mix\wav16k\min\train-100_8000samples"
set_antrenare = SetDate(folder_antrenare, caz=caz, nr_exemple=nr_exemple)
dataloader_antrenare = DataLoader(set_antrenare, batch_size=4, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WaveUNet(canale_intrare=canale_intrare, strat_atentie=strat_atentie, timp=64000)
model.to(device)

optimizator = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizator, step_size=2, gamma=0.4) # cand mse se modifica mai greu ne trebuie pasi mai mici, ne apropiem de punctul optim

def permutare_mse_loss(iesire, surse):
    mse1 = (F.mse_loss(iesire[:, 0], surse[:, 0]) + F.mse_loss(iesire[:, 1], surse[:, 1])) / 2
    mse2 = (F.mse_loss(iesire[:, 0], surse[:, 1]) + F.mse_loss(iesire[:, 1], surse[:, 0])) / 2
    return torch.min(mse1, mse2)

istoric_loss = []

for epoca in range(nr_epoci):
    model.train()
    loss_total = 0
    for intrare, raspunsuri in dataloader_antrenare:
        intrare = intrare.to(device)
        raspunsuri = raspunsuri.to(device)

        optimizator.zero_grad()
        medie = intrare.mean(dim=-1, keepdim=True)
        dev = intrare.std(dim=-1, keepdim=True) + 0.0000000001
        intrare = (intrare - medie) / dev
        iesire = model(intrare)
        iesire = iesire * dev + medie
        iesire = iesire / torch.max(torch.abs(iesire), dim=-1, keepdim=True)[0]
        raspunsuri = raspunsuri / torch.max(torch.abs(raspunsuri), dim=-1, keepdim=True)[0]
        # e bine sa normalizam intre -1 1 (standardul pt wav) pt ca intr-un mix o sursa poate avea ponderea 0.5,
        # deci rezultatul ar avea amplitudini mai mici decat ground truth, chiar daca forma similara
        loss = permutare_mse_loss(iesire, raspunsuri)
        loss.backward()
        optimizator.step()
        loss_total += loss.item()

    loss_mediu = loss_total / len(dataloader_antrenare)
    istoric_loss.append(loss_mediu)
    print(f"epoca {epoca + 1} - loss mediu: {loss_mediu:.6f}")
    scheduler.step()

if save:
    if strat_atentie:
        torch.save(model.state_dict(), f"model_wave_transf_{canale_intrare}.pth")
    else:
        torch.save(model.state_dict(), f"model_wave_{canale_intrare}.pth")

plt.figure(figsize=(8, 4))
plt.plot(istoric_loss, marker='o')
plt.title("loss pe parcurs")
plt.xlabel("epoca")
plt.ylabel("loss mediu")
plt.grid(True)
plt.tight_layout()
plt.show()
