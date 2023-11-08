import pandas as pd
import os

import torch
from emitopt.utils import compute_emits, get_quad_strength_conversion_factor

data = pd.read_csv("D:\\AWA\\auto_quad_scan\\run_data.txt", delimiter="\t")


beam_energy = 45e-3 # GeV
q_len = 0.12 # m
distance = 1.33-0.265 # m
gamma = beam_energy*1e3 / 0.511

data["grad"] = data["AWA:Bira3Ctrl:Ch04"] * 100 * 8.93e-3
data["int_grad"] = data["grad"] * q_len * 10
scale_factor = get_quad_strength_conversion_factor(beam_energy, q_len)
data["sx_m"] = 3.9232781168265036e-05 * data["13ARV1:Sx"]
data["sy_m"] = 3.9232781168265036e-05 * data["13ARV1:Sy"]


k = data.dropna()["int_grad"].to_numpy() * scale_factor
x = data.dropna()["sx_m"].to_numpy()*1e3
y = data.dropna()["sy_m"].to_numpy()*1e3

xemits, _, _,_ = compute_emits(
    torch.tensor(k), torch.tensor(x).unsqueeze(0), q_len, distance)
yemits, _, _,_ = compute_emits(
    torch.tensor(k), torch.tensor(y).unsqueeze(0), q_len, distance)
print((xemits**0.5)*gamma)
print((yemits**0.5)*gamma)

