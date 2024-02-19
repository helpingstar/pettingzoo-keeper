from network import PPOAgent
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{device=}")

net1 = PPOAgent().to(device)

print(next(net1.parameters()).device)
