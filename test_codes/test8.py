from network import PPOAgent
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{device=}")

net1 = PPOAgent().to(device)
net2 = PPOAgent().to(device)
net2.eval()
print(net2.training)
torch.save(net1.state_dict(), "test.pth")
net2.load_state_dict(torch.load("test.pth"))
# net2.eval()
print(net2.training)
