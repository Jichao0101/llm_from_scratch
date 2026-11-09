import torch

@torch.compile
def elu(x, alpha=1.0):
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))

x = torch.randn(1024, device="cuda")

y = elu(x)

torch.cuda.synchronize()
print(y[:5])