import torch
import torch.nn.functional as F


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device = torch.device('cpu'),
) -> tuple[float, float]:
    model.train()
    train_loss = 0.
    n_correct = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = F.cross_entropy(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss * len(X)
        n_correct += (out.argmax(-1) == y).sum()
    N = len(train_loader.dataset)
    return train_loss.item() / N, n_correct.item() / N
