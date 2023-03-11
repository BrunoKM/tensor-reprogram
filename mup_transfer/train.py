import torch
import tqdm
import torch.nn.functional as F


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device = torch.device('cpu'),
) -> tuple[float, float]:
    """
    Train the model for one epoch. Returns the training loss for that epoch and accuracy.
    """
    model.train()
    train_loss = 0.
    n_correct = 0
    for X, y in tqdm.tqdm(train_loader, desc="Training steps"):
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
