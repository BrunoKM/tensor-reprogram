import torch
import torch.nn.functional as F


@torch.no_grad()
def eval(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device('cpu'),
) -> tuple[float, float]:
    model.eval()
    eval_loss = 0.
    n_correct = 0
    for X, y in eval_loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = F.cross_entropy(out, y, reduction='sum')
        eval_loss += loss
        n_correct += (out.argmax(-1) == y).sum()
    N = len(eval_loader.dataset)
    return eval_loss.item() / N, n_correct.item() / N
