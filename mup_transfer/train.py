from typing import Optional

import torch
import tqdm
import torch.nn.functional as F

from mup_transfer.loggers.logger import LoggerBase, NullLogger


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    clip_grad: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: torch.device = torch.device('cpu'),
    logger: LoggerBase = NullLogger(),
) -> tuple[float, float]:
    """
    Train the model for one epoch. Returns the training loss for that epoch and accuracy.
    """
    model.train()
    epoch_loss = 0.
    epoch_n_correct = 0
    for x, y in tqdm.tqdm(train_loader, desc="Training steps"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), y.view(-1))
        optim.zero_grad()
        loss.backward()

        # Clip gradients
        if clip_grad < float('inf'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optim.step()
        if scheduler is not None:
            scheduler.step()

        # Log metrics
        n_correct = (out.argmax(-1) == y).sum().detach().item()
        logger.log_scalar("train.batch_accuracy", n_correct / len(out))
        logger.log_scalar("train.batch_loss", loss.detach().item())
        logger.increment_step()

        epoch_loss += loss.detach().item() * len(x)
        epoch_n_correct += n_correct
    N = len(train_loader.dataset)
    return epoch_loss / N, epoch_n_correct / N
