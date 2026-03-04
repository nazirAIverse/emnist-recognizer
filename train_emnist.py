"""Train EMNIST (byclass, 62 classes) with preprocessing that matches the web app.

Creates:
  - model.pt  (best validation accuracy)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms

from model import Net

INVERT = False  # must match app.py

def fix_emnist_orientation(t: torch.Tensor) -> torch.Tensor:
    # Must match app.py: transpose H/W then flip vertically
    t = t.transpose(1, 2)
    t = torch.flip(t, [1])
    return t

class WebLikeTransform:
    def __init__(self, invert: bool = False):
        self.to_tensor = transforms.ToTensor()
        self.invert = invert

    def __call__(self, img):
        t = self.to_tensor(img)  # 0..1, (1,28,28)
        if self.invert:
            t = 1.0 - t

        t = fix_emnist_orientation(t)

        # normalize to -1..1 (same as (x-0.5)/0.5)
        t = (t - 0.5) / 0.5
        return t

@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tfm = WebLikeTransform(invert=INVERT)

    train_ds = EMNIST(root="data", split="byclass", train=True, download=True, transform=tfm)
    val_ds   = EMNIST(root="data", split="byclass", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = Net(num_classes=62).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    epochs = 2

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        val_acc = accuracy(model, val_loader, device)
        print(f"Epoch {ep:02d} | loss={total_loss/len(train_loader):.4f} | val_acc={val_acc*100:.2f}%")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), "model.pt")
            print(f"  Saved model.pt (best val_acc={best*100:.2f}%)")

    print("Done. Best val_acc:", best)

if __name__ == "__main__":
    main()
