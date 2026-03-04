import torch
import torch.nn as nn
import torch.nn.functional as F

def idx_to_char(i: int) -> str:
    if 0 <= i <= 9:
        return chr(ord("0") + i)
    if 10 <= i <= 35:
        return chr(ord("A") + (i - 10))
    if 36 <= i <= 61:
        return chr(ord("a") + (i - 36))
    return "?"

class Net(nn.Module):
    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64 * 7 * 7, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
