import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed"
SAVE_PATH = "models"
os.makedirs(SAVE_PATH, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]

# ============================================================
# SPEC AUGMENT
# ============================================================

def spec_augment(spec, time_mask_param=20, freq_mask_param=15):
    spec = spec.clone()

    # Time mask
    t = random.randint(0, time_mask_param)
    t0 = random.randint(0, max(0, spec.shape[1] - t))
    spec[:, t0:t0+t] = 0

    # Frequency mask
    f = random.randint(0, freq_mask_param)
    f0 = random.randint(0, max(0, spec.shape[0] - f))
    spec[f0:f0+f, :] = 0

    return spec

# ============================================================
# DATASET
# ============================================================

class MusicDataset(Dataset):
    def __init__(self, X, y, train=False):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        spec = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)

        if self.train:
            spec = spec_augment(spec)

        spec = spec.unsqueeze(0)  # (1, 128, 130)
        return spec, label

# ============================================================
# MODELS
# ============================================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.squeeze(1).permute(0,2,1)  # (B,T,F)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class Hybrid_CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64*32,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv(x)          # (B,64,32,32)
        B,C,F,T = x.shape
        x = x.permute(0,3,1,2).reshape(B,T,C*F)
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model, train_loader, val_loader, name):

    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3
    )

    # LABEL SMOOTHING ADDED
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = 0
    patience = 8
    counter = 0

    train_acc_hist = []
    val_acc_hist = []

    for epoch in range(EPOCHS):

        # ---------------- TRAIN ----------------
        model.train()
        correct = total = 0
        running_loss = 0

        for X,y in train_loader:
            X,y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs,y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct/total

        # ---------------- VALIDATION ----------------
        model.eval()
        correct = total = 0
        val_loss = 0

        with torch.no_grad():
            for X,y in val_loader:
                X,y = X.to(DEVICE), y.to(DEVICE)

                outputs = model(X)
                loss = criterion(outputs,y)
                val_loss += loss.item()

                preds = outputs.argmax(1)
                correct += (preds==y).sum().item()
                total += y.size(0)

        val_acc = 100 * correct/total

        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
              f"Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%")

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # ---------- EARLY STOPPING ----------
        if val_acc > best_val:
            best_val = val_acc
            counter = 0
            torch.save(model.state_dict(),
                       os.path.join(SAVE_PATH, f"{name}_best.pth"))
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    return train_acc_hist, val_acc_hist

# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading data...")
    X = np.load(os.path.join(DATA_PATH,"spectrograms.npy"))
    y = np.load(os.path.join(DATA_PATH,"labels.npy"))

    # split
    n = len(X)
    idx = np.random.permutation(n)

    train_idx = idx[:int(0.7*n)]
    val_idx   = idx[int(0.7*n):int(0.85*n)]
    test_idx  = idx[int(0.85*n):]

    train_ds = MusicDataset(X[train_idx], y[train_idx], train=True)
    val_ds   = MusicDataset(X[val_idx], y[val_idx])
    test_ds  = MusicDataset(X[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds,BATCH_SIZE,shuffle=True)
    val_loader   = DataLoader(val_ds,BATCH_SIZE)
    test_loader  = DataLoader(test_ds,BATCH_SIZE)

    models = {
        "CNN": CNN(),
        "LSTM": LSTM(),
        "Hybrid_CNN_LSTM": Hybrid_CNN_LSTM()
    }

    for name,model in models.items():
        print(f"\nTRAINING {name}")
        train_model(model, train_loader, val_loader, name)


if __name__ == "__main__":
    main()
