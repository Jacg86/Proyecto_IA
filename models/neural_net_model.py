"""
Artificial Neural Network (ANN / MLP) Model implementation in PyTorch.
Features dense layers, Batch Normalization, Dropout, Adam optimizer,
learning rate scheduling, and early stopping.
"""
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.base_model import BaseLEGOModel
import config

class MLPNet(nn.Module):
    def __init__(self, input_dim, num_classes=20, hidden_dims=(256, 128), dropout_rate=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LEGONeuralNetwork(BaseLEGOModel):
    def __init__(self, input_dim=325, num_classes=20, hidden_dims=(256, 128), lr=0.001, 
                 epochs=45, batch_size=64, weight_decay=1e-4, device=config.DEVICE):
        super().__init__(name="Neural Network (ANN)")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        
        self.net = None
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        start_time = time.perf_counter()
        self.input_dim = X_train.shape[1]
        self.net = MLPNet(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Prepare datasets
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        has_val = (X_val is not None and y_val is not None)
        if has_val:
            val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        best_val_acc = 0.0
        best_weights = None
        
        print(f"\n--- Entrenando Red Neuronal Artificial (ANN) [{self.epochs} épocas] ---")
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            running_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_y)
                
            epoch_train_loss = running_loss / len(train_ds)
            self.train_losses.append(epoch_train_loss)
            
            if has_val:
                self.net.eval()
                val_loss = 0.0
                correct = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        vo = self.net(vx)
                        v_loss = criterion(vo, vy)
                        val_loss += v_loss.item() * len(vy)
                        preds = torch.argmax(vo, dim=1)
                        correct += (preds == vy).sum().item()
                        
                epoch_val_loss = val_loss / len(val_ds)
                epoch_val_acc = correct / len(val_ds)
                self.val_losses.append(epoch_val_loss)
                self.val_accuracies.append(epoch_val_acc)
                scheduler.step(epoch_val_acc)
                
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_weights = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                    
                if epoch % 10 == 0 or epoch == self.epochs:
                    print(f"  Época {epoch:2d}/{self.epochs} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
            else:
                if epoch % 10 == 0 or epoch == self.epochs:
                    print(f"  Época {epoch:2d}/{self.epochs} -> Train Loss: {epoch_train_loss:.4f}")
                    
        if best_weights is not None:
            self.net.load_state_dict(best_weights)
            
        self.train_time = time.perf_counter() - start_time
        self.plot_learning_curves()
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.net(tx)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.net(tx)
            probas = torch.softmax(logits, dim=1).cpu().numpy()
        return probas

    def plot_learning_curves(self):
        out_dir = os.path.join(config.RESULTS_DIR, "models")
        os.makedirs(out_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(self.train_losses, label="Train Loss", color='blue', linewidth=2)
        if self.val_losses:
            ax1.plot(self.val_losses, label="Val Loss", color='red', linestyle='--', linewidth=2)
        ax1.set_title("Curva de Pérdida (ANN / MLP)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Épocas", fontsize=10)
        ax1.set_ylabel("Pérdida (Cross Entropy)", fontsize=10)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        if self.val_accuracies:
            ax2.plot([acc * 100 for acc in self.val_accuracies], label="Val Accuracy", color='green', linewidth=2)
            ax2.set_title("Exactitud de Validación (ANN / MLP)", fontsize=12, fontweight='bold')
            ax2.set_xlabel("Épocas", fontsize=10)
            ax2.set_ylabel("Exactitud (%)", fontsize=10)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        save_path = os.path.join(out_dir, "ann_learning_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Curvas de aprendizaje ANN guardadas en: {save_path}")

