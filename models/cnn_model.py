"""
Convolutional Neural Network (CNN) Model implementation in PyTorch.
Features multi-layer ConvNet with BatchNorm, Dropout, Adaptive Pooling,
and Data Augmentation for robust spatial feature learning on LEGO bricks.
"""
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models.base_model import BaseLEGOModel
import config

class LEGOCropDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        fpath = os.path.join(config.CROPPED_IMAGES_DIR, rec["relative_path"])
        # Load with PIL for standard torchvision transforms
        img = Image.open(fpath).convert("RGB")
        img = img.resize(config.CNN_INPUT_SIZE, Image.Resampling.BILINEAR)
        
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
            
        label = rec["class_idx"]
        return img_tensor, label

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.block(x)

class LEGOConvNet(nn.Module):
    def __init__(self, num_classes=20, dropout_rate=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 64x64 -> 32x32
            ConvBlock(32, 64),   # 32x32 -> 16x16
            ConvBlock(64, 128),  # 16x16 -> 8x8
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits

class LEGOCNN(BaseLEGOModel):
    def __init__(self, num_classes=20, lr=0.001, epochs=30, batch_size=64, 
                 dropout_rate=0.4, device=config.DEVICE):
        super().__init__(name="Convolutional Neural Network (CNN)")
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.device = device
        
        self.net = None
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Deterministic evaluation transform
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def fit_from_splits(self, train_records, val_records):
        start_time = time.perf_counter()
        self.net = LEGOConvNet(num_classes=self.num_classes, dropout_rate=self.dropout_rate).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        
        train_ds = LEGOCropDataset(train_records, transform=self.train_transform)
        val_ds = LEGOCropDataset(val_records, transform=self.eval_transform)
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        best_val_acc = 0.0
        best_weights = None
        
        print(f"\n--- Entrenando CNN con Data Augmentation [{self.epochs} épocas] ---")
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            running_loss = 0.0
            
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(labels)
                
            epoch_train_loss = running_loss / len(train_ds)
            self.train_losses.append(epoch_train_loss)
            
            # Validation
            self.net.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for v_imgs, v_labels in val_loader:
                    v_imgs, v_labels = v_imgs.to(self.device), v_labels.to(self.device)
                    v_out = self.net(v_imgs)
                    loss = criterion(v_out, v_labels)
                    val_loss += loss.item() * len(v_labels)
                    preds = torch.argmax(v_out, dim=1)
                    correct += (preds == v_labels).sum().item()
                    
            epoch_val_loss = val_loss / len(val_ds)
            epoch_val_acc = correct / len(val_ds)
            self.val_losses.append(epoch_val_loss)
            self.val_accuracies.append(epoch_val_acc)
            scheduler.step()
            
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_weights = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                
            if epoch % 5 == 0 or epoch == self.epochs:
                print(f"  Época {epoch:2d}/{self.epochs} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
                
        if best_weights is not None:
            self.net.load_state_dict(best_weights)
            
        self.train_time = time.perf_counter() - start_time
        self.plot_learning_curves()
        return self

    def predict_records(self, records):
        self.net.eval()
        ds = LEGOCropDataset(records, transform=self.eval_transform)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                outputs = self.net(imgs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
        return np.array(all_preds)

    def predict_single_image(self, img_bgr):
        """
        Inference on a single cv2 BGR image crop.
        Returns: predicted_class_idx, confidence, class_name
        """
        self.net.eval()
        # Convert BGR to RGB PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize(config.CNN_INPUT_SIZE, Image.Resampling.BILINEAR)
        tensor = self.eval_transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.net(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            
        return pred_idx, conf, config.IDX_TO_CLASS[pred_idx]

    def plot_learning_curves(self):
        out_dir = os.path.join(config.RESULTS_DIR, "models")
        os.makedirs(out_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(self.train_losses, label="Train Loss", color='crimson', linewidth=2)
        if self.val_losses:
            ax1.plot(self.val_losses, label="Val Loss", color='navy', linestyle='--', linewidth=2)
        ax1.set_title("Curva de Pérdida (CNN)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Épocas", fontsize=10)
        ax1.set_ylabel("Pérdida (Cross Entropy)", fontsize=10)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        if self.val_accuracies:
            ax2.plot([acc * 100 for acc in self.val_accuracies], label="Val Accuracy", color='darkgreen', linewidth=2)
            ax2.set_title("Exactitud de Validación (CNN)", fontsize=12, fontweight='bold')
            ax2.set_xlabel("Épocas", fontsize=10)
            ax2.set_ylabel("Exactitud (%)", fontsize=10)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        save_path = os.path.join(out_dir, "cnn_learning_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Curvas de aprendizaje CNN guardadas en: {save_path}")

