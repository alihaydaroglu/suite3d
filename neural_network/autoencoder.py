import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import HDF5Dataset, Augmentation3D, InfoNCELoss
from tqdm import tqdm
import os
import numpy as np
from umap import UMAP


class Autoencoder3D(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        
        # Encoder: (1, 5, 20, 20) -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=(1, 2, 2), padding=1),  # -> (16, 5, 10, 10)
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1), # -> (32, 5, 5, 5)
            nn.ReLU(),
        )
        
        # Bottleneck
        self.flatten_size = 32 * 5 * 5 * 5  # 4000
        self.fc_encode = nn.Linear(self.flatten_size, 128)
        self.fc_decode = nn.Linear(128, self.flatten_size)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        
        # Decoder: latent_dim -> (1, 5, 20, 20)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)), # -> (16, 5, 10, 10)
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # -> (1, 5, 20, 20)
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        """Encode to latent representation"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encode(x)
        return x
    
    def decode(self, z):
        """Decode from latent representation"""
        x = self.fc_decode(z)
        x = x.view(x.size(0), 32, 5, 5, 5)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Full autoencoder forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def save_ckpt(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),}, path)


def train_autoencoder(model, dataloader, num_epochs=50, lr=1e-3, device='cuda'):
    """Stage 1: Train as standard autoencoder"""
    model = model.to(device)
    print(f"Using device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for aug1, aug2 in tqdm(dataloader):
            # Assuming batch shape: (B, 5, 20, 20)
            images = aug1.to(device)

            optimizer.zero_grad()
            recon, _ = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Reconstruction Loss: {avg_loss:.6f}")


def train_contrastive(model, dataloader, loss_func, num_epochs=50, lr=1e-4, device='cuda'):
    """Stage 2: Train with contrastive learning"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for aug1, aug2 in dataloader:
            aug1, aug2 = aug1.to(device), aug2.to(device)
            
            optimizer.zero_grad()
            
            # Encode both views
            z1 = model.encode(aug1)
            z1 = model.projection_head(z1)
            z2 = model.encode(aug2)
            z2 = model.projection_head(z2)
            
            # Contrastive loss
            loss = loss_func(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Contrastive Loss: {avg_loss:.6f}")


def extract_features(model, dataloader, device):
    """Extract features for downstream analysis (UMAP, etc.)"""
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for view1, _ in tqdm(dataloader, desc='Extracting features'):
            view1 = view1.to(device)
            features = model.encode(view1)
            features = model.projection_head(features)
            all_features.append(features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)


if __name__ == "__main__":

    model = Autoencoder3D(latent_dim=128)
    
    hdf5_path = r"\\path\to\dataset.h5"
    AEdataset = HDF5Dataset(hdf5_path, dataset_key='data', augmentation=Augmentation3D(prob=0, rotation_range=0), normalize=False)
    CLdataset = HDF5Dataset(hdf5_path, dataset_key='data', augmentation=Augmentation3D(), normalize=False)
    AEloader = DataLoader(AEdataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    CLloader = DataLoader(CLdataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    loss = InfoNCELoss()
    
    model = Autoencoder3D()
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    # Stage 1: Autoencoder training
    train_autoencoder(model, AEloader, num_epochs=30, lr=1e-3)
    
    # Stage 2: Contrastive learning
    train_contrastive(model, CLloader, loss, num_epochs=50, lr=1e-3)

    model.save_ckpt(os.path.join("models", "AE16.pth"))
    feat = extract_features(model, AEloader, device='cuda')
    np.save("AE16_features.npy", feat)

    print("Extracted features shape:", feat.shape)

    umap_model = UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=0,
    )
    Y = umap_model.fit_transform(feat)

    out_dir = r"\\path\to\save\output"
    
    # Save UMAP results
    np.save(os.path.join(out_dir, f"AE16_umap.npy"), Y.astype(np.float32))

