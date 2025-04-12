import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, image_dim, text_dim, latent_dim):
        super(CVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 12, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Flatten()
        )
        flattened_dim = 128 * (image_dim // 4) * (image_dim // 4)  # 128 -> 32 (downsampled by 4)
        self.fc_mu = nn.Linear( latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim , latent_dim)

        # Decoder
        self.fc = nn.Linear(latent_dim + text_dim, flattened_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),  # 128 -> 128
            nn.Sigmoid()
        )

    def encode(self, x, c):
        img_features = self.encoder(x)
        #if c.dim() == 1:
        #    c = c.unsqueeze(0).repeat(x.size(0), 1)
        #if c.dim() == 2:
        #    c = c.mean(dim=0).unsqueeze(0)
        #combined = torch.cat([img_features, c], dim=1)
        mu = self.fc_mu(img_features)
        logvar = self.fc_logvar(img_features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        #if c.dim() == 1:
        #    c = c.unsqueeze(0).repeat(z.size(0), 1)
        #if c.dim() == 2:
        #    c = c.mean(dim=0).unsqueeze(0)
        combined = torch.cat([z, c], dim=-1)
        out = self.fc(combined).view(z.size(0), 128, 32, 32)
        out = self.decoder(out)
        return out

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
