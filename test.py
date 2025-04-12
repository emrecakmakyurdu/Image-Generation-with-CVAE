import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader

# Function to calculate FID score
def calculate_fid(real_activations, generated_activations):
    mu_real = np.mean(real_activations, axis=0)
    mu_generated = np.mean(generated_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    sigma_generated = np.cov(generated_activations, rowvar=False)

    diff = mu_real - mu_generated

    # Compute square root of product of covariance matrices
    cov_mean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)

    # Handle numerical issues
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2 * cov_mean)
    return fid

# Function to get activations from InceptionV3
@torch.no_grad()
def get_inception_activations(images, model):
    model.eval()
    activations = []
    for img in images:
        img = TF.resize(img, (299, 299))
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = img.unsqueeze(0).to(device)
        activations.append(model(img).cpu().numpy())
    return np.concatenate(activations, axis=0)

# Load pre-trained InceptionV3 for FID calculation
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()  # Use the penultimate layer for activations

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model = inception_model.to(device)

# Evaluate model on the test set
@torch.no_grad()
def evaluate_model(test_loader, model):
    model.eval()

    real_images = []
    generated_images = []
    mse_losses = []

    for image_paths, images, _, text_embeddings in test_loader:
        images = images.to(device)
        text_embeddings = text_embeddings.to(device)

        # Generate images
        recon_images, _, _ = model(images, text_embeddings)

        # Collect real and generated images
        real_images.extend(images.cpu())
        generated_images.extend(recon_images.cpu())

        # Compute MSE loss
        mse_losses.append(mse_loss(recon_images, images).item())

    # Calculate FID score
    real_activations = get_inception_activations(real_images, inception_model)
    generated_activations = get_inception_activations(generated_images, inception_model)
    fid_score = calculate_fid(real_activations, generated_activations)

    # Average MSE loss
    avg_mse_loss = np.mean(mse_losses)

    return fid_score, avg_mse_loss

# Example Usage
# Assuming you have a test DataLoader and a trained model
# Replace `test_loader` with your DataLoader and `cvae_model` with your trained model
fid, mse = evaluate_model(test_loader, cvae_model)
print(f"FID Score: {fid:.2f}")
print(f"Average MSE Loss: {mse:.4f}")
