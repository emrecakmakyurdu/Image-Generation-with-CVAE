import os
from torchvision import transforms
from torch.utils.data import DataLoader
from coco_dataset import CocoDataset
from precompute_embeddings import precompute_embeddings_gpu, precompute_embeddings_gpu_clip
from cvae import CVAE
from cVae import cVAE
from trainer import Trainer
from category_sampler import CategorySampler 
from category_batch_sampler import CategoryBatchSampler
import torch

# Paths
root = "./coco/images/train2017"
annotation_file = "./coco/annotations/captions_train2017.json"
instances_file = "./coco/annotations/instances_train2017.json"
embedding_dir = "./precomputed_embeddings_clip"

# Data transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Precompute embeddings
if not os.path.exists(embedding_dir) or len(os.listdir(embedding_dir)) == 0:
    dataset_for_precompute = CocoDataset(root, annotation_file, instances_file, transform=transform)
    print("Precomputing embeddings...")
    precompute_embeddings_gpu_clip(dataset_for_precompute, embedding_dir)

# Load dataset
dataset = CocoDataset(root, annotation_file, instances_file, transform=transform, embedding_dir=embedding_dir)

# Generate grouped and flattened data
grouped_data, flattened_data = dataset.group_by_category()

# Create the custom sampler using precomputed flattened_data
sampler = CategoryBatchSampler(flattened_data, batch_size=64, shuffle_categories=True)

# DataLoader with custom sampler
dataloader = DataLoader(dataset, batch_sampler=sampler)

import wandb

# Initialize wandb
wandb.init(project="cvae-image-generation", config={
    "learning_rate": 0.001,
    "batch_size": 64,
    "latent_dim": 768,
    "image_dim": 128,
    "text_dim": 768,
    "epochs": 15
})
config = wandb.config

# Initialize model
cvae = cVAE()
#cvae = CVAE(image_dim=config.image_dim, text_dim=config.text_dim, latent_dim=config.latent_dim)
#cvae = CVAE(image_dim=128, text_dim=768, latent_dim=768)
# Train model
trainer = Trainer(model=cvae, dataloader=dataloader, lr=config.learning_rate, device="cuda",save_path="saved_models/cvae_best_model1.pth")
trainer.train(epochs=config.epochs)

# Finish wandb session
wandb.finish()

'''
# Initialize model
image_dim = 128
text_dim = 768
latent_dim = 256
cvae = CVAE(image_dim=image_dim, text_dim=text_dim, latent_dim=latent_dim)

# Train model
trainer = Trainer(model=cvae, dataloader=dataloader, lr=0.001, device="cuda")
trainer.train(epochs=10)
'''
