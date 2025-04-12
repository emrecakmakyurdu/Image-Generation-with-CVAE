import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import os 
import random 

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
import random

class Trainer:
    def __init__(self, model, dataloader, lr=0.001, device="cuda", save_path="model_checkpoint.pth"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)  # Adjust T_max and eta_min as needed
        self.device = device
        self.save_path = save_path
        self.beta = 0.0001

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_loss * beta

    def train(self, epochs):
        best_loss = float("inf")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for step, (image_paths, images, _, text_embeddings) in enumerate(tqdm(self.dataloader)):
                images = images.to(self.device)
                text_embeddings = text_embeddings.to(self.device)
                shuffle_indices = torch.randperm(images.size(0))
                images_target = images[shuffle_indices]
                text_embeddings_target = text_embeddings[shuffle_indices]

                self.optimizer.zero_grad()

                text_source = text_embeddings_target
                recon_images, mu, logvar = self.model(images, text_source)
                recon_loss, kl_loss = self.loss_function(recon_images, images_target, mu, logvar, self.beta)
                loss = recon_loss + kl_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if total_loss < best_loss:
                    best_loss = total_loss
                    self.save_model(self.save_path)
                    print(f"Model saved with loss {best_loss:.4f}")

                if step % 20 == 0:
                    wandb.log({
                        "train/reconstruction_loss": recon_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/total_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                    })

                if step % 100 == 0:
                    wandb.log({
                        "generated_image": [wandb.Image(recon_images[0].cpu(), caption="Generated Image")],
                        "image_to_generate": [wandb.Image(images_target[0].cpu(), caption="Image to Generate")],
                        "image_from": [wandb.Image(images[0].cpu(), caption="Image from Generation")]
                    })

            # Step the scheduler after each epoch
            self.scheduler.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(self.dataloader):.4f}")
            wandb.log({"epoch_loss": total_loss / len(self.dataloader), "learning_rate": self.scheduler.get_last_lr()[0]})

            # Final save
            self.save_model("saved_models/final_model.pth")

    def save_model(self, path):
        """Save the model to the specified path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


'''
class Trainer:
    def __init__(self, model, dataloader, lr=0.001, device="cuda", save_path="model_checkpoint.pth"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.device = device
        self.save_path = save_path
        self.beta = 0.0005

    @staticmethod
    def loss_function(recon_x, x, mu, logvar,beta):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_loss * beta

    def train(self, epochs):
        #wandb.watch(self.model, log="all", log_freq=10)  # Watch model gradients and weights
        best_loss = float("inf")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for step, (image_paths, images,_, text_embeddings) in enumerate(tqdm(self.dataloader)):
            #for step, (images, text_embeddings, category) in enumerate(tqdm(self.dataloader)):
                #print(category)
                images = images.to(self.device)
                text_embeddings = text_embeddings.to(self.device)
                shuffle_indices = torch.randperm(images.size(0))
                images_target = images[shuffle_indices]
                text_embeddings_target = text_embeddings[shuffle_indices]
  
                self.optimizer.zero_grad()

         
                #text_source = text_embeddings_target[:,random.randint(0, text_embeddings_target.shape[1]-1),:]
                text_source = text_embeddings_target
                recon_images, mu, logvar = self.model(images, text_source)
                recon_loss, kl_loss = self.loss_function(recon_images, images_target, mu, logvar, self.beta)
                loss = recon_loss + kl_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                    
                #avg_loss = total_loss / len(self.dataloader)
    
                # Save model if it improves
                if total_loss < best_loss:
                    best_loss = total_loss
                    self.save_model(self.save_path)
                    print(f"Model saved with loss {best_loss:.4f}")
                    
                # Log metrics to wandb
                if step % 20 == 0:
                    wandb.log({
                        "train/reconstruction_loss": recon_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/total_loss": loss.item(),
                    })

                # Log generated image to wandb
                if step % 100 == 0:
                    wandb.log({
                        "generated_image": [wandb.Image(recon_images[0].cpu(), caption="Generated Image")],
                        "image_to_generate": [wandb.Image(images_target[0].cpu(), caption="Image to Generate")],
                        "image_from": [wandb.Image(images[0].cpu(), caption="Image from Generation")]
                    })

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(self.dataloader):.4f}")
            wandb.log({"epoch_loss": total_loss / len(self.dataloader)})
            
            # Final save
            self.save_model("saved_models/final_model.pth")

    def save_model(self, path):
        """Save the model to the specified path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
'''
'''
class Trainer:
    def __init__(self, model, dataloader, lr=0.001, device="cuda"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.device = device

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_loss

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for step, (images, text_embeddings, _) in enumerate(tqdm(self.dataloader)):
                images = images.to(self.device)
                text_embeddings = text_embeddings.to(self.device)

                self.optimizer.zero_grad()
                recon_images, mu, logvar = self.model(images, text_embeddings)
                recon_loss, kl_loss = self.loss_function(recon_images, images, mu, logvar)
                loss = recon_loss + kl_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(self.dataloader):.4f}")
'''