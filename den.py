import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms
from PIL import Image

import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import open_clip as clip
from tqdm import tqdm
from cVae import cVAE
from trainer import Trainer

class CUBDataset(Dataset):
    def __init__(self, images_file, attributes_file, image_class_labels_file, image_attributes_file, transform=None, base_dir="", precomputed_dir=""):
        """
        Args:
            images_file (str): Path to the images.txt file containing image IDs and paths.
            attributes_file (str): Path to the attributes/attributes.txt file.
            image_class_labels_file (str): Path to the image_class_labels.txt file.
            image_attributes_file (str): Path to the image_attribute_labels.txt file.
            transform (callable, optional): Transform to be applied to images.
            base_dir (str): Base directory for image paths.
        """
        self.transform = transform
        self.base_dir = base_dir
        self.precomputed_dir = precomputed_dir
        #print(self.precomputed_dir)
        # Load attributes
        self.attributes = {}
        with open(attributes_file, 'r') as f:
            for line in f:
                attr_id, attr_name = line.strip().split(' ', 1)
                self.attributes[int(attr_id)] = attr_name

        # Load image-class labels
        self.image_classes = {}
        with open(image_class_labels_file, 'r') as f:
            for line in f:
                image_id, class_id = line.strip().split()
                self.image_classes[int(image_id)] = int(class_id)

        # Load image-attributes
        self.image_attributes = {}
        with open(image_attributes_file, 'r') as f:
            for line in f:
                att = line.strip().split()
                image_id = int(att[0])
                attr_id = int(att[1])
                is_present = int(att[2])

                if is_present == 1:
                    if image_id not in self.image_attributes:
                        self.image_attributes[image_id] = []
                    self.image_attributes[image_id].append(self.attributes[attr_id])

        # Get all image paths
        self.image_paths = []
        with open(images_file, 'r') as f:
            for line in f:
                image_id, image_path = line.strip().split(' ', 1)
                full_path = os.path.join(self.base_dir, image_path)
                self.image_paths.append((int(image_id), full_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_id, image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Generate caption from attributes
        attributes = self.image_attributes.get(image_id, [])
        caption = self.attributes_to_sentence(attributes)

        # Load precomputed embedding
        embedding_path = os.path.join(self.precomputed_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.pt")
        embedding = torch.load(embedding_path)

        return image_path, image, caption, embedding

    def attributes_to_sentence(self, attributes):
        """
        Converts a list of attributes into a meaningful sentence.
        """
        grouped_attributes = {
            "bill shape": [],
            "wing color": [],
            "upperparts color": [],
            "underparts color": [],
            "breast pattern": [],
            "back color": [],
            "tail shape": [],
            "upper tail color": [],
            "head pattern": [],
            "breast color": [],
            "throat color": [],
            "eye color": [],
            "bill length": [],
            "forehead color": [],
            "under tail color": [],
            "nape color": [],
            "belly color": [],
            "size": [],
            "shape": [],
            "back pattern": [],
            "tail pattern": [],
            "belly pattern": [],
            "primary color": [],
            "leg color": [],
            "bill color": [],
            "crown color": [],
            "wing pattern": []
        }

        for attr in attributes:
            category, value = attr.split("::")
            category = category.replace("has_", "").replace("_", " ").strip()
            value = value.replace("_", " ").strip()

            if category in grouped_attributes:
                grouped_attributes[category].append(value)

        sentence_parts = []
        for category, values in grouped_attributes.items():
            if values:
                if category.endswith("color"):
                    sentence_parts.append(f"{', '.join(values)} {category}")
                elif category.endswith("pattern"):
                    sentence_parts.append(f"{category} with {', '.join(values)} pattern")
                elif category.endswith("shape"):
                    sentence_parts.append(f"a {', '.join(values)} {category}")
                else:
                    sentence_parts.append(f"{', '.join(values)} {category}")

        sentence = f"This bird has {'; '.join(sentence_parts)}."
        return sentence

def precompute_embeddings_gpu_clip(dataset, output_dir, batch_size=64):
    """
    Precompute text embeddings using the CLIP model and save them to disk.

    Args:
        dataset: A dataset object with captions to process.
        output_dir: Directory where embeddings will be saved.
        batch_size: Batch size for processing captions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to('cuda')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(data_loader, desc="Precomputing embeddings with CLIP"):
        image_paths, images, captions = batch

        # Tokenize captions
        tokenized_captions = clip.tokenize(captions).to(device)

        # Compute embeddings
        with torch.no_grad():
            text_embeddings = model.encode_text(tokenized_captions).type(torch.float32)

        # Save embeddings
        for img_path, embedding in zip(image_paths, text_embeddings):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            torch.save(embedding.cpu(), os.path.join(output_dir, f"{img_id}.pt"))


import wandb

# Initialize wandb
wandb.init(project="cvae-image-generation-cub", config={
    "learning_rate": 0.001,
    "batch_size": 1024,
    "latent_dim": 1024,
    "image_dim": 128,
    "text_dim": 768,
    "epochs": 1200,
    "beta": 0.0001
})
config = wandb.config

# Define paths
data_dir = "./cub/CUB_200_2011/images.txt"
attributes_file = "./cub/CUB_200_2011/attributes/attributes.txt"
image_class_labels_file = "./cub/CUB_200_2011/image_class_labels.txt"
image_attributes_file = "./cub/CUB_200_2011/attributes/image_attribute_labels.txt"
base_dir = "./cub/CUB_200_2011/images"
output_dir = "./precomputed_embeddings_clip_cub"



# Precompute embeddings
if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0: 
    cub_dataset = CUBDataset(
        images_file=data_dir,
        attributes_file=attributes_file,
        image_class_labels_file=image_class_labels_file,
        image_attributes_file=image_attributes_file,
        transform=transform,
        base_dir=base_dir,
        precomputed_dir=output_dir
    )
    print("Precomputing embeddings...")    
    precompute_embeddings_gpu_clip(cub_dataset, output_dir)
'''    
# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
)
'''

# Data transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cub_dataset = CUBDataset(
    images_file=data_dir,
    attributes_file=attributes_file,
    image_class_labels_file=image_class_labels_file,
    image_attributes_file=image_attributes_file,
    transform=transform,
    base_dir=base_dir,
    precomputed_dir=output_dir
)

data_loader = DataLoader(cub_dataset, batch_size=config.batch_size, shuffle=True)
cvae = cVAE(latent_dim=config.latent_dim)
#cvae = CVAE(image_dim=config.image_dim, text_dim=config.text_dim, latent_dim=config.latent_dim)
#cvae = CVAE(image_dim=128, text_dim=768, latent_dim=768)
# Train model
trainer = Trainer(model=cvae, dataloader=data_loader, lr=config.learning_rate, device="cuda",save_path="saved_models/cvae_best_model1_cub.pth")
trainer.train(epochs=config.epochs)
