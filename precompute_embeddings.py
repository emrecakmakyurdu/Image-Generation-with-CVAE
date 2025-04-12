import os
import torch
from tqdm import tqdm
import open_clip as clip

def precompute_embeddings_gpu(dataset, output_dir, batch_size=64):
    os.makedirs(output_dir, exist_ok=True)
    dataset._initialize_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.model.to(device)

    for img_id in tqdm(dataset.ids, desc="Precomputing embeddings"):
        captions = [ann['caption'] for ann in dataset.coco_captions.imgToAnns[img_id]]
        tokenized = dataset.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=16).to(device)

        with torch.no_grad():
            embeddings = dataset.model(**tokenized).last_hidden_state.mean(dim=1).cpu()
        torch.save(embeddings, os.path.join(output_dir, f"{img_id}.pt"))

def precompute_embeddings_gpu_clip(dataset, output_dir, batch_size=64):
    """
    Precompute text embeddings using the CLIP model and save them to disk.

    Args:
        dataset: A dataset object with captions to process.
        output_dir: Directory where embeddings will be saved.
        batch_size: Batch size for processing captions.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset._initialize_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.clip_model.to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for img_id in tqdm(dataset.ids, desc="Precomputing embeddings with CLIP"):
        # Extract captions for the current image ID
        captions = [ann['caption'] for ann in dataset.coco_captions.imgToAnns[img_id]]

        # Tokenize and process captions with CLIP
        embeddings = []
        for sentence in captions:
            with torch.no_grad():
                tokenized = clip.tokenize(sentence).to(device)
                embedding = dataset.clip_model.encode_text(tokenized).type(torch.float32)
                embeddings.append(embedding)
        
        # Take the mean of embeddings across all captions for this image
        embeddings = torch.mean(torch.stack(embeddings), dim=0).cpu()

        # Save the embeddings
        torch.save(embeddings, os.path.join(output_dir, f"{img_id}.pt"))

