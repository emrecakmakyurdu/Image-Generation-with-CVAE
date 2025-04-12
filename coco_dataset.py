import os
from torchvision import transforms
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import open_clip as clip

class CocoDataset(Dataset):
    def __init__(self, root, captions_file, instances_file, transform, embedding_dir=None):
        self.root = root
        self.coco_captions = COCO(captions_file)
        self.coco_instances = COCO(instances_file)
        self.ids = list(self.coco_captions.imgToAnns.keys())
        self.transform = transform
        self.embedding_dir = embedding_dir

        self.tokenizer = None
        self.clip_model = None
        self.grouped_data = None
        self.flattened_data = None

    def _initialize_model(self):
        if self.tokenizer is None or self.clip_model is None:
            #self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            #self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to("cuda")
            self.clip_model, _, _ = clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            self.clip_model = self.clip_model.to('cuda')
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def __len__(self):
        return len(self.flattened_data) if self.flattened_data else len(self.ids)

    def __getitem__(self, idx):
        if self.flattened_data:
            category, img_id, captions = self.flattened_data[idx]
        else:
            img_id = self.ids[idx]
            ann = self.coco_captions.imgToAnns.get(img_id, [])
            captions = [a['caption'] for a in ann]
            category = None
        if not captions:
            raise ValueError(f"No captions found for image ID {img_id}")

        img_info = self.coco_captions.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        '''
        if self.embedding_dir:
            embedding_path = os.path.join(self.embedding_dir, f"{img_id}.pt")
            if not os.path.exists(embedding_path):
                raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
            embeddings = torch.load(embedding_path)
            #print("embeddings shapein get item ", embeddings.shape)
            if len(embeddings) > 5:
                embeddings = embeddings[:5]
        else:
            self._initialize_model()
            embeddings = []
            for caption in captions:
                tokenized = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=16).to("cuda")
                with torch.no_grad():
                    text_embedding = self.model(**tokenized).last_hidden_state.mean(dim=1).squeeze(0).cpu()
                    embeddings.append(text_embedding)
            embeddings = torch.stack(embeddings)
            if len(embeddings) > 5:
                embeddings = embeddings[:5]
        '''
        embedding_path = os.path.join(self.embedding_dir, f"{img_id}.pt")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        embeddings = torch.load(embedding_path)
        #print("embeddings shapein get item ", embeddings.shape)
        if len(embeddings) > 5:
            embeddings = embeddings[:5]
        
        return image, embeddings, category

    def group_by_category(self):
        categories = self.coco_instances.loadCats(self.coco_instances.getCatIds())
        category_id_to_name = {cat['id']: cat['name'] for cat in categories}
        grouped_data = {cat['name']: [] for cat in categories}

        for img_id in self.ids:
            ann_ids = self.coco_instances.getAnnIds(imgIds=img_id)
            anns = self.coco_instances.loadAnns(ann_ids)
            category_ids = {ann['category_id'] for ann in anns}
            caption_ann_ids = self.coco_captions.getAnnIds(imgIds=img_id)
            caption_anns = self.coco_captions.loadAnns(caption_ann_ids)
            captions = [ann['caption'] for ann in caption_anns]

            for category_id in category_ids:
                category_name = category_id_to_name[category_id]
                grouped_data[category_name].append((img_id, captions))

        self.grouped_data = grouped_data
        self.flattened_data = [
            (category_name, img_id, captions)
            for category_name, samples in grouped_data.items()
            for img_id, captions in samples
        ]
        return grouped_data, self.flattened_data



'''
class CocoDataset(Dataset):
    def __init__(self, root, captions_file, instances_file, transform, embedding_dir=None):
        self.root = root
        self.coco_captions = COCO(captions_file)
        self.coco_instances = COCO(instances_file)
        self.ids = list(self.coco_captions.imgToAnns.keys())
        self.transform = transform
        self.embedding_dir = embedding_dir

        self.tokenizer = None
        self.model = None
        self.grouped_data = None
        self.flattened_data = None  # Store precomputed flattened data

    def _initialize_model(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to("cuda")

    def __len__(self):
        if self.flattened_data:  # Use precomputed flattened_data length
            return len(self.flattened_data)
        return len(self.ids)

    def __getitem__(self, idx):
        if self.flattened_data:
            category, img_id, captions = self.flattened_data[idx]
        else:
            img_id = self.ids[idx]
            ann = self.coco_captions.imgToAnns.get(img_id, [])
            captions = [a['caption'] for a in ann]
            category = None
        #print(category, img_id, captions)
        if not captions:
            print(f"Warning: No captions found for image ID {img_id}")
            return None  # Skip if no captions are available
    
        img_info = self.coco_captions.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
    
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            return None  # Skip if the image file is missing
    
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None  # Skip if the image cannot be opened
    
        if self.transform:
            image = self.transform(image)
    
        if self.embedding_dir:
            embedding_path = os.path.join(self.embedding_dir, f"{img_id}.pt")
            if not os.path.exists(embedding_path):
                print(f"Warning: Embedding file not found: {embedding_path}")
                return None  # Skip if the embedding file is missing
            embeddings = torch.load(embedding_path)
            if len(embeddings) > 5:
                embeddings = embeddings[:5]
        else:
            self._initialize_model()
            embeddings = []
            for caption in captions:
                tokenized = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=16).to("cuda")
                with torch.no_grad():
                    text_embedding = self.model(**tokenized).last_hidden_state.mean(dim=1).squeeze(0).cpu()
                    embeddings.append(text_embedding)
            embeddings = torch.stack(embeddings)

        return image, embeddings, category

    def group_by_category(self):
        categories = self.coco_instances.loadCats(self.coco_instances.getCatIds())
        category_id_to_name = {cat['id']: cat['name'] for cat in categories}
        grouped_data = {cat['name']: [] for cat in categories}
        flattened_data = []
    
        for img_id in self.ids:
            ann_ids = self.coco_instances.getAnnIds(imgIds=img_id)
            anns = self.coco_instances.loadAnns(ann_ids)
            category_ids = {ann['category_id'] for ann in anns}
            caption_ann_ids = self.coco_captions.getAnnIds(imgIds=img_id)
            caption_anns = self.coco_captions.loadAnns(caption_ann_ids)
            captions = [ann['caption'] for ann in caption_anns]
    
            for category_id in category_ids:
                category_name = category_id_to_name[category_id]
                #if category_name == "person":  # Filter for specific category
                grouped_data[category_name].append((img_id, captions))
                flattened_data.append((category_name, img_id, captions))

        self.grouped_data = grouped_data
        self.flattened_data = flattened_data  # Store precomputed flattened data
        return grouped_data, flattened_data

'''
