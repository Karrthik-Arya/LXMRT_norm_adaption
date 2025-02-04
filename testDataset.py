import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define Feature Extractor using Faster R-CNN
class FeatureExtractor:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # Resize the image for consistency
        ])
    
    def get_features(self, image):
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to(device)
            output = self.model(image_tensor)
            boxes = output[0]["boxes"]
            return boxes[:36]  # Limit to the top 36 regions for consistency with LXMERT

# Initialize feature extractor
feature_extractor = FeatureExtractor()

class TestDataset(Dataset):
    def __init__(self, img_path, questions_path):
        df = pd.read_csv(questions_path)
        self.img_path = img_path
        self.vocab = {}
        
        # Load vocabulary from file
        with open('common_vocab.txt', 'r') as file:
            for i, line in enumerate(file):
                self.vocab[line.strip()] = i

        # Filter rows based on vocabulary presence
        self.df = df[df["answer"].isin(self.vocab)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # Get image path, question, and answer
        image_path = os.path.expanduser(os.path.join(self.img_path, self.df["image"][index]))
        question = self.df["question"][index]
        selected_answer = self.df["answer"][index]

        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        image_features = feature_extractor.get_features(img)

        # Convert answer to vocabulary index
        answer = torch.tensor(self.vocab[selected_answer])

        return {"img_features": image_features, "question": question, "answer": answer, "img_path": self.df["image"][index]}
