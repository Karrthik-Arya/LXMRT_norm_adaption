import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import json
import torch
import numpy as np
import re
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch.nn.functional as F

# Define the device and Faster R-CNN for object feature extraction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom Feature Extractor using Faster R-CNN
class FeatureExtractor:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # Resize the image before passing to Faster R-CNN
        ])

    def get_features(self, image):
        self.model.eval()
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to(device)
            output = self.model(image_tensor)
            # Extracting bounding box features
            features = output[0]["boxes"]
            return features[:36]  # Limit to the top 36 regions, similar to LXMERT

# Initialize feature extractor
feature_extractor = FeatureExtractor()

def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)

def preprocessing(text):
    input_text = text.lower()
    input_text = re.sub(r'(?<!\d)\.(?!\d)', '', input_text)
    input_text = input_text.replace('clothes', 'cloth')

    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
        "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
        "eighty": "80", "ninety": "90"
    }
    input_text = re.sub(r'\b(?:' + '|'.join(number_words.keys()) + r')\b', lambda x: number_words[x.group()], input_text)
    input_text = re.sub(r'\b(?:a|an|the)\b', '', input_text)
    input_text = re.sub(r'\b(\w+(?<!e)(?<!a))nt\b', r"\1n't", input_text)
    input_text = re.sub(r'[^\w\':]|(?<=\d),(?=\d)', ' ', input_text)
    input_text = re.sub(r'\s+', ' ', input_text).strip()
    words = input_text.split()
    # filtered_words = [word for word in words if word in self.vocab]
    # return filtered_words[0] if filtered_words else ""
    return input_text

class TrainDataset(Dataset):
    IMAGE_PATH = {
        "train": { 
             "questions": "v2_OpenEnded_mscoco_train2014_questions.json",
             "answers":  "v2_mscoco_train2014_annotations.json",
             "img_folder": "train2014"
             },
        "val": {
            "questions": "v2_OpenEnded_mscoco_val2014_questions.json", 
            "answers": "v2_mscoco_val2014_annotations.json",
            "img_folder": "val2014"
        }
    }

    def __init__(self, root, subset, transform=None):
        self.subset = subset
        self.root = root
        self.transform = transform
        self.selection = most_common_from_dict

        # Load questions
        q_path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[subset]["questions"]))
        with open(q_path, 'r') as f:
            data = json.load(f)
        df_questions = pd.DataFrame(data["questions"])
        df_questions["image_path"] = df_questions["image_id"].apply(
            lambda x: f"{self.IMAGE_PATH[subset]['img_folder']}/COCO_{self.IMAGE_PATH[subset]['img_folder']}_{x:012d}.jpg"
        )

        # Load answers
        a_path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[subset]["answers"]))
        with open(a_path, 'r') as f:
            data = json.load(f)
        df_answers = pd.DataFrame(data["annotations"])

        # Build vocabulary from common_vocab.txt
        self.vocab = {}
        with open('common_vocab.txt', 'r') as file:
            for idx, line in enumerate(file):
                self.vocab[line.strip()] = idx

        # Filter answers not in the vocabulary
        indices_to_keep = []
        for idx, row in df_answers.iterrows():
            selected_answer = self.selection(row["answers"])
            preprocessed_answer = preprocessing(selected_answer)
            if preprocessed_answer in self.vocab:
                indices_to_keep.append(idx)

        df_answers = df_answers.loc[indices_to_keep].reset_index(drop=True)

        # Merge questions and answers
        self.df = pd.merge(df_questions, df_answers, left_on='question_id', right_on='question_id', how='inner')
        self.n_samples = self.df.shape[0]

    def __getitem__(self, index):
        image_path = self.df["image_path"][index]
        question = self.df["question"][index]
        selected_answer = preprocessing(self.selection(self.df["answers"][index]))
        answer = torch.tensor(self.vocab[selected_answer])

        # Load image and preprocess
        image_path = os.path.expanduser(os.path.join(self.root, image_path))
        img = Image.open(image_path).convert('RGB')

        # Apply Faster R-CNN to extract features
        features = feature_extractor.get_features(img)
        
        return {"img_features": features, "question": question, "answer": answer}
    
    def __len__(self):
        return len(self.df["answers"])
    
    @staticmethod
    def pad_batch(batch):
        """
        Pads tensors in the batch to the maximum dimensions of the batch.

        Args:
            batch (list of dict): A list of dictionaries, where each dictionary contains:
                - 'img_features' (torch.Tensor): Tensor of varying shapes to be padded.

        Returns:
            list of dict: A new batch where all 'img_features' tensors are padded to the maximum dimensions.
        """
        # Extract the 'img_features' tensors from the batch
        img_features = [item['img_features'] for item in batch]

        # Determine the maximum dimensions in the batch
        max_len = max(f.size(0) for f in img_features)  # Maximum length (dimension 0)
        max_dim = max(f.size(1) for f in img_features)  # Maximum feature dimension (dimension 1)

        # Pad each 'img_features' tensor to the maximum dimensions
        padded_img_features = [
            F.pad(f, (0, max_dim - f.size(1), 0, max_len - f.size(0))) for f in img_features
        ]

        # Replace the 'img_features' in each item with the padded tensor
        for i, item in enumerate(batch):
            item['img_features'] = padded_img_features[i]

        return batch
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.
        Pads img_features tensors within the batch to ensure consistent shapes.

        Args:
            batch (list of dict): A batch of data items.

        Returns:
            dict: A dictionary containing the padded tensors and other data.
        """
        # Pad the batch using the static method
        padded_batch = TrainDataset.pad_batch(batch)

        # Collect other keys (e.g., question, answer)
        questions = [item['question'] for item in padded_batch]
        answers = torch.stack([item['answer'] for item in padded_batch])

        # Collect img_features
        img_features = torch.stack([item['img_features'] for item in padded_batch])

        return {
            'img_features': img_features,
            'question': questions,
            'answer': answers,
        }