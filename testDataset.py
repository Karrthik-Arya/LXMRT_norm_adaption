import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from trainDataset import load_obj_tsv
from torchvision import transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class TestDataset(Dataset):
    def __init__(self, img_path, questions_path, captions_path):
        df = pd.read_csv(questions_path)
        df = (
            df
            .assign(
                prefix=lambda x: x['image'].str.split('_').str[0],
                img_id=lambda x: x['image'].str.split('_').str[1].str.split('.').str[0]
            )
            .query("prefix in ['GQA', 'VG']")
            .dropna(subset=['img_id'])
        )
        self.captions = pd.read_csv(captions_path)
        self.captions.set_index("id", inplace=True)

        self.feature_data = load_obj_tsv(os.path.expanduser(img_path))
        self.img_feature_map = {item['img_id']: item for item in self.feature_data}

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
        feature_entry = self.img_feature_map[self.df["img_id"][index]]
        features = torch.tensor(feature_entry['features'], dtype=torch.float32)
        boxes = torch.tensor(feature_entry['boxes'], dtype=torch.float32)

        img_h, img_w = feature_entry['img_h'], feature_entry['img_w']
        boxes[:, [0, 2]] /= img_w  # Normalize x coordinates
        boxes[:, [1, 3]] /= img_h  # Normalize y coordinates

        question = self.df["question"][index]
        selected_answer = self.df["answer"][index]
        caption = self.captions.loc[self.df["image"][index], "caption"]

        # Convert answer to vocabulary index
        answer = torch.tensor(self.vocab[selected_answer])

        return {"img_features": features, "boxes": boxes, "question": question, "answer": answer, "img_id": self.df["image"][index], "caption": caption}
