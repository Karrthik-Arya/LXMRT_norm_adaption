import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import json
import torch
import numpy as np
import re
from torchvision import transforms
import torch.nn.functional as F
import csv
import base64
import sys
import time


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    counts = {}
    for ans in lst:
        counts[ans] = counts.get(ans, 0) + 1
    max_count = max(counts.values())
    candidates = [ans for ans, cnt in counts.items() if cnt == max_count]
    # Break ties by sorting and picking the first (e.g., alphabetical order)
    return sorted(candidates)[0]

def preprocessing(text):
  input_text = text
  input_text = input_text.lower()

  # Removing periods except if it occurs as decimal
  input_text = re.sub(r'(?<!\d)\.(?!\d)', '', input_text)

  # Converting number words to digits
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

  # Removing articles (a, an, the)
  if len(input_text)>3:
    input_text = re.sub(r'\b(?:a|an|the)\b', '', input_text)

  # Adding apostrophe if a contraction is missing it
  input_text = re.sub(r'\b(\w+(?<!e)(?<!a))nt\b', r"\1n't", input_text)

  # input_text = re.sub(r'\b(\w+(?<!t))ent\b', r"\1en't", input_text)

  # Replacing all punctuation (except apostrophe and colon) withinput_text a space character
  input_text = re.sub(r'[^\w\':]|(?<=\d),(?=\d)', ' ', input_text)

  # Removing extra spaces
  input_text = re.sub(r'\s+', ' ', input_text).strip()

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

    def __init__(self, root, subset, img_path, transform=None):
        self.subset = subset
        self.root = root
        self.transform = transform
        self.selection = most_common_from_dict
        self.captions = pd.read_csv(os.path.expanduser(os.path.join(root, f"captions_{subset}.csv")))
        self.captions.set_index('id', inplace=True)
        self.image_features = load_obj_tsv(os.path.expanduser(img_path))
        self.img_feature_map = {item['img_id']: item for item in self.image_features}

        # Load questions
        q_path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[subset]["questions"]))
        with open(q_path, 'r') as f:
            data = json.load(f)
        df_questions = pd.DataFrame(data["questions"])
        df_questions["comp_image_id"] = df_questions["image_id"].apply(
            lambda x: f"COCO_{self.IMAGE_PATH[subset]['img_folder']}_{x:012d}"
        )

        df_questions["caption"] = df_questions["image_id"].apply(
            lambda x: self.captions.loc[x, "caption"]
        )

        # Load answers
        a_path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[subset]["answers"]))
        with open(a_path, 'r') as f:
                    data = json.load(f)
        df_annotations = pd.DataFrame(data["annotations"])
        self.vocab={}
        i=0
        self.vocab = {}
        with open('common_vocab.txt','r') as f:
            for idx, line in enumerate(f):
                tok = line.strip()              # removes *all* whitespace/newlines
                tok = tok.lower()               # if your answers are lowercased
                if tok:                         # skip empty lines
                    self.vocab[tok] = idx
        indices=[]
        for i in range(len(df_annotations)):
                selected_answer = self.selection(df_annotations["answers"][i])
                selected_answer = preprocessing(selected_answer)
                if selected_answer not in self.vocab.keys():
                    indices.append(i)
                    # print(selected_answer)
        df_annotations.drop(indices,axis=0,inplace=True)
        df_annotations.reset_index(inplace=True,drop=True) 
        #    print(df_annotations)
        df = pd.merge(df_questions, df_annotations, left_on='question_id', right_on='question_id', how='right')
        df["image_id"] = df["image_id_x"]
        if not all(df["image_id_y"] == df["image_id_x"]):
                    print("There is something wrong with image_id")
        del df["image_id_x"]
        del df["image_id_y"]
        self.df = df
        # print(len(self.df))
        for idx, row in self.df.iterrows():
            selected_answer = self.selection(row["answers"])
            preprocessed_answer = preprocessing(selected_answer)
            if preprocessed_answer not in self.vocab.keys():
                print(selected_answer)
        self.n_samples = self.df.shape[0]

    def __getitem__(self, index):
        # print(len(self.df))
        # for idx, row in self.df.iterrows():
        #     selected_answer = self.selection(row["answers"])
        #     preprocessed_answer = preprocessing(selected_answer)
        #     if preprocessed_answer not in self.vocab.keys():
        #         print(selected_answer)
        image_id = self.df["comp_image_id"][index]
        feature_entry = self.img_feature_map[image_id]
        features = torch.tensor(feature_entry['features'], dtype=torch.float32)
        boxes = torch.tensor(feature_entry['boxes'], dtype=torch.float32)

        img_h, img_w = feature_entry['img_h'], feature_entry['img_w']
        boxes[:, [0, 2]] /= img_w  # Normalize x coordinates
        boxes[:, [1, 3]] /= img_h  # Normalize y coordinates

        question = self.df["question"][index]
        caption = self.df["caption"][index]
        selected_answer = preprocessing(self.selection(self.df["answers"][index]))
        answer = torch.tensor(self.vocab[selected_answer])
        
        return {"img_features": features,"boxes": boxes, "question": question, "answer": answer, "caption": caption}
    
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
        boxes = torch.stack([item['boxes'] for item in padded_batch])
        captions = [item['caption'] for item in padded_batch]

        return {
            'img_features': img_features,
            'boxes': boxes,
            'question': questions,
            'answer': answers,
            'caption': captions
        }