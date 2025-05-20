import os
import json
import pickle
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from trainDataset import TrainDataset  # Import from your existing code

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class FeatureExtractor:
    def __init__(self):
        # Create a ResNet101 FPN backbone for 2048-dimensional features
        backbone = resnet_fpn_backbone('resnet101', pretrained=True)
        
        # Create FasterRCNN with custom backbone
        self.model = FasterRCNN(backbone, num_classes=91).to(device)
        
        # Standard image transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Set to evaluation mode
        self.model.eval()
    
    def get_features(self, pil_image):
        with torch.no_grad():
            # Apply transformations to the image (no batch dimension yet)
            img_tensor = self.transform(pil_image).to(device)
            
            # Forward pass through the model - FasterRCNN expects a list of images
            outputs = self.model([img_tensor])
            
            # Get the predicted boxes for the first (and only) image
            boxes = outputs[0]["boxes"][:36]  # Take top 36 boxes
            
            # To get feature maps, we need to add batch dimension for backbone
            img_batch = img_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
            
            # Get features from backbone
            backbone_features = self.model.backbone(img_batch)
            
            # Get pooled features using ROI pooling
            image_sizes = [(pil_image.height, pil_image.width)]
            pooled_features = self.model.roi_heads.box_roi_pool(
                backbone_features, [boxes], image_sizes
            )
            
            # Extract box features - will be 2048-dim with ResNet101
            box_features = self.model.roi_heads.box_head(pooled_features)
            
            # Print shape of features for validation
            if boxes.shape[0] > 0:  # If there are detected boxes
                print(f"Box features shape: {box_features.shape}")
                print(f"Feature dimension: {box_features.shape[-1]}")
                assert box_features.shape[-1] == 2048, "Features are not 2048-dimensional!"
            
            # Return both visual features and positions
            return {
                "visual_feats": box_features,   # Shape: [num_boxes, 2048]
                "visual_pos": boxes             # Shape: [num_boxes, 4]
            }


def precompute_frcnn_features(dataset_root, subset):
    # Load questions JSON to get image IDs
    q_path = os.path.join(dataset_root, TrainDataset.IMAGE_PATH[subset]["questions"])
    with open(q_path, "r") as f:
        data = json.load(f)
    df_questions = pd.DataFrame(data["questions"])
    image_ids = df_questions["image_id"].unique().tolist()

    # Initialize Faster R-CNN feature extractor
    extractor = FeatureExtractor()

    # Check output dimension on a sample image (if available)
    if len(image_ids) > 0:
        try:
            # Build image path for first image
            img_folder = TrainDataset.IMAGE_PATH[subset]["img_folder"]
            img_filename = f"COCO_{img_folder}_{image_ids[0]:012d}.jpg"
            img_path = os.path.join(dataset_root, img_folder, img_filename)
            
            # Extract features for dimension validation
            img = Image.open(img_path).convert("RGB")
            features = extractor.get_features(img)
            print(f"Verified: Using {features['visual_feats'].shape[-1]}-dimensional features")
        except Exception as e:
            print(f"Could not verify feature dimensions: {e}")

    result = []

    # Process each image
    for i, image_id in enumerate(image_ids):
        try:
            # Build image path
            img_folder = TrainDataset.IMAGE_PATH[subset]["img_folder"]
            img_filename = f"COCO_{img_folder}_{image_id:012d}.jpg"
            img_path = os.path.join(dataset_root, img_folder, img_filename)

            # Extract features
            img = Image.open(img_path).convert("RGB")
            features = extractor.get_features(img)
            
            # Print progress
            if i % 100 == 0:
                print(f"Processed {i}/{len(image_ids)} images")
                if i > 0 and 'visual_feats' in features:
                    print(f"Feature shape: {features['visual_feats'].shape}")

            result.append((image_id, features["visual_feats"], features["visual_pos"]))
            
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue

    # Save results to pickle file
    output_path = os.path.join(dataset_root, f'{subset}_img_feats.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Features saved to {output_path}")
    print(f"Feature dimension: 2048")


if __name__ == "__main__":
    root_dir = 'data/data/vqa_v2/'
    precompute_frcnn_features(root_dir, 'train')