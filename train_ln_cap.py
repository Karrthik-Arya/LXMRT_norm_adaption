from transformers import LxmertModel, LxmertTokenizer, OFATokenizer, OFAModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from trainDataset import TrainDataset
from testDataset import TestDataset
from tqdm import tqdm
import numpy as np
import copy
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

ln_params = {}

def source_hook(module: nn.LayerNorm, input, output):
    ln_params["source"] = torch.cat((module.weight.data.clone(), module.bias.data.clone()), dim=0)

def target_hook(module, input, output):
    ln_params["target"] = torch.cat((module.weight.data.clone(), module.bias.data.clone()), dim=0)

class FeatureExtractor:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(device)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_features(self, image):
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to(device)
            output = self.model(image_tensor)
            boxes = output[0]["boxes"]
            features = boxes[:36]
            return features

# Initialize feature extractor
feature_extractor = FeatureExtractor()

class TransferModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(TransferModel, self).__init__()
        self.device = device

        self.model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        last_cross_modal_layer = self.model.encoder.x_layers[-1]
        last_layernorm = last_cross_modal_layer.visual_attention.output.LayerNorm
        self.source_ln = copy.deepcopy(last_layernorm)
        self.target_ln = copy.deepcopy(last_layernorm)

        self.source_ln.register_forward_hook(source_hook)
        self.target_ln.register_forward_hook(target_hook)

        for param in self.model.parameters():
            param.requires_grad = False

        self.source_ln.requires_grad_ = True
        self.target_ln.requires_grad_ = True

        self.source_model = copy.deepcopy(self.model)
        self.target_model = copy.deepcopy(self.model)

        self.classifier = nn.Linear(768, num_classes)

    def forward(self, image_features, text, caption_embedding=None):
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        multimodal_embedding_source = None
        multimodal_embedding_target = None

        if "source" in text:
            outputs_source = self.source_model(
                input_ids=text_inputs["input_ids"],
                visual_feats=image_features["source"].to(self.device),
                attention_mask=text_inputs["attention_mask"]
            )
            if caption_embedding is not None:
                outputs_source = self.inject_prompts(outputs_source, caption_embedding)
            multimodal_embedding_source = outputs_source.pooler_output

        if "target" in text:
            outputs_target = self.target_model(
                input_ids=text_inputs["input_ids"],
                visual_feats=image_features["target"].to(self.device),
                attention_mask=text_inputs["attention_mask"]
            )
            if caption_embedding is not None:
                outputs_target = self.inject_prompts(outputs_target, caption_embedding)
            multimodal_embedding_target = outputs_target.pooler_output

        if multimodal_embedding_source is not None and multimodal_embedding_target is not None:
            multimodal_embedding = torch.cat((multimodal_embedding_source, multimodal_embedding_target), dim=0)
        elif multimodal_embedding_source is not None:
            multimodal_embedding = multimodal_embedding_source
        elif multimodal_embedding_target is not None:
            multimodal_embedding = multimodal_embedding_target

        output = self.classifier(multimodal_embedding)
        return output

    
    def inject_prompts(self, outputs, prompt_embeddings):
        """Inject caption embeddings into the model's hidden states (only in language encoder)"""
        lang_layers = self.source_model.config.num_hidden_layers  # Number of language encoder layers
        inject_layers = np.random.choice(range(1, lang_layers + 1), size=2, replace=False)  # Skip input embeddings (index 0)
        for layer in inject_layers:
            if layer < len(outputs.hidden_states):
                outputs.hidden_states[layer] = outputs.hidden_states[layer] + prompt_embeddings[:, :outputs.hidden_states[layer].size(1), :]
        return outputs

def mixed_data_loader(loader1, loader2):
    """Mixes data from two loaders into a single batch"""
    loader1_iter = iter(loader1)
    loader2_iter = iter(loader2)

    while True:
        try:
            batch1 = next(loader1_iter)
        except StopIteration:
            loader1_iter = iter(loader1)
            batch1 = next(loader1_iter)

        try:
            batch2 = next(loader2_iter)
        except StopIteration:
            loader2_iter = iter(loader2)
            batch2 = next(loader2_iter)

        mixed_batch = {
            "img_features": {
                "source": batch1["img_features"],
                "target": batch2["img_features"]
            },
            "question": {
                "source": batch1["question"],
                "target": batch2["question"]
            },
            "answer": torch.cat((batch1["answer"], batch2["answer"]), dim=0),
            "image_path": batch1["image_path"] + batch2["image_path"]
        }

        yield mixed_batch

def generate_caption(image_path):
    # OFA configuration
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 480
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    img = Image.open(image_path).convert("RGB")
    patch_img = patch_resize_transform(img).unsqueeze(0).to(device)
    question = " what does the image describe?"
    tokens = ofa_tokenizer([question], return_tensors="pt").to(device)
    gen = ofa_model.generate(
        tokens.input_ids, 
        patch_images=patch_img, 
        num_beams=5, 
        no_repeat_ngram_size=3, 
        early_stopping=True
    )
    return ofa_tokenizer.decode(gen[0], skip_special_tokens=True)

def main():
    batch_size = 128
    num_workers = 4
    lr = 1e-3
    epochs = 50

    train_dataset = TrainDataset('data/data/vqa_v2', 'train')
    val_dataset = TrainDataset('data/data/vqa_v2', 'val', 'VQAv2')
    train_targ_dataset = TestDataset('data/data/test/images', "data/data/test/train_questions.csv")

    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=int(batch_size * 0.75), shuffle=False)
    train_targ_loader = DataLoader(train_targ_dataset, num_workers=num_workers, batch_size=int(batch_size * 0.25), shuffle=False)
    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    transfer_model = TransferModel(num_classes=1000)
    transfer_model = transfer_model.to(device)
    optimizer = AdamW(transfer_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    val_accuracy_meter = AverageMeter()
    best_val_acc = 0

    # Initialize OFA tokenizer and model
    global ofa_tokenizer, ofa_model
    ckpt_dir = "../OFA-large-caption"
    ofa_tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
    ofa_model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
    ofa_model.to(device)

    for epoch in range(epochs):
        transfer_model.train()
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        mixed_loader = mixed_data_loader(train_loader, train_targ_loader)

        for data in tqdm(mixed_loader):
            img_features = data["img_features"]
            ques = data["question"]
            ans = data["answer"].to(device)
            image_paths = data["image_path"]

            # Generate captions for each image
            captions = [generate_caption(path) for path in image_paths]
            tokenized_captions = ofa_tokenizer(captions, padding=True, return_tensors="pt").to(device)
            caption_embedding = ofa_model.generate(
                tokenized_captions.input_ids,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            output = transfer_model(img_features, ques, caption_embedding)
            
            xloss = F.cross_entropy(output[:len(ques['source'])], ans[:len(ques['source'])])
            probs_target = F.softmax(output[len(ques['source']):], dim=-1)
            target_losses = -torch.sum(probs_target * torch.log(probs_target + 1e-9), dim=-1).mean()
            cosine_sim = F.cosine_similarity(ln_params['source'], ln_params['target'], dim=0)
            loss_add = -cosine_sim.mean()

            total_loss = loss_add + xloss + target_losses

            train_loss_meter.update(total_loss.item(), ans.size(0))
            acc1 = accuracy(output, ans, topk=(1,))
            train_accuracy_meter.update(acc1[0].item(), ans.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        print(f'Epoch: {epoch + 1}, Training Loss: {train_loss_meter.avg:.4f}, Training Accuracy: {train_accuracy_meter.avg:.2f}')

        transfer_model.eval()
        val_loss_meter.reset()
        val_accuracy_meter.reset()

        for data in tqdm(val_loader):
            img_features = data["img_features"]
            ques = data["question"]
            ans = data["answer"].to(device)
            image_paths = data["image_path"]

            # Generate captions for each image
            captions = [generate_caption(path) for path in image_paths]
            tokenized_captions = ofa_tokenizer(captions, padding=True, return_tensors="pt").to(device)
            caption_embedding = ofa_model.generate(
                tokenized_captions.input_ids,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            output = transfer_model(img_features.to(device), ques, caption_embedding)
            loss = F.cross_entropy(output, ans)

            val_loss_meter.update(loss.item(), ans.size(0))
            acc1 = accuracy(output, ans, topk=(1,))
            val_accuracy_meter.update(acc1[0].item(), ans.size(0))

        print(f'Epoch: {epoch + 1}, Validation Loss: {val_loss_meter.avg:.4f}, Validation Accuracy: {val_accuracy_meter.avg:.2f}')

        if val_accuracy_meter.avg > best_val_acc:
            torch.save(transfer_model.state_dict(), 'lxmert_vqa_v2.pth')
            best_val_acc = val_accuracy_meter.avg
            print("Model saved with better validation accuracy!")

if __name__ == "__main__":
    main()