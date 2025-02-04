import torch
import torch.nn as nn
from transformers import LxmertModel, LxmertTokenizer
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from torch.utils.data import DataLoader
from testDataset import TestDataset
from tqdm import tqdm
import copy

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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

ln_params = {}

def source_hook(module: nn.LayerNorm, input, output):
    ln_params["source"] = torch.cat((module.weight.data.clone(), module.bias.data.clone()), dim=0)

def target_hook(module, input, output):
    ln_params["target"] = torch.cat((module.weight.data.clone(), module.bias.data.clone()), dim=0)

class FeatureExtractor:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True).eval().to("cuda:0")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_features(self, image):
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to("cuda:0")
            output = self.model(image_tensor)
            boxes = output[0]["boxes"]
            features = boxes[:36]  # Limit to 36 regions as LXMERT uses a max of 36 features per image
            return features

class TransferModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(TransferModel, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Initialize LXMERT model and tokenizer
        self.model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        
        # LayerNorm layers for cosine similarity loss
        self.source_ln = copy.deepcopy(self.model.lxmert.encoder.layer[-1].output.LayerNorm)
        self.target_ln = copy.deepcopy(self.model.lxmert.encoder.layer[-1].output.LayerNorm)

        self.source_ln.register_forward_hook(source_hook)
        self.target_ln.register_forward_hook(target_hook)

        for param in self.model.parameters():
            param.requires_grad = False

        self.source_ln.requires_grad_ = True
        self.target_ln.requires_grad_ = True

        self.source_model = copy.deepcopy(self.model)
        self.target_model = copy.deepcopy(self.model)
        
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, image_features, text):
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        if "source" in text:
            outputs_source = self.source_model(
                input_ids=text_inputs["input_ids"],
                visual_feats=image_features["source"].to(self.device),
                attention_mask=text_inputs["attention_mask"]
            )
            multimodal_embedding_source = outputs_source.pooler_output

        if "target" in text:
            outputs_target = self.target_model(
                input_ids=text_inputs["input_ids"],
                visual_feats=image_features["target"].to(self.device),
                attention_mask=text_inputs["attention_mask"]
            )
            multimodal_embedding_target = outputs_target.pooler_output

        if "source" in text and "target" in text:
            multimodal_embedding = torch.cat((multimodal_embedding_source, multimodal_embedding_target), dim=0)
        elif "source" in text:
            multimodal_embedding = multimodal_embedding_source
        elif "target" in text:
            multimodal_embedding = multimodal_embedding_target

        output = self.classifier(multimodal_embedding)
        return output

# Initialize the feature extractor
feature_extractor = FeatureExtractor()

# Load the transfer model
transfer_model = TransferModel(num_classes=1000)
transfer_model = transfer_model.to('cuda:0')
transfer_model.load_state_dict(torch.load('./lxmert_vqa_v2.pth'))
transfer_model.eval()   

# Prepare dataset and dataloader
test_dataset = TestDataset('data/data/test/images', 'data/data/test/test_questions.csv')
batch_size = 128
num_workers = 4
test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

# Initialize meters
test_accuracy_meter = AverageMeter()
cat_accuracy_meters = {
    "VQAabs": AverageMeter(),
    "VG": AverageMeter(),
    "GQA": AverageMeter()
}

# Test loop
for data in tqdm(test_loader):
    img = data["img"]
    ques = data["question"]
    ans = data["answer"]
    img_path = data["img_path"]

    # Prepare input data format
    img = {"target": img}
    ques = {"target": ques}

    # Extract image features
    img_features = {
        "target": torch.stack([feature_extractor.get_features(img) for img in img["target"]])
    }

    # Run model and calculate accuracy
    output = transfer_model(img_features, ques)

    # Categorize outputs for separate accuracy measurement
    cat_output = {"GQA": [], "VG": [], "VQAabs": []}
    cat_ans = {"GQA": [], "VG": [], "VQAabs": []}

    for i, image in enumerate(img_path):
        cat = image.split("_")[0]
        cat_output[cat].append(output[i])
        cat_ans[cat].append(ans[i])

    # Calculate accuracy per category
    for cat in cat_output:
        answer = torch.tensor(cat_ans[cat]).to("cuda:0")
        if answer.size(0) > 0:
            pred = torch.stack(cat_output[cat]).to("cuda:0")
            acc1 = accuracy(pred, answer, topk=(1,))
            cat_accuracy_meters[cat].update(acc1[0].item(), answer.size(0))

    # Overall test accuracy
    ans = ans.to("cuda:0")
    acc1 = accuracy(output, ans, topk=(1,))
    test_accuracy_meter.update(acc1[0].item(), ans.size(0))

print(f'Total Test Accuracy: {test_accuracy_meter.avg:.2f}')
for cat, acc in cat_accuracy_meters.items():
    print(f'{cat} Test Accuracy: {acc.avg:.2f}')
