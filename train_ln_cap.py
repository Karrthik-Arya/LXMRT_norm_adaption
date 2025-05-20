import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LxmertModel, LxmertTokenizer
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.multiprocessing as mp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Utility Classes and Functions
# -----------------------------
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
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

# Dictionary to store LayerNorm parameters from hooks
ln_params = {}

def source_hook(module: nn.LayerNorm, input, output):
    ln_params["source"] = torch.cat((module.weight.data.clone(), module.bias.data.clone()), dim=0)

def target_hook(module: nn.LayerNorm, input, output):
    ln_params["target"] = torch.cat((module.weight.data.clone(), module.bias.data.clone()), dim=0)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        # x: (seq_len, batch, dim)
        # context: (batch, dim) or (batch, 1, dim)
        seq_len, B, D = x.shape
        if context.dim() == 2:
            context = context.unsqueeze(1)    # → (B,1,D)

        # project
        q = self.q(x.permute(1,0,2))        # (B, seq_len, D)
        k = self.k(context)                 # (B, 1, D)
        v = self.v(context)                 # (B, 1, D)

        # reshape for multi-head
        q = q.view(B, seq_len, self.num_heads, D//self.num_heads).permute(0,2,1,3)
        k = k.view(B,    1, self.num_heads, D//self.num_heads).permute(0,2,3,1)
        v = v.view(B,    1, self.num_heads, D//self.num_heads).permute(0,2,1,3)

        attn = (q @ k) * self.scale        # (B, heads, seq_len, 1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)                   # (B, heads, seq_len, head_dim)
        out = out.permute(0,2,1,3).reshape(B, seq_len, D)
        out = self.proj(out).permute(1,0,2)  # → (seq_len, B, D)
        out = self.proj_drop(out)

        # residual + norm
        return self.norm(x + out)



# -----------------------------
# Transfer Model Definition
# -----------------------------
class TransferModel(nn.Module):
    def __init__(self, num_classes=1000, use_captions=True):
        super(TransferModel, self).__init__()
        self.device = device
        self.use_captions = use_captions

        # Load LXMERT and its tokenizer
        self.model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased", output_hidden_states=True)
        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        # Create source and target copies
        self.source_model = copy.deepcopy(self.model)
        self.target_model = copy.deepcopy(self.model)

        # --- LayerNorm Adaptation ---
        # Retrieve the final LayerNorm from the last cross-modal layer and copy it for source and target.
        last_source = self.source_model.encoder.x_layers[-1]
        last_ln_source = last_source.visual_attention.output.LayerNorm
        self.source_ln = copy.deepcopy(last_ln_source)
        self.source_ln.register_forward_hook(source_hook)

        last_target = self.target_model.encoder.x_layers[-1]
        last_ln_target = last_target.visual_attention.output.LayerNorm
        self.target_ln = copy.deepcopy(last_ln_target)
        self.target_ln.register_forward_hook(target_hook)

        # Freeze all parameters initially
        for param in self.source_model.parameters():
            param.requires_grad = False
        for param in self.target_model.parameters():
            param.requires_grad = False

        # Unfreeze specific layers for caption injection 
        self.layers_to_modify = [1, 3, 4]  #injection points
        for i in self.layers_to_modify:
            for param in self.source_model.encoder.x_layers[i].parameters():
                param.requires_grad = True
            for param in self.target_model.encoder.x_layers[i].parameters():
                param.requires_grad = True

        # --- Caption Injection Modules ---
        self.cross_attns = nn.ModuleList([
            CrossAttention(dim=768, num_heads=8) 
            for _ in self.layers_to_modify
        ])

        # Classifier: Concatenate image and text features (each 768-d) → 1536-d input.
        self.classifier = nn.Linear(768 * 2, num_classes)

    def embed_caption(self, caption):
        """
        Embed a caption using LXMERT's text encoder.
        """
        toks = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(self.device)
        bsz = toks.input_ids.size(0)
        dummy_vf = torch.zeros(bsz, 36, 2048, device=self.device)
        dummy_vp = torch.zeros(bsz, 36, 4, device=self.device)
        with torch.no_grad():
            out = self.source_model(
                input_ids=toks.input_ids,
                attention_mask=toks.attention_mask,
                visual_feats=dummy_vf,
                visual_pos=dummy_vp
            )
        cap_emb = out.language_hidden_states[-1].mean(1)
        return F.normalize(cap_emb, dim=-1)

    def embed_text_with_captions(self, text_input, caption_emb, branch='source'):
        """
        Process text tokens through LXMERT's language encoder with caption injection.
        """
        # Tokenize text input
        tokens = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Choose the appropriate model branch
        if branch == 'source':
            model_branch = self.source_model
            final_ln = self.source_ln
        else:
            model_branch = self.target_model
            final_ln = self.target_ln
        
        # Instead of directly accessing the embeddings, use the model's input processing
        # Get the language model output directly from LXMERT
        extended_attention_mask = model_branch.get_extended_attention_mask(
            tokens["attention_mask"], 
            tokens["attention_mask"].size(), 
            self.device
        )
        
        # Get language features using the language encoder
        lang_feats = model_branch.embeddings(
            input_ids=tokens["input_ids"],
            token_type_ids=torch.zeros_like(tokens["input_ids"])  # Default token type IDs
        )
        
        # We'll iterate through language layers and apply our cross-attention where needed
        caption_idx = 0
        
        # Process through language layers
        for i, layer_module in enumerate(model_branch.encoder.layer):
            lang_feats = layer_module(lang_feats, extended_attention_mask)[0]
            
            # Apply caption cross-attention at specified layers
            if i in self.layers_to_modify:
                # Convert to format expected by cross-attention
                lang_feats_transposed = lang_feats.permute(1, 0, 2)  # [batch, seq, dim] -> [seq, batch, dim]
                
                # Apply cross-attention
                ca = self.cross_attns[caption_idx]
                lang_feats_with_caption = ca(lang_feats_transposed, caption_emb)
                
                # Convert back to original format
                lang_feats = lang_feats_with_caption.permute(1, 0, 2)  # [seq, batch, dim] -> [batch, seq, dim]
                caption_idx += 1
        
        # Apply final layer norm
        lang_feats = final_ln(lang_feats)
        
        # Get the [CLS] token representation (first token)
        text_features = lang_feats[:, 0]
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def forward_branch(self, text, img_feats, boxes, model, ln, caption_emb=None):
        # Prepare inputs
        toks = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        bsz = toks.input_ids.size(0)

        if self.use_captions and caption_emb is not None:
            # Inject captions in the language encoder
            ext_mask = model.get_extended_attention_mask(toks.attention_mask, toks.attention_mask.size(), self.device)
            lang_feats = model.embeddings(toks.input_ids, token_type_ids=torch.zeros_like(toks.input_ids))
            cap_idx = 0
            for i, layer in enumerate(model.encoder.layer):
                lang_feats = layer(lang_feats, ext_mask)[0]
                if i in self.layers_to_modify:
                    lang_feats = self.cross_attns[cap_idx](lang_feats.permute(1,0,2), caption_emb).permute(1,0,2)
                    cap_idx += 1
            # Run cross-modal with modified embeddings via inputs_embeds
            out = model(
                inputs_embeds=lang_feats,          # use modified token embeddings
                attention_mask=toks.attention_mask,
                visual_feats=img_feats,            # proper 2048-d visual features
                visual_pos=boxes
            )
        else:
            # Standard cross-modal forward
            out = model(
                input_ids=toks.input_ids,
                attention_mask=toks.attention_mask,
                visual_feats=img_feats,
                visual_pos=boxes
            )

        # Return pooled cross-modal embedding
        return F.normalize(out.pooled_output, dim=-1)


    def forward(self, image_features, boxes, text, caption_embeddings):
        """
        Process source and target branches.
          - text: dict with keys "source" and/or "target"
          - image_features: dict with keys "source" and/or "target"
          - caption_embeddings: dict with keys "source" and/or "target"
        """
        src_rep = self.forward_branch(text['source'], image_features['source'].to(self.device), boxes["source"].to(self.device),
                                      self.source_model, self.source_ln, caption_embeddings.get('source'))
        tgt_rep = self.forward_branch(text['target'], image_features['target'].to(self.device), boxes["target"].to(self.device),
                                      self.target_model, self.target_ln, caption_embeddings.get('target'))
        combined = torch.cat((src_rep, tgt_rep), dim=0)
        return self.classifier(combined)


# -----------------------------
# Mixed Data Loader Function
# -----------------------------
def mixed_data_loader(loader1, loader2):
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

        max_i_dim = max(batch1["img_features"].size(1),
                        batch2["img_features"].size(1))
        i1 = F.pad(batch1["img_features"],
                   (0, max_i_dim - batch1["img_features"].size(1)))
        i2 = F.pad(batch2["img_features"],
                   (0, max_i_dim - batch2["img_features"].size(1)))
        mixed_batch = {
            "img_features": {
                "source": i1,
                "target": i2
            },
            "boxes": {
                "source": batch1["boxes"],
                "target": batch2["boxes"]
            },
            "question": {
                "source": batch1["question"],
                "target": batch2["question"]
            },
            "answer": torch.cat((batch1["answer"], batch2["answer"]), dim=0),
            "caption": {
                "source": batch1["caption"],
                "target": batch2["caption"]
            }
        }
        yield mixed_batch

# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    batch_size = 128
    num_workers = 4
    lr = 1e-3
    epochs = 50

    # Assume TrainDataset and TestDataset are defined as before
    from trainDataset import TrainDataset
    from testDataset import TestDataset

    train_dataset = TrainDataset('data/vqa_v2', 'train', 'data/mscoco_imgfeat/train2014_obj36.tsv')
    val_dataset = TrainDataset('data/vqa_v2', 'val', 'data/mscoco_imgfeat/val2014_obj36.tsv')
    train_targ_dataset = TestDataset('data/vg_gqa_imgfeat/vg_gqa_obj36.tsv', "data/test/train_questions.csv",  "data/test/captions.csv")

    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=int(batch_size * 0.75), shuffle=False)
    train_targ_loader = DataLoader(train_targ_dataset, num_workers=num_workers, batch_size=int(batch_size * 0.25), shuffle=False)
    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    transfer_model = TransferModel(num_classes=1000, use_captions=True).to(device)
    optimizer = AdamW(transfer_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    val_accuracy_meter = AverageMeter()
    best_val_acc = 0

    for epoch in range(epochs):
        transfer_model.train()
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        mixed_loader = mixed_data_loader(train_loader, train_targ_loader)

        for data in tqdm(mixed_loader, desc=f"Epoch {epoch+1} Training"):
            img_features = data["img_features"]
            boxes = data["boxes"]
            ques = data["question"]
            ans = data["answer"].to(device)
            captions = data["caption"]

            # Embed captions for source and target using embed_caption 
            caption_emb_source = torch.stack([transfer_model.embed_caption(cap) for cap in captions["source"]]).to(device)
            caption_emb_target = torch.stack([transfer_model.embed_caption(cap) for cap in captions["target"]]).to(device)
            caption_embeddings = {"source": caption_emb_source, "target": caption_emb_target}

            # Forward pass
            output = transfer_model(img_features, boxes, ques, caption_embeddings)
            src_count = len(ques["source"])
            tgt_count = len(ques["target"])

            # Compute losses :
            loss_source = F.cross_entropy(output[:src_count], ans[:src_count])
            probs_target = F.softmax(output[src_count:src_count+tgt_count], dim=-1)
            loss_target = -torch.sum(probs_target * torch.log(probs_target + 1e-9