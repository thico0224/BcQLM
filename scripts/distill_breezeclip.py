import os
import math
import time
import random
import argparse
from typing import Dict, Any, Tuple

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

import timm
from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor

from bcqlm import BreezeCLIP, PairedFolderDataset


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import numpy as np
    np.random.seed(seed)

def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [B, D], b: [B, D]
    return a @ b.t()

def info_nce_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Symmetric InfoNCE:
      - logits_per_image = sim(img, txt) / T
      - logits_per_text  = sim(txt, img) / T
    labels = arange(B)
    """
    logits_img = cosine_sim(img_emb, txt_emb) / temperature
    logits_txt = logits_img.t()

    targets = torch.arange(img_emb.size(0), device=img_emb.device)
    loss_i = F.cross_entropy(logits_img, targets)
    loss_t = F.cross_entropy(logits_txt, targets)
    return (loss_i + loss_t) * 0.5

def save_checkpoint(save_dir: str, epoch: int, model: BreezeCLIP, optimizer: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler, best_val: float = None):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"breezeclip_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val": best_val
    }, ckpt_path)
    return ckpt_path

def load_checkpoint(ckpt_path: str, model: BreezeCLIP, optimizer=None, scaler=None) -> Tuple[int, float]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("best_val", float("inf"))
    return start_epoch, best_val


# -----------------------------
# Build student & teacher
# -----------------------------

class IdentityPoolHead(nn.Module):
    """
    Wrap a timm backbone and pool to a vector.
    If the model returns a feature map, apply global average pooling.
    """
    def __init__(self, backbone_name: str, embed_dim: int):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.out_dim = embed_dim


        self.gap = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = self.gap(feats).flatten(1)
        return feats  # [B, D']


def build_student(cfg: Dict[str, Any], device: torch.device) -> Tuple[BreezeCLIP, AutoTokenizer]:
    # text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_backbone"])

    # vision encoder
    image_encoder = IdentityPoolHead(cfg["vision_backbone"], embed_dim=cfg["vision_embed_dim"])

    text_encoder = AutoModel.from_pretrained(cfg["text_backbone"])

    image_proj = nn.Linear(cfg["vision_embed_dim"], cfg["proj_embed_dim"])
    text_proj  = nn.Linear(text_encoder.config.hidden_size, cfg["proj_embed_dim"])

    model = BreezeCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_proj=image_proj,
        text_proj=text_proj,
        temperature=cfg.get("temperature", 0.07)
    )
    model = model.to(device)
    return model, tokenizer


def build_teacher(cfg: Dict[str, Any], device: torch.device) -> Tuple[CLIPModel, CLIPProcessor]:
    teacher = CLIPModel.from_pretrained(cfg["teacher_name"]).to(device)
    teacher.eval()
    processor = CLIPProcessor.from_pretrained(cfg["teacher_name"])
    return teacher, processor


# -----------------------------
# Data
# -----------------------------

def build_dataloaders(cfg: Dict[str, Any], tokenizer: AutoTokenizer):
    from torchvision import transforms

    # 学生图像变换（与 timm backbone 习惯一致）
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225],
        )
    ])

    ds = PairedFolderDataset(
        image_folder=cfg["image_folder"],
        text_folder=cfg["text_folder"],
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_text_length=cfg.get("max_text_length", 77),
        return_stem=False,
    )

    train_ratio = float(cfg.get("train_ratio", 0.8))
    n_total = len(ds)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    g = torch.Generator().manual_seed(cfg.get("seed", 42))
    train_set, val_set = random_split(ds, [n_train, n_val], generator=g)

    dl_train = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 4), pin_memory=True, drop_last=True
    )
    dl_val = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg.get("num_workers", 4), pin_memory=True, drop_last=False
    )
    return dl_train, dl_val


# -----------------------------
# Train / Eval loops
# -----------------------------

def forward_teacher(teacher: CLIPModel, processor: CLIPProcessor, batch: Dict[str, torch.Tensor], device):

    pixel = batch["pixel_values"]

    mean = torch.tensor([0.485, 0.456, 0.406], device=pixel.device)[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=pixel.device)[None, :, None, None]
    img_01 = (pixel * std + mean).clamp(0, 1)

    raw_texts = batch.get("raw_text", None)
    if raw_texts is None:

        raw_texts = [""] * img_01.size(0)

    with torch.no_grad():
        proc = processor(images=[img_01[i].detach().cpu().permute(1, 2, 0).numpy() for i in range(img_01.size(0))],
                         text=list(raw_texts), return_tensors="pt", padding=True, truncation=True)
        proc = {k: v.to(device) for k, v in proc.items()}
        out = teacher(**proc)

        img_emb_t = F.normalize(out.image_embeds, dim=-1)   # [B, D]
        txt_emb_t = F.normalize(out.text_embeds, dim=-1)    # [B, D]
    return img_emb_t, txt_emb_t


def run_epoch(model: BreezeCLIP,
              teacher: CLIPModel,
              processor: CLIPProcessor,
              loader: DataLoader,
              optimizer: torch.optim.Optimizer,
              scaler: torch.cuda.amp.GradScaler,
              device: torch.device,
              alpha: float,
              beta: float,
              temperature: float,
              train: bool = True) -> Dict[str, float]:

    model.train(train)
    total, loss_sum, loss_ctr_sum, loss_dst_sum = 0, 0.0, 0.0, 0.0
    t0 = time.time()

    for batch in loader:
        batch = to_device(batch, device)

        with torch.cuda.amp.autocast():

            img_emb_s, txt_emb_s = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )


            img_emb_t, txt_emb_t = forward_teacher(teacher, processor, batch, device)

            # losses
            loss_ctr = info_nce_loss(img_emb_s, txt_emb_s, temperature)
            loss_dst = F.mse_loss(img_emb_s, img_emb_t) + F.mse_loss(txt_emb_s, txt_emb_t)
            loss = alpha * loss_ctr + beta * loss_dst

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).step(optimizer)
            scaler.update()

        bs = img_emb_s.size(0)
        total += bs
        loss_sum += loss.item() * bs
        loss_ctr_sum += loss_ctr.item() * bs
        loss_dst_sum += loss_dst.item() * bs

    elapsed = time.time() - t0
    return {
        "loss": loss_sum / max(total, 1),
        "loss_ctr": loss_ctr_sum / max(total, 1),
        "loss_dst": loss_dst_sum / max(total, 1),
        "time": elapsed,
        "ips": total / max(elapsed, 1e-6)
    }


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", None))

    # build student & teacher
    model, tokenizer = build_student(cfg, device)
    teacher, processor = build_teacher(cfg, device)
    temperature = float(cfg.get("temperature", 0.07))
    alpha = float(cfg.get("alpha", 1.0))
    beta = float(cfg.get("beta", 1.0))

    # data
    dl_train, dl_val = build_dataloaders(cfg, tokenizer)

    # optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)))
    scaler = torch.cuda.amp.GradScaler()

    # resume
    start_epoch = 1
    best_val = float("inf")
    if cfg.get("resume"):
        if os.path.isfile(cfg["resume"]):
            start_epoch, best_val = load_checkpoint(cfg["resume"], model, optimizer, scaler)
            print(f"[Resume] Loaded from {cfg['resume']}, start_epoch={start_epoch}, best_val={best_val:.4f}")
        else:
            print(f"[Resume] File not found: {cfg['resume']} (start from scratch)")

    # train loop
    epochs = int(cfg.get("num_epochs", 50))
    save_dir = cfg.get("saved_model_dir", "./checkpoints/distill_breezeclip")
    save_every = int(cfg.get("save_every", 1))
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        # train
        tr_log = run_epoch(model, teacher, processor, dl_train, optimizer, scaler, device,
                           alpha, beta, temperature, train=True)
        # val
        with torch.no_grad():
            va_log = run_epoch(model, teacher, processor, dl_val, optimizer, scaler, device,
                               alpha, beta, temperature, train=False)

        msg = (f"[Epoch {epoch:03d}] "
               f"train: loss={tr_log['loss']:.4f} (ctr={tr_log['loss_ctr']:.4f}, dst={tr_log['loss_dst']:.4f}), "
               f"ips={tr_log['ips']:.1f}/s | "
               f"val: loss={va_log['loss']:.4f} (ctr={va_log['loss_ctr']:.4f}, dst={va_log['loss_dst']:.4f}), "
               f"time={va_log['time']:.1f}s")
        print(msg)

        # checkpoint
        if (epoch % save_every) == 0:
            path = save_checkpoint(save_dir, epoch, model, optimizer, scaler, best_val)
            print(f"  -> saved checkpoint: {path}")

        # best
        if va_log["loss"] < best_val:
            best_val = va_log["loss"]
            best_path = os.path.join(save_dir, "breezeclip_best.pth")
            torch.save({"model": model.state_dict(), "best_val": best_val, "epoch": epoch}, best_path)
            print(f"  -> new best ({best_val:.4f}) saved to: {best_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()