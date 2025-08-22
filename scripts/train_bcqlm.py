import os
import time
import argparse
import json
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from bcqlm import BreezeCLIP, VisualAdapter, DynamicGatedCrossAttention


# ---------------------------
# Dataset: QA JSON → samples
# ---------------------------
class QADataset(Dataset):

    def __init__(self, json_files, image_folder, tokenizer, max_length=128):
        self.samples = []
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.samples.extend(data)

        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        q = item["question"]
        a = item["answer"]

        text_in = f"Question: {q}\nAnswer:"
        text_out = a

        enc = self.tokenizer(
            text_in, return_tensors="pt",
            max_length=self.max_length, truncation=True
        )
        dec = self.tokenizer(
            text_out, return_tensors="pt",
            max_length=self.max_length, truncation=True
        )

        sample = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": dec["input_ids"].squeeze(0),
            "image_path": os.path.join(self.image_folder, item["image"])
        }
        return sample


# ---------------------------
# Trainer
# ---------------------------
def train(cfg):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Tokenizer & LLaMA
    llama_tokenizer = AutoTokenizer.from_pretrained(cfg["llama_model_path"])
    llama_model = AutoModelForCausalLM.from_pretrained(cfg["llama_model_path"]).to(device)

    # BreezeCLIP (student, load from ckpt)
    breezeclip = BreezeCLIP(
        image_encoder=nn.Identity(),  # placeholder (load from ckpt)
        text_encoder=nn.Identity(),
        image_proj=nn.Identity(),
        text_proj=nn.Identity()
    )
    ckpt = torch.load(cfg["breezeclip_ckpt"], map_location="cpu")
    breezeclip.load_state_dict(ckpt["model"], strict=False)
    breezeclip = breezeclip.to(device)
    breezeclip.eval()
    # Fusion modules
    adapter = VisualAdapter(input_dim=cfg["proj_dim"], output_dim=cfg["llama_embedding_dim"]).to(device)
    dgca = DynamicGatedCrossAttention(
        img_dim=cfg["llama_embedding_dim"],
        text_dim=cfg["proj_dim"],
        num_heads=8
    ).to(device)

    # Optimizer
    params = list(adapter.parameters()) + list(dgca.parameters()) + list(llama_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"])

    # Dataset & DataLoader
    dataset = QADataset(
        cfg["json_files"],
        cfg["image_folder"],
        tokenizer=llama_tokenizer,
        max_length=cfg.get("max_length", 128)
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg.get("num_workers", 4))

    # Training loop
    for epoch in range(1, cfg["num_epochs"] + 1):
        llama_model.train()
        adapter.train()
        dgca.train()

        total_loss, total_count = 0.0, 0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


            B = batch["input_ids"].size(0)
            img_feats = torch.randn(B, 197, cfg["proj_dim"], device=device)


            txt_tokens = torch.randn(B, 77, cfg["proj_dim"], device=device)
            txt_global = txt_tokens.mean(dim=1)


            vis_emb = adapter(img_feats)

            # Cross-modal fusion
            fused, _ = dgca(vis_emb, txt_tokens, txt_global)

            inputs_embeds = fused
            outputs = llama_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
            optimizer.step()

            total_loss += loss.item() * B
            total_count += B

        avg_loss = total_loss / max(total_count, 1)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}, time={elapsed:.1f}s")

        # Save checkpoint
        os.makedirs(cfg["save_dir"], exist_ok=True)
        ckpt_path = os.path.join(cfg["save_dir"], f"bcqlm_epoch{epoch}.pth")
        torch.save({
            "adapter": adapter.state_dict(),
            "dgca": dgca.state_dict(),
            "llama": llama_model.state_dict(),
            "loss": avg_loss,
            "epoch": epoch
        }, ckpt_path)
        print(f"  -> checkpoint saved: {ckpt_path}")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()