import os
import json
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

from bcqlm import BreezeCLIP, VisualAdapter, DynamicGatedCrossAttention


# -----------------------------
# Dataset
# -----------------------------
class EvalDataset(Dataset):

    def __init__(self, json_path, image_folder, tokenizer, max_length=128):
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        q, a = item["question"], item["answer"]
        text_in = f"Question: {q}\nAnswer:"

        enc = self.tokenizer(
            text_in, return_tensors="pt",
            max_length=self.max_length, truncation=True, padding="max_length"
        )

        # load image
        img_path = os.path.join(self.image_folder, item["image"])
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.img_transform(image)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "label_text": a
        }


# -----------------------------
# Metrics
# -----------------------------
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


def compute_metrics(preds, refs):
    results = {}
    exact = sum([int(p.strip().lower() == r.strip().lower()) for p, r in zip(preds, refs)])
    results["exact_match"] = exact / len(refs)

    results.update(bleu.compute(predictions=preds, references=[[r] for r in refs]))
    results.update(rouge.compute(predictions=preds, references=refs))
    bs = bertscore.compute(predictions=preds, references=refs, lang="en")
    results["bertscore_f1"] = sum(bs["f1"]) / len(bs["f1"])
    return results


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(cfg):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Tokenizer & LLaMA
    llama_tokenizer = AutoTokenizer.from_pretrained(cfg["llama_dir"])
    llama_model = AutoModelForCausalLM.from_pretrained(cfg["llama_dir"]).to(device)

    # BreezeCLIP student
    breezeclip = BreezeCLIP(
        image_encoder=nn.Identity(),
        text_encoder=nn.Identity(),
        image_proj=nn.Identity(),
        text_proj=nn.Identity()
    )
    if os.path.isfile(cfg["breezeclip_path"]):
        ckpt = torch.load(cfg["breezeclip_path"], map_location="cpu")
        breezeclip.load_state_dict(ckpt["model"], strict=False)
    breezeclip = breezeclip.to(device).eval()

    # Fusion modules
    adapter = VisualAdapter(input_dim=cfg.get("proj_dim", 768),
                            output_dim=cfg["llama_embedding_dim"]).to(device)
    dgca = DynamicGatedCrossAttention(
        img_dim=cfg["llama_embedding_dim"],
        text_dim=cfg.get("proj_dim", 768),
        num_heads=8
    ).to(device)

    if os.path.isfile(cfg["adapter_path"]):
        adapter.load_state_dict(torch.load(cfg["adapter_path"], map_location="cpu"))
    if os.path.isfile(cfg["dynamic_attn_path"]):
        dgca.load_state_dict(torch.load(cfg["dynamic_attn_path"], map_location="cpu"))

    adapter.eval()
    dgca.eval()
    llama_model.eval()

    # Dataset & loader
    dataset = EvalDataset(cfg["eval_json_path"], cfg["image_folder"], llama_tokenizer)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

    preds, refs = [], []
    total_loss, total_count = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            labels = batch["label_text"]

            img_emb, txt_emb = breezeclip(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            # 2. VisualAdapter
            vis_emb = adapter(img_emb.unsqueeze(1))  # [B, 1, D]

            # 3. DGCA cross-attention
            fused, _ = dgca(vis_emb, txt_emb.unsqueeze(1), txt_emb)

            # 4. LLaMA generate
            outputs = llama_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=64,
                num_beams=3
            )
            out_texts = llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # 5. Loss
            loss_out = llama_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"]
            )
            loss = loss_out.loss

            preds.extend(out_texts)
            refs.extend(labels)
            total_loss += loss.item() * len(labels)
            total_count += len(labels)

    avg_loss = total_loss / max(total_count, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    metrics = compute_metrics(preds, refs)
    metrics["loss"] = avg_loss
    metrics["perplexity"] = ppl

    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to eval config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate_model(cfg)


if __name__ == "__main__":
    main()