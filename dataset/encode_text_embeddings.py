import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def build_text(row):
    title = str(row.get("title", "") or "").strip()
    desc = str(row.get("description", "") or "").strip()
    brand = str(row.get("brand", "") or "").strip()
    category = str(row.get("category", "") or "").strip()

    parts = []
    if title:
        parts.append(f"title: {title}")
    if brand:
        parts.append(f"brand: {brand}")
    if category:
        parts.append(f"category: {category}")
    if desc:
        parts.append(f"description: {desc}")
    return " | ".join(parts)


def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
    denom = mask.sum(1).clamp(min=1e-9)
    return (last_hidden * mask).sum(1) / denom


def encode_texts(meta_df, model_name, batch_size, max_length, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    item_num = int(meta_df["item_id"].max()) + 1
    emb_dim = int(model.config.hidden_size)
    output = np.zeros((item_num + 1, emb_dim), dtype=np.float32)

    texts = [build_text(r) for _, r in meta_df.iterrows()]
    ids = meta_df["item_id"].astype(int).tolist()
    missing = 0

    for start in range(0, len(texts), batch_size):
        batch_text = texts[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]

        encoded = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**encoded)
            emb = mean_pool(out.last_hidden_state, encoded["attention_mask"])
            emb = torch.nn.functional.normalize(emb, dim=-1)
            emb = emb.detach().cpu().numpy().astype(np.float32)

        for iid, txt, vec in zip(batch_ids, batch_text, emb):
            if not txt:
                missing += 1
                continue
            output[int(iid)] = vec

    output[item_num] = 0.0  # pad item row
    return output, missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_meta", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.item_meta):
        raise FileNotFoundError(args.item_meta)

    meta_df = pd.read_pickle(args.item_meta)
    if "item_id" not in meta_df.columns:
        raise ValueError("item_meta.df must include item_id column")

    emb, missing = encode_texts(
        meta_df=meta_df,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=torch.device(args.device),
    )
    np.save(args.output_path, emb)
    print(f"saved text embeddings: {args.output_path} shape={emb.shape} missing_text_rows={missing}")
