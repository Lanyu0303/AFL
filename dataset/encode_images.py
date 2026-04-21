import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except ModuleNotFoundError:
    Image = None
    HAS_PIL = False

try:
    import torchvision.io as tvio
    HAS_TVIO = True
except ModuleNotFoundError:
    tvio = None
    HAS_TVIO = False


def load_image(path):
    try:
        if HAS_PIL:
            return Image.open(path).convert("RGB")
        if HAS_TVIO:
            img = tvio.read_image(str(path))  # [C,H,W], uint8
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] > 3:
                img = img[:3]
            return img
        raise RuntimeError(
            "Neither Pillow nor torchvision.io is available for image decoding. "
            "Install Pillow via: pip install Pillow"
        )
    except Exception:
        return None


def extract_image_features(model, inputs):
    try:
        raw_feats = model.get_image_features(**inputs)
    except Exception:
        # Compatibility fallback for environments where get_image_features returns
        # an unexpected object or is overridden.
        raw_feats = model(**inputs)

    if torch.is_tensor(raw_feats):
        return raw_feats
    if hasattr(raw_feats, "image_embeds") and torch.is_tensor(raw_feats.image_embeds):
        return raw_feats.image_embeds
    if hasattr(raw_feats, "pooler_output") and torch.is_tensor(raw_feats.pooler_output):
        return raw_feats.pooler_output
    if hasattr(raw_feats, "last_hidden_state") and torch.is_tensor(raw_feats.last_hidden_state):
        return raw_feats.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported image feature output type: {type(raw_feats)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./data/amazon_beauty/image_embeddings.npy")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow_empty_output", action="store_true", help="Allow writing an empty embeddings file.")
    args = parser.parse_args()

    device = torch.device(args.device)
    if not HAS_PIL and HAS_TVIO:
        print("[WARN] Pillow not found; using torchvision.io for image loading.")
    if not HAS_PIL and not HAS_TVIO:
        raise RuntimeError("Missing image backends. Install Pillow (`pip install Pillow`) or torchvision.")
    print(f"loading CLIP model {args.model_name} on {device}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    img_dir = Path(args.img_dir)
    id2path = {}
    skipped_non_integer_filenames = 0
    for p in img_dir.iterdir():
        if not p.is_file():
            continue
        try:
            iid = int(p.stem)
        except ValueError:
            skipped_non_integer_filenames += 1
            continue
        id2path[iid] = p

    item_ids = sorted(id2path.keys())
    print(f"found numeric-id images: {len(item_ids)} (skipped non-numeric filenames: {skipped_non_integer_filenames})")
    embeddings = {}
    decode_failed = 0

    for start in tqdm(range(0, len(item_ids), args.batch_size)):
        batch_ids = item_ids[start:start + args.batch_size]
        images = []
        valid_ids = []
        for iid in batch_ids:
            img = load_image(id2path[iid])
            if img is not None:
                images.append(img)
                valid_ids.append(iid)
            else:
                decode_failed += 1
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = extract_image_features(model, inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu().numpy()
        for iid, feat in zip(valid_ids, feats):
            embeddings[iid] = feat

    print(f"decoded successfully: {len(embeddings)} | decode failed: {decode_failed}")
    if not embeddings and not args.allow_empty_output:
        raise RuntimeError(
            "No image embeddings were produced. Common causes: stale/invalid image files, "
            "unsupported image formats, or wrong --img_dir. "
            "Re-run download_images.py and verify files, or pass --allow_empty_output to force-save."
        )

    np.save(args.output_path, embeddings)
    print(f"saved embeddings: {args.output_path}")
    if embeddings:
        print(f"embedding dim: {next(iter(embeddings.values())).shape[0]}")


if __name__ == "__main__":
    main()
