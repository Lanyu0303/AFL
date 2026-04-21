import argparse
import json
import os

import numpy as np
import pandas as pd


def load_item_num(data_statis_path: str) -> int:
    statis = pd.read_pickle(data_statis_path)
    return int(statis["item_num"].iloc[0])


def load_image_dict(path: str):
    arr = np.load(path, allow_pickle=True)
    if hasattr(arr, "item"):
        try:
            v = arr.item()
            if isinstance(v, dict):
                return v
        except ValueError:
            pass
    if isinstance(arr, dict):
        return arr
    raise ValueError(f"Expected dict-style image embeddings in {path}")


def align_image_features(image_dict, item_num):
    if not image_dict:
        raise ValueError("image embedding dict is empty")
    sample_dim = int(next(iter(image_dict.values())).shape[0])
    mat = np.zeros((item_num + 1, sample_dim), dtype=np.float32)
    present = 0
    for iid, vec in image_dict.items():
        iid = int(iid)
        if 0 <= iid < item_num:
            mat[iid] = np.asarray(vec, dtype=np.float32)
            present += 1
    mat[item_num] = 0.0  # pad row
    return mat, present


def ensure_dense(path: str, item_num: int, name: str):
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.shape}")
    if arr.shape[0] < item_num + 1:
        raise ValueError(f"{name} rows {arr.shape[0]} < required {item_num + 1}")
    return arr[: item_num + 1].astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_statis", type=str, required=True)
    parser.add_argument("--image_embeddings", type=str, default="")
    parser.add_argument("--text_embeddings", type=str, default="")
    parser.add_argument("--out_image", type=str, default="")
    parser.add_argument("--out_text", type=str, default="")
    parser.add_argument("--report_json", type=str, default="")
    args = parser.parse_args()

    item_num = load_item_num(args.data_statis)
    report = {"item_num": item_num}

    if args.image_embeddings:
        img_dict = load_image_dict(args.image_embeddings)
        img_mat, present = align_image_features(img_dict, item_num)
        out_image = args.out_image or os.path.join(os.path.dirname(args.image_embeddings), "item_image_emb.npy")
        np.save(out_image, img_mat)
        report["image_shape"] = list(img_mat.shape)
        report["image_present_items"] = int(present)
        report["image_coverage"] = float(present / max(item_num, 1))
        print(f"saved aligned image matrix: {out_image}")

    if args.text_embeddings:
        txt_mat = ensure_dense(args.text_embeddings, item_num, "text embeddings")
        out_text = args.out_text or os.path.join(os.path.dirname(args.text_embeddings), "item_text_emb.npy")
        np.save(out_text, txt_mat)
        nonzero = int((np.linalg.norm(txt_mat[:-1], axis=1) > 0).sum())
        report["text_shape"] = list(txt_mat.shape)
        report["text_present_items"] = nonzero
        report["text_coverage"] = float(nonzero / max(item_num, 1))
        print(f"saved aligned text matrix: {out_text}")

    report_path = args.report_json or os.path.join(os.path.dirname(args.data_statis), "multimodal_alignment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"saved report: {report_path}")
