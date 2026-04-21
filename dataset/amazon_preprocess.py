import argparse
import ast
import gzip
import os
from collections import defaultdict
from urllib.request import urlretrieve

import pandas as pd


BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"


def _parse_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield ast.literal_eval(line)


def _download_if_missing(data_dir, category):
    os.makedirs(data_dir, exist_ok=True)
    reviews_name = f"reviews_{category}_5.json.gz"
    meta_name = f"meta_{category}.json.gz"
    reviews_path = os.path.join(data_dir, reviews_name)
    meta_path = os.path.join(data_dir, meta_name)
    if not os.path.exists(reviews_path):
        urlretrieve(f"{BASE_URL}/{reviews_name}", reviews_path)
    if not os.path.exists(meta_path):
        urlretrieve(f"{BASE_URL}/{meta_name}", meta_path)
    return reviews_path, meta_path


def _load_reviews(reviews_path):
    rows = []
    for r in _parse_gz(reviews_path):
        if "reviewerID" not in r or "asin" not in r or "unixReviewTime" not in r:
            continue
        rows.append(
            {
                "user_raw": r["reviewerID"],
                "item_raw": r["asin"],
                "timestamp": int(r["unixReviewTime"]),
            }
        )
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["user_raw", "item_raw", "timestamp"])
    df = df.sort_values(["timestamp", "user_raw"], kind="mergesort").reset_index(drop=True)
    return df


def _load_meta(meta_path):
    def _extract_img_url(record):
        # Legacy Amazon metadata commonly uses "imUrl".
        url = (record.get("imUrl") or "").strip()
        if url:
            return url

        # Some newer dumps store image urls as lists.
        for key in ("imageURLHighRes", "imageURL"):
            values = record.get(key) or []
            if isinstance(values, str):
                values = [values]
            for v in values:
                v = (v or "").strip()
                if v:
                    return v
        return ""

    meta = {}
    for m in _parse_gz(meta_path):
        asin = m.get("asin")
        if not asin:
            continue
        title = (m.get("title") or "").replace("\n", " ").strip()
        img_url = _extract_img_url(m)
        meta[asin] = {
            "title": title if title else f"item_{asin}",
            "img_url": img_url,
        }
    return meta


def _reindex(df):
    users = sorted(df["user_raw"].unique())
    items = sorted(df["item_raw"].unique())
    user2id = {u: idx for idx, u in enumerate(users)}
    item2id = {i: idx for idx, i in enumerate(items)}
    df = df.copy()
    df["user_id"] = df["user_raw"].map(user2id)
    df["item_id"] = df["item_raw"].map(item2id)
    return df, user2id, item2id


def _build_sequences(df, min_user_interactions):
    seqs = []
    for user_id, grp in df.groupby("user_id"):
        items = grp.sort_values("timestamp", kind="mergesort")["item_id"].tolist()
        if len(items) >= min_user_interactions:
            seqs.append((user_id, items))
    return seqs


def _split_sequences(seqs, max_len=10, split_mode="leave_one_out"):
    train_rows, val_rows, test_rows = [], [], []
    for user_id, items in seqs:
        n = len(items)
        if split_mode == "leave_one_out":
            if n < 3:
                continue
            for idx in range(1, n - 2):
                seq = items[max(0, idx - max_len):idx]
                train_rows.append((user_id, seq, len(seq), items[idx]))
            val_idx = n - 2
            test_idx = n - 1
            val_seq = items[max(0, val_idx - max_len):val_idx]
            test_seq = items[max(0, test_idx - max_len):test_idx]
            val_rows.append((user_id, val_seq, len(val_seq), items[val_idx]))
            test_rows.append((user_id, test_seq, len(test_seq), items[test_idx]))
        elif split_mode == "ratio_8_1_1":
            train_end = max(1, int(n * 0.8))
            val_end = max(train_end + 1, int(n * 0.9))
            val_end = min(val_end, n - 1)
            for idx in range(1, train_end):
                seq = items[max(0, idx - max_len):idx]
                train_rows.append((user_id, seq, len(seq), items[idx]))
            for idx in range(train_end, val_end):
                seq = items[max(0, idx - max_len):idx]
                val_rows.append((user_id, seq, len(seq), items[idx]))
            for idx in range(val_end, n):
                seq = items[max(0, idx - max_len):idx]
                test_rows.append((user_id, seq, len(seq), items[idx]))
        else:
            raise ValueError(f"Unsupported split_mode: {split_mode}")
    cols = ["user_id", "seq", "len_seq", "next"]
    return (
        pd.DataFrame(train_rows, columns=cols),
        pd.DataFrame(val_rows, columns=cols),
        pd.DataFrame(test_rows, columns=cols),
    )


def build_amazon_dataset(data_dir, category="Beauty", min_user_interactions=20, split_mode="leave_one_out"):
    reviews_path, meta_path = _download_if_missing(data_dir, category)
    reviews_df = _load_reviews(reviews_path)
    reviews_df, _, item2id = _reindex(reviews_df)
    meta = _load_meta(meta_path)

    seqs = _build_sequences(reviews_df, min_user_interactions=min_user_interactions)
    train_df, val_df, test_df = _split_sequences(seqs, split_mode=split_mode)

    seq_size = 10
    item_num = len(item2id)
    pad_token = item_num
    data_statis = pd.DataFrame({"seq_size": [seq_size], "item_num": [item_num]})
    data_statis_path = os.path.join(data_dir, "data_statis.df")
    data_statis.to_pickle(data_statis_path)
    print(f"saved data_statis.df: seq_size={seq_size}, item_num={item_num}")

    def pad(seq):
        out = list(seq)
        while len(out) < seq_size:
            out.append(pad_token)
        return out[:seq_size]

    for df in (train_df, val_df, test_df):
        df["seq"] = df["seq"].apply(pad)
        # Keep list[int] dtype to avoid string sequences after save/load.
        df["seq"] = df["seq"].apply(lambda x: [int(i) for i in x])
        df["len_seq"] = df["len_seq"].astype(int).clip(upper=seq_size)
        # Align schema with downstream loaders expecting [seq, len_seq, next].
        if "user_id" in df.columns:
            df.drop(columns=["user_id"], inplace=True)

    id2name = defaultdict(str)
    id2imgurl = defaultdict(str)
    for raw, iid in item2id.items():
        info = meta.get(raw, {"title": f"item_{raw}", "img_url": ""})
        id2name[iid] = info.get("title", f"item_{raw}")
        id2imgurl[iid] = info.get("img_url", "")

    with open(os.path.join(data_dir, "id2name.txt"), "w", encoding="utf-8") as f:
        for iid in sorted(id2name.keys()):
            f.write(f"{iid}::{id2name[iid]}\n")
    id2imgurl_path = os.path.join(data_dir, "id2imgurl.txt")
    with open(id2imgurl_path, "w", encoding="utf-8") as f:
        for iid in sorted(id2imgurl.keys()):
            f.write(f"{iid}::{id2imgurl[iid]}\n")
    non_empty_img_urls = sum(1 for v in id2imgurl.values() if v)
    print(f"id2imgurl.txt written with {len(id2imgurl)} entries ({non_empty_img_urls} non-empty URLs): {id2imgurl_path}")

    train_df.to_pickle(os.path.join(data_dir, "train_data.df"))
    val_df.to_pickle(os.path.join(data_dir, "Val_data.df"))
    test_df.to_pickle(os.path.join(data_dir, "Test_data.df"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/amazon_beauty")
    parser.add_argument("--category", type=str, default="Beauty")
    parser.add_argument("--min_user_interactions", type=int, default=20)
    parser.add_argument("--split_mode", type=str, default="leave_one_out", choices=["leave_one_out", "ratio_8_1_1"])
    args = parser.parse_args()
    build_amazon_dataset(args.data_dir, args.category, args.min_user_interactions, args.split_mode)
