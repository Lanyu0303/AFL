import argparse
import ast
import gzip
import os
from typing import Dict

import pandas as pd


def parse_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield ast.literal_eval(line)


def load_item_id_maps(data_dir: str):
    id2name_path = os.path.join(data_dir, "id2name.txt")
    if not os.path.exists(id2name_path):
        raise FileNotFoundError(f"Missing id2name.txt: {id2name_path}")

    item_id2title: Dict[int, str] = {}
    title2item_id: Dict[str, int] = {}
    with open(id2name_path, "r", encoding="utf-8") as f:
        for line in f:
            if "::" not in line:
                continue
            iid_s, title = line.strip().split("::", 1)
            iid = int(iid_s)
            title = title.strip()
            item_id2title[iid] = title
            if title and title not in title2item_id:
                title2item_id[title] = iid
    return item_id2title, title2item_id


def extract_image_url(record):
    url = (record.get("imUrl") or "").strip()
    if url:
        return url
    for key in ("imageURLHighRes", "imageURL"):
        values = record.get(key) or []
        if isinstance(values, str):
            values = [values]
        for v in values:
            v = (v or "").strip()
            if v:
                return v
    return ""


def build_meta_df(data_dir: str, meta_gz: str, output_path: str):
    item_id2title, _ = load_item_id_maps(data_dir)

    asin_meta = {}
    meta_by_title = {}
    for rec in parse_gz(meta_gz):
        asin = (rec.get("asin") or "").strip()
        if not asin:
            continue
        title = (rec.get("title") or "").replace("\n", " ").strip()
        desc = rec.get("description", "")
        if isinstance(desc, list):
            desc = " ".join([str(x).strip() for x in desc if str(x).strip()])
        else:
            desc = str(desc or "").strip()
        brand = str(rec.get("brand") or "").strip()
        cats = rec.get("categories") or []
        category = ""
        if isinstance(cats, list) and len(cats) > 0:
            if isinstance(cats[0], list):
                category = " > ".join([str(x).strip() for x in cats[0] if str(x).strip()])
            else:
                category = " > ".join([str(x).strip() for x in cats if str(x).strip()])
        asin_meta[asin] = {
            "asin": asin,
            "title": title,
            "description": desc,
            "brand": brand,
            "category": category,
            "image_url": extract_image_url(rec),
        }
        if title and title not in meta_by_title:
            meta_by_title[title] = asin_meta[asin]

    rows = []
    matched = 0
    for item_id, mapped_title in item_id2title.items():
        meta = None
        if mapped_title in meta_by_title:
            meta = meta_by_title[mapped_title]
        if meta is None:
            meta = {
                "asin": "",
                "title": mapped_title,
                "description": "",
                "brand": "",
                "category": "",
                "image_url": "",
            }
        else:
            matched += 1

        rows.append(
            {
                "item_id": int(item_id),
                "item_raw": meta["asin"],
                "title": meta["title"] or mapped_title,
                "description": meta["description"],
                "brand": meta["brand"],
                "category": meta["category"],
                "image_url": meta["image_url"],
            }
        )

    meta_df = pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)
    meta_df.to_pickle(output_path)

    cov = {
        "num_items": int(len(meta_df)),
        "title_cov": float(meta_df["title"].fillna("").astype(str).str.len().gt(0).mean()),
        "description_cov": float(meta_df["description"].fillna("").astype(str).str.len().gt(0).mean()),
        "brand_cov": float(meta_df["brand"].fillna("").astype(str).str.len().gt(0).mean()),
        "category_cov": float(meta_df["category"].fillna("").astype(str).str.len().gt(0).mean()),
        "image_url_cov": float(meta_df["image_url"].fillna("").astype(str).str.len().gt(0).mean()),
        "meta_title_exact_match_count": int(matched),
    }
    print(cov)
    print(f"saved item_meta.df -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--meta_gz", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    out = args.output_path or os.path.join(args.data_dir, "item_meta.df")
    build_meta_df(args.data_dir, args.meta_gz, out)
