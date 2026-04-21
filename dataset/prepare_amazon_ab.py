import argparse
import glob
import os
import pickle
import shutil
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from dataset.amazon_preprocess import _load_reviews, _reindex
except ModuleNotFoundError:
    # fallback when executed as a plain script from the dataset folder
    from amazon_preprocess import _load_reviews, _reindex


def _pick_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _find_reviews_gz(source_data_dir):
    patterns = [
        os.path.join(source_data_dir, "raw_data", "reviews_*_5.json.gz"),
        os.path.join(source_data_dir, "reviews_*_5.json.gz"),
    ]
    for p in patterns:
        files = sorted(glob.glob(p))
        if files:
            return files[0]
    return None


def _load_source_interactions(source_data_dir):
    reviews_gz = _find_reviews_gz(source_data_dir)
    if reviews_gz is not None:
        reviews_df = _load_reviews(reviews_gz)
        reviews_df, _, _ = _reindex(reviews_df)
        return reviews_df

    all_csv = os.path.join(source_data_dir, "all.csv")
    if os.path.exists(all_csv):
        df = pd.read_csv(all_csv)
        required = {"user_id", "item_id", "timestamp"}
        if not required.issubset(df.columns):
            raise ValueError(f"{all_csv} must contain columns: {sorted(required)}")
        df = df.sort_values(["timestamp", "user_id"], kind="mergesort").reset_index(drop=True)
        return df

    raise FileNotFoundError(
        f"Could not find reviews_*.json.gz or all.csv under {source_data_dir}. "
        "Please provide a source_data_dir that already contains interaction history."
    )


def build_amazon_ab(
    source_data_dir,
    output_dir,
    copy_from_dir=None,
    min_user_interactions=20,
    window_size=20,
):
    reviews_df = _load_source_interactions(source_data_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Prefer mappings/statistics from source_data_dir to keep ids consistent with
    # the Amazon dataset used for SASRec training.
    copy_from_dir = copy_from_dir or source_data_dir
    src_id2name = _pick_existing(
        [
            os.path.join(source_data_dir, "id2name_AB_llara.txt"),
            os.path.join(source_data_dir, "id2name.txt"),
            os.path.join(copy_from_dir, "id2name_AB_llara.txt"),
            os.path.join(copy_from_dir, "id2name.txt"),
        ]
    )
    dst_id2name = os.path.join(output_dir, "id2name_AB_llara.txt")
    if src_id2name is None:
        raise FileNotFoundError(f"Missing source id2name file in: {copy_from_dir}")
    shutil.copy2(src_id2name, dst_id2name)

    src_statis = _pick_existing(
        [
            os.path.join(source_data_dir, "data_statis.df"),
            os.path.join(copy_from_dir, "data_statis.df"),
        ]
    )
    dst_statis = os.path.join(output_dir, "data_statis.df")
    if src_statis is None:
        raise FileNotFoundError(
            f"Missing source data_statis in {source_data_dir} and fallback {copy_from_dir}"
        )
    shutil.copy2(src_statis, dst_statis)

    # Build test_llara.pkl rows expected by LastfmABDataset
    rows = []
    for user_id, grp in reviews_df.groupby("user_id"):
        full_list = grp.sort_values("timestamp", kind="mergesort")["item_id"].tolist()
        if len(full_list) < max(min_user_interactions, window_size):
            continue
        item_list = full_list[-window_size:]
        rows.append(
            {
                "user_id": int(user_id),
                "item_list": item_list,
                "full_list": full_list,
            }
        )

    out_pkl = os.path.join(output_dir, "test_llara.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(rows, f)

    print(f"copied {src_id2name} -> {dst_id2name}")
    print(f"copied {src_statis} -> {dst_statis}")
    print(f"wrote {out_pkl} with {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data_dir", type=str, required=True, help="Amazon source dir used for preprocess")
    parser.add_argument("--output_dir", type=str, required=True, help="Output *_ab directory")
    parser.add_argument(
        "--copy_from_dir",
        type=str,
        default="",
        help="Optional fallback directory to copy id2name/data_statis from.",
    )
    parser.add_argument("--min_user_interactions", type=int, default=20)
    parser.add_argument("--window_size", type=int, default=20, help="Length of item_list in test_llara.pkl")
    args = parser.parse_args()

    build_amazon_ab(
        source_data_dir=args.source_data_dir,
        output_dir=args.output_dir,
        copy_from_dir=args.copy_from_dir,
        min_user_interactions=args.min_user_interactions,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
