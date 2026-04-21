import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from dataset.amazon_dataset import AmazonDataset
    from dataset.lastfmAB_dataset import LastfmABDataset
    from utils.model import SASRec
except ModuleNotFoundError:
    # fallback when executed as plain script in environments with different PYTHONPATH
    sys.path.insert(0, os.path.join(ROOT_DIR, "dataset"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "utils"))
    from amazon_dataset import AmazonDataset
    from lastfmAB_dataset import LastfmABDataset
    from model import SASRec


class SeqTrainDataset(Dataset):
    def __init__(self, df, target_col):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            torch.tensor(row["seq"], dtype=torch.long),
            torch.tensor(int(row["len_seq"]), dtype=torch.long),
            torch.tensor(int(row[self.target_col]), dtype=torch.long),
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def validate_feature_matrix(path: str, item_num: int, feature_name: str):
    if not path:
        return
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{feature_name} must be 2D, got {arr.shape}")
    if arr.shape[0] < item_num + 1:
        raise ValueError(
            f"{feature_name} rows {arr.shape[0]} are fewer than required {item_num + 1} (including pad row)"
        )

def build_data_statis(data_dir: str, force_rebuild: bool = False):
    train_path = os.path.join(data_dir, "train_data.df")
    val_path = os.path.join(data_dir, "Val_data.df")
    test_path = os.path.join(data_dir, "Test_data.df")
    train_df = pd.read_pickle(train_path)
    val_df = pd.read_pickle(val_path)
    test_df = pd.read_pickle(test_path)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    target_col = "next" if "next" in all_df.columns else "item_id"
    out_path = os.path.join(data_dir, "data_statis.df")

    seq_size = 10
    max_seq = int(all_df["seq"].apply(max).max())
    max_target = int(all_df[target_col].max())
    # item_num in SASRec is both:
    #  - the output dimension upper bound (targets must be < item_num)
    #  - the padding token id used in sequences.
    # We therefore require item_num >= max_target + 1 and >= any token seen in seq.
    inferred_item_num = max(max_seq, max_target + 1)

    if os.path.exists(out_path) and not force_rebuild:
        statis_df = pd.read_pickle(out_path)
        cached_item_num = int(statis_df["item_num"][0])
        if cached_item_num >= inferred_item_num:
            return statis_df, train_df, val_df, test_df, target_col
        print(
            f"existing data_statis.df item_num={cached_item_num} is too small; "
            f"rebuilding with item_num={inferred_item_num}"
        )

    statis_df = pd.DataFrame({"seq_size": [seq_size], "item_num": [inferred_item_num]})
    statis_df.to_pickle(out_path)
    return statis_df, train_df, val_df, test_df, target_col


def evaluate_hits(model, df, device, target_col, k=10):
    model.eval()
    hits_k = 0
    hits_1 = 0
    total = 0
    with torch.no_grad():
        for _, row in df.iterrows():
            seq = torch.tensor([row["seq"]], dtype=torch.long, device=device)
            len_seq = np.array([int(row["len_seq"])])
            target = int(row[target_col])
            logits = model.forward_eval(seq, len_seq).detach().cpu()

            # If neg_item_id exists, evaluate on candidate set (target + negatives)
            if "neg_item_id" in row and row["neg_item_id"] is not None:
                negs = row["neg_item_id"]
                if not isinstance(negs, (list, tuple)):
                    negs = [int(negs)]
                candidates = [target] + [int(x) for x in list(negs)]
                candidate_scores = logits[candidates]
                topk = min(k, len(candidates))
                _, topk_idx = candidate_scores.topk(topk)
                topk_items = [candidates[i] for i in topk_idx.tolist()]
                hits_k += int(target in topk_items)
                top1_idx = int(torch.argmax(candidate_scores).item())
                hits_1 += int(candidates[top1_idx] == target)
            else:
                # Full ranking fallback
                topk = min(k, logits.shape[0])
                _, topk_idx = logits.topk(topk)
                hits_k += int(target in topk_idx.tolist())
                hits_1 += int(int(torch.argmax(logits).item()) == target)
            total += 1
    denom = max(total, 1)
    return hits_1 / denom, hits_k / denom


def train_one_config(train_df, val_df, item_num, seq_size, device, epochs, batch_size, lr, l2, embedding_dim, target_col, mm_config=None):
    model = SASRec(embedding_dim, item_num, seq_size, 0.1, device, mm_config=mm_config).to(device)
    ds = SeqTrainDataset(train_df, target_col=target_col)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_history = []
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for seq, len_seq, target in dl:
            seq = seq.to(device)
            len_seq = len_seq.cpu().numpy()
            target = target.to(device)
            logits = model.forward(seq, len_seq)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_hit1, val_hit10 = evaluate_hits(model, val_df, device, target_col=target_col, k=10)
        avg_loss = total_loss / max(len(dl), 1)
        epoch_history.append(
            {
                "epoch": ep,
                "train_loss": avg_loss,
                "val_hit@1": val_hit1,
                "val_hit@10": val_hit10,
            }
        )
        print(f"[l2={l2}][epoch {ep}] loss={avg_loss:.4f} val_hit@1={val_hit1:.4f} val_hit@10={val_hit10:.4f}")

    final_hit1, final_hit10 = evaluate_hits(model, val_df, device, target_col=target_col, k=10)
    return model, final_hit1, final_hit10, epoch_history


def train_sasrec(train_df, val_df, item_num, seq_size, device, epochs, batch_size, lr, embedding_dim, l2_grid, target_col, mm_config=None):
    best_model = None
    best_val = -1.0
    best_l2 = None
    records = []
    best_history = []
    for l2 in l2_grid:
        model, val_hit1, val_hit10, epoch_history = train_one_config(
            train_df=train_df,
            val_df=val_df,
            item_num=item_num,
            seq_size=seq_size,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            l2=l2,
            embedding_dim=embedding_dim,
            target_col=target_col,
            mm_config=mm_config,
        )
        records.append({"l2": l2, "val_hit@1": val_hit1, "val_hit@10": val_hit10, "history": epoch_history})
        if val_hit10 > best_val:
            best_val = val_hit10
            best_model = model
            best_l2 = l2
            best_history = epoch_history

    return best_model, best_l2, best_val, records, best_history


def build_prior_csv(model, data_dir, stage, cans_num, out_csv, device):
    dataset = AmazonDataset(data_dir=data_dir, stage=stage, cans_num=cans_num, sep=", ", no_augment=True)
    rows = []
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            seq = torch.tensor([sample["seq"]], dtype=torch.long, device=device)
            len_seq = np.array([int(sample["len_seq"])])
            logits = model.forward_eval(seq, len_seq).detach().cpu()

            # Score only candidate items. Some datasets may have an off-by-one
            # mismatch between candidate ids and model output indices.
            candidate_ids = [int(x) for x in sample["cans"]]
            model_indices = []
            kept_candidates = []
            for cid in candidate_ids:
                if 0 <= cid < logits.shape[0]:
                    model_indices.append(cid)
                    kept_candidates.append(cid)
                elif 0 <= cid - 1 < logits.shape[0]:
                    model_indices.append(cid - 1)
                    kept_candidates.append(cid)

            if not model_indices:
                raise ValueError(
                    f"No valid candidate indices for sample id={sample['id']}; "
                    f"candidate range=[{min(candidate_ids)}, {max(candidate_ids)}], logits_size={logits.shape[0]}"
                )

            candidate_scores = logits[model_indices]
            pred_pos = int(torch.argmax(candidate_scores).item())
            pred_item_id = kept_candidates[pred_pos]
            pred_name = dataset.item_id2name.get(pred_item_id, str(pred_item_id))
            real_name = sample.get("correct_answer", sample.get("item_name", ""))
            rows.append({"id": sample["id"], "generate": pred_name, "real": real_name})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote prior file: {out_csv} ({len(rows)} rows)")


def build_prior_csv_ab(model, ab_data_dir, out_csv, train_cans_num, device):
    test_llara_path = os.path.join(ab_data_dir, "test_llara.pkl")
    id2name_ab_path = os.path.join(ab_data_dir, "id2name_AB_llara.txt")
    id2name_fallback = os.path.join(ab_data_dir, "id2name.txt")
    if not os.path.exists(test_llara_path):
        raise FileNotFoundError(
            f"ab_data_dir must contain test_llara.pkl, but not found: {test_llara_path}"
        )
    if not os.path.exists(id2name_ab_path):
        if os.path.exists(id2name_fallback):
            shutil.copy2(id2name_fallback, id2name_ab_path)
            print(f"[build_prior_csv_ab] copied fallback id2name.txt -> {id2name_ab_path}")
        else:
            raise FileNotFoundError(
                f"ab_data_dir must contain id2name_AB_llara.txt (or id2name.txt fallback), "
                f"but neither exists in: {ab_data_dir}"
            )

    dataset = LastfmABDataset(
        data_dir=ab_data_dir,
        stage="test",
        train_cans_num=train_cans_num,
        eval_cans_num=train_cans_num,
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            seq = torch.tensor([sample["seq"]], dtype=torch.long, device=device)
            len_seq = np.array([int(sample["len_seq"])])
            logits = model.forward_eval(seq, len_seq).detach().cpu()

            candidate_ids = [int(x) for x in sample["cans"]]
            model_indices = []
            kept_candidates = []
            for cid in candidate_ids:
                if 0 <= cid < logits.shape[0]:
                    model_indices.append(cid)
                    kept_candidates.append(cid)
                elif 0 <= cid - 1 < logits.shape[0]:
                    model_indices.append(cid - 1)
                    kept_candidates.append(cid)

            if not model_indices:
                raise ValueError(
                    f"No valid AB candidate indices for sample id={sample['id']}; "
                    f"candidate range=[{min(candidate_ids)}, {max(candidate_ids)}], logits_size={logits.shape[0]}"
                )

            candidate_scores = logits[model_indices]
            pred_pos = int(torch.argmax(candidate_scores).item())
            pred_item_id = kept_candidates[pred_pos]
            pred_name = dataset.item_id2name.get(pred_item_id, str(pred_item_id))
            real_name = sample.get("correct_answer", sample.get("item_name", ""))
            rows.append({"id": sample["id"], "generate": pred_name, "real": real_name})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote AB prior file: {out_csv} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--output_prior_file", type=str, required=True)
    parser.add_argument("--output_prior_file_ab", type=str, default="", help="Optional output csv path for AB prior.")
    parser.add_argument("--stage", type=str, default="test")
    parser.add_argument("--cans_num", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--l2_grid", type=str, default="1e-3,1e-4,1e-5,1e-6,1e-7")
    parser.add_argument("--use_text_feat", action="store_true")
    parser.add_argument("--use_image_feat", action="store_true")
    parser.add_argument("--use_meta_feat", action="store_true")
    parser.add_argument("--fusion_type", type=str, default="concat", choices=["concat", "gated"])
    parser.add_argument("--text_feat_path", type=str, default="")
    parser.add_argument("--image_feat_path", type=str, default="")
    parser.add_argument("--meta_feat_path", type=str, default="")
    parser.add_argument("--ab_data_dir", type=str, default="", help="Optional *_ab dataset dir for AB prior csv generation.")
    parser.add_argument("--ab_train_cans_num", type=int, default=20, help="train_cans_num used to generate AB prior csv.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force_rebuild_data_statis", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_prior_file), exist_ok=True)

    statis_df, train_df, val_df, test_df, target_col = build_data_statis(
        args.data_dir,
        force_rebuild=args.force_rebuild_data_statis,
    )
    seq_size = int(statis_df["seq_size"][0])
    item_num = int(statis_df["item_num"][0])
    print(f"saved data_statis.df with seq_size={seq_size}, item_num={item_num}")

    device = torch.device(args.device)
    l2_grid = [float(x) for x in args.l2_grid.split(",") if x.strip()]

    mm_config = {
        "use_text_feat": args.use_text_feat,
        "use_image_feat": args.use_image_feat,
        "use_meta_feat": args.use_meta_feat,
        "fusion_type": args.fusion_type,
        "text_feat_path": args.text_feat_path,
        "image_feat_path": args.image_feat_path,
        "meta_feat_path": args.meta_feat_path,
    }
    if mm_config["use_text_feat"]:
        validate_feature_matrix(mm_config["text_feat_path"], item_num, "text features")
    if mm_config["use_image_feat"]:
        validate_feature_matrix(mm_config["image_feat_path"], item_num, "image features")
    if mm_config["use_meta_feat"]:
        validate_feature_matrix(mm_config["meta_feat_path"], item_num, "meta features")
    model, best_l2, best_val, records, best_history = train_sasrec(
        train_df=train_df,
        val_df=val_df,
        item_num=item_num,
        seq_size=seq_size,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        l2_grid=l2_grid,
        target_col=target_col,
        mm_config=mm_config,
    )
    print(f"best_l2={best_l2} best_val_hit@10={best_val:.4f}")

    torch.save(model, args.output_model_path)
    print(f"saved SASRec model: {args.output_model_path}")

    build_prior_csv(
        model=model,
        data_dir=args.data_dir,
        stage=args.stage,
        cans_num=args.cans_num,
        out_csv=args.output_prior_file,
        device=device,
    )

    ab_out = ""
    if args.ab_data_dir:
        ab_out = args.output_prior_file_ab or args.output_prior_file.replace(".csv", "_ab.csv")
        os.makedirs(os.path.dirname(ab_out), exist_ok=True)
        build_prior_csv_ab(
            model=model,
            ab_data_dir=args.ab_data_dir,
            out_csv=ab_out,
            train_cans_num=args.ab_train_cans_num,
            device=device,
        )
    summary_payload = {
        "data_dir": args.data_dir,
        "item_num": item_num,
        "seq_size": seq_size,
        "embedding_dim": args.embedding_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "l2_grid": l2_grid,
        "best_l2": best_l2,
        "best_val_hit@10": best_val,
        "best_history": best_history,
        "prior_file": args.output_prior_file,
        "prior_file_ab": ab_out,
        "records": records,
        "mm_config": mm_config,
    }
    summary_path = args.output_prior_file.replace(".csv", "_training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"wrote training summary: {summary_path}")

    if ab_out:
        ab_summary_path = ab_out.replace(".csv", "_training_summary.json")
        ab_summary_payload = dict(summary_payload)
        ab_summary_payload["summary_for"] = "ab_prior"
        with open(ab_summary_path, "w", encoding="utf-8") as f:
            json.dump(ab_summary_payload, f, indent=2)
        print(f"wrote AB training summary: {ab_summary_path}")


if __name__ == "__main__":
    main()
