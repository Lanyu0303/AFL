import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset.amazon_dataset import AmazonDataset
from dataset.lastfmAB_dataset import LastfmABDataset

try:
    import tensorflow as tf
    from afl.model_org import Model
    TF_AVAILABLE = True
except ModuleNotFoundError:
    tf = None
    Model = None
    TF_AVAILABLE = False

from afl.prepare_amazon_artifacts import (
    build_data_statis as build_data_statis_torch,
    build_prior_csv,
    train_sasrec,
)

if TF_AVAILABLE:
    tf.compat.v1.disable_eager_execution()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.compat.v1.set_random_seed(seed)


def _require_files(base_dir, required_names, label):
    missing = [name for name in required_names if not os.path.exists(os.path.join(base_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"{label} missing files in {base_dir}: {missing}. "
            f"Found files: {sorted(os.listdir(base_dir)) if os.path.isdir(base_dir) else 'directory not found'}"
        )


def load_from_preprocessed(data_dir):
    train_df = pd.read_pickle(os.path.join(data_dir, "train_data.df"))
    val_df = pd.read_pickle(os.path.join(data_dir, "Val_data.df"))
    test_df = pd.read_pickle(os.path.join(data_dir, "Test_data.df"))
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    target_col = "next" if "next" in all_df.columns else "item_id"
    item_num = int(max(all_df["seq"].apply(max).max(), all_df[target_col].max() + 1))
    seq_size = int(max(all_df["len_seq"].max(), 10))
    return train_df, val_df, test_df, target_col, item_num, seq_size


def load_from_txt(txt_path, maxlen=50):
    user_hist = {}
    item_max = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, i = int(parts[0]), int(parts[1])
            user_hist.setdefault(u, []).append(i)
            item_max = max(item_max, i)

    rows_train, rows_val, rows_test = [], [], []
    for u, items in user_hist.items():
        if len(items) < 3:
            continue
        for idx in range(1, len(items) - 2):
            seq = items[max(0, idx - maxlen):idx]
            rows_train.append((u, seq + [0] * (maxlen - len(seq)), len(seq), items[idx]))
        val_idx = len(items) - 2
        test_idx = len(items) - 1
        vseq = items[max(0, val_idx - maxlen):val_idx]
        tseq = items[max(0, test_idx - maxlen):test_idx]
        rows_val.append((u, vseq + [0] * (maxlen - len(vseq)), len(vseq), items[val_idx]))
        rows_test.append((u, tseq + [0] * (maxlen - len(tseq)), len(tseq), items[test_idx]))

    cols = ["user_id", "seq", "len_seq", "next"]
    return pd.DataFrame(rows_train, columns=cols), pd.DataFrame(rows_val, columns=cols), pd.DataFrame(rows_test, columns=cols), "next", item_max + 1, maxlen


def make_batch(df, target_col, item_num, batch_size):
    idxs = np.random.choice(len(df), size=min(batch_size, len(df)), replace=False)
    users, seqs, poss, negs = [], [], [], []
    for i in idxs:
        row = df.iloc[int(i)]
        seq = list(row["seq"])
        l = int(row["len_seq"])
        tgt = int(row[target_col])
        pos = [0] * len(seq)
        neg = [0] * len(seq)
        if l > 0:
            pos[l - 1] = tgt
            seen = set(seq[:l]) | {0, tgt}
            n = random.randint(1, item_num - 1)
            tries = 0
            while n in seen and tries < 50:
                n = random.randint(1, item_num - 1)
                tries += 1
            neg[l - 1] = n
        users.append(int(row.get("user_id", 0)))
        seqs.append(seq)
        poss.append(pos)
        negs.append(neg)
    return np.array(users), np.array(seqs), np.array(poss), np.array(negs)


def evaluate_hit10(sess, model, df, target_col, item_num):
    hit10 = 0
    hit1 = 0
    total = 0
    for _, row in df.iterrows():
        seq = np.array([row["seq"]])
        target = int(row[target_col])
        seen = set([x for x in row["seq"] if int(x) > 0]) | {target}
        negs = []
        while len(negs) < 100:
            x = random.randint(1, item_num - 1)
            if x not in seen:
                negs.append(x)
        candidates = [target] + negs
        logits = model.predict(sess, [int(row.get("user_id", 0))], seq, candidates)[0]
        order = np.argsort(-logits)
        top10 = [candidates[i] for i in order[:10]]
        top1 = candidates[int(order[0])]
        hit10 += int(target in top10)
        hit1 += int(top1 == target)
        total += 1
    if total == 0:
        return 0.0, 0.0
    return hit1 / total, hit10 / total


def build_prior_csv_org(sess, model, data_dir, stage, cans_num, out_csv):
    dataset = AmazonDataset(data_dir=data_dir, stage=stage, cans_num=cans_num, sep=", ", no_augment=True)
    rows = []
    for sample in dataset:
        seq = np.array([sample["seq"]])
        candidates = [int(x) for x in sample["cans"]]
        logits = model.predict(sess, [0], seq, candidates)[0]
        pred = candidates[int(np.argmax(logits))]
        rows.append({"id": sample["id"], "generate": dataset.item_id2name.get(pred, str(pred)), "real": sample.get("correct_answer", "")})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote prior file: {out_csv} ({len(rows)} rows)")


def build_prior_csv_ab_org(sess, model, ab_data_dir, out_csv, train_cans_num):
    dataset = LastfmABDataset(data_dir=ab_data_dir, stage="test", train_cans_num=train_cans_num, eval_cans_num=train_cans_num)
    rows = []
    for sample in dataset:
        seq = np.array([sample["seq"]])
        candidates = [int(x) for x in sample["cans"]]
        logits = model.predict(sess, [int(sample.get("user_id", 0))], seq, candidates)[0]
        pred = candidates[int(np.argmax(logits))]
        rows.append({"id": sample["id"], "generate": dataset.item_id2name.get(pred, str(pred)), "real": sample.get("correct_answer", "")})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote AB prior file: {out_csv} ({len(rows)} rows)")


def build_prior_csv_ab_org_torch(model, ab_data_dir, out_csv, train_cans_num, device):
    dataset = LastfmABDataset(data_dir=ab_data_dir, stage="test", train_cans_num=train_cans_num, eval_cans_num=train_cans_num)
    rows = []
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            seq = torch.tensor([sample["seq"]], dtype=torch.long, device=device)
            len_seq = np.array([int(sample["len_seq"])])
            logits = model.forward_eval(seq, len_seq).detach().cpu()
            candidates = [int(x) for x in sample["cans"]]
            scores = logits[candidates]
            pred = candidates[int(torch.argmax(scores).item())]
            rows.append(
                {
                    "id": sample["id"],
                    "generate": dataset.item_id2name.get(pred, str(pred)),
                    "real": sample.get("correct_answer", sample.get("item_name", "")),
                }
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote AB prior file: {out_csv} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", help="Preprocessed Amazon dir with train/Val/Test .df")
    parser.add_argument("--txt_path", type=str, default="", help="Original SASRec txt (user item per line)")
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--output_prior_file", type=str, required=True)
    parser.add_argument("--output_prior_file_ab", type=str, default="")
    parser.add_argument("--ab_data_dir", type=str, default="")
    parser.add_argument("--stage", type=str, default="test")
    parser.add_argument("--cans_num", type=int, default=20)
    parser.add_argument("--ab_train_cans_num", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--l2_emb", type=float, default=0.0)
    parser.add_argument("--maxlen", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data_dir and not args.txt_path:
        raise ValueError("Set either --data_dir or --txt_path")
    if args.data_dir:
        _require_files(args.data_dir, ["train_data.df", "Val_data.df", "Test_data.df"], "data_dir")
    if args.ab_data_dir:
        _require_files(args.ab_data_dir, ["test_llara.pkl", "id2name_AB_llara.txt"], "ab_data_dir")
    if args.txt_path and not os.path.exists(args.txt_path):
        raise FileNotFoundError(f"txt_path not found: {args.txt_path}")

    if not TF_AVAILABLE:
        print("[WARN] TensorFlow is not available; falling back to PyTorch SASRec backend in org script.")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        if os.path.dirname(args.output_model_path):
            os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
        if os.path.dirname(args.output_prior_file):
            os.makedirs(os.path.dirname(args.output_prior_file), exist_ok=True)

        if args.data_dir:
            statis_df, train_df, val_df, test_df, target_col = build_data_statis_torch(args.data_dir, force_rebuild=False)
            item_num = int(statis_df["item_num"][0])
            seq_size = int(statis_df["seq_size"][0])
        else:
            train_df, val_df, test_df, target_col, item_num, seq_size = load_from_txt(args.txt_path, maxlen=args.maxlen)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, best_l2, best_val, records, best_history = train_sasrec(
            train_df=train_df,
            val_df=val_df,
            item_num=item_num,
            seq_size=seq_size,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            embedding_dim=args.hidden_units,
            l2_grid=[1e-3, 1e-4, 1e-5],
            target_col=target_col,
        )
        torch.save(model, args.output_model_path)
        if args.data_dir:
            build_prior_csv(
                model=model,
                data_dir=args.data_dir,
                stage=args.stage,
                cans_num=args.cans_num,
                out_csv=args.output_prior_file,
                device=device,
            )
        else:
            # txt-mode fallback: no id2name/candidate file, so export item-id predictions.
            rows = []
            model.eval()
            with torch.no_grad():
                for i, row in test_df.reset_index(drop=True).iterrows():
                    seq = torch.tensor([row["seq"]], dtype=torch.long, device=device)
                    len_seq = np.array([int(row["len_seq"])])
                    logits = model.forward_eval(seq, len_seq).detach().cpu()
                    pred_item = int(torch.argmax(logits).item())
                    rows.append({"id": i, "generate": str(pred_item), "real": str(int(row[target_col]))})
            pd.DataFrame(rows).to_csv(args.output_prior_file, index=False)
            print(f"wrote txt-mode prior file: {args.output_prior_file} ({len(rows)} rows)")
        if args.ab_data_dir:
            ab_out = args.output_prior_file_ab or args.output_prior_file.replace(".csv", "_ab.csv")
            if os.path.dirname(ab_out):
                os.makedirs(os.path.dirname(ab_out), exist_ok=True)
            build_prior_csv_ab_org_torch(
                model=model,
                ab_data_dir=args.ab_data_dir,
                out_csv=ab_out,
                train_cans_num=args.ab_train_cans_num,
                device=device,
            )

        summary_path = args.output_prior_file.replace(".csv", "_training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "org_script_fallback_pytorch",
                    "tensorflow_available": False,
                    "data_dir": args.data_dir,
                    "item_num": item_num,
                    "seq_size": seq_size,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "best_l2": best_l2,
                    "best_val_hit@10": best_val,
                    "best_history": best_history,
                    "records": records,
                    "prior_file": args.output_prior_file,
                    "prior_file_ab": (args.output_prior_file_ab or args.output_prior_file.replace(".csv", "_ab.csv"))
                    if args.ab_data_dir
                    else "",
                },
                f,
                indent=2,
            )
        print(f"wrote training summary: {summary_path}")
        if args.ab_data_dir:
            ab_out = args.output_prior_file_ab or args.output_prior_file.replace(".csv", "_ab.csv")
            ab_summary_path = ab_out.replace(".csv", "_training_summary.json")
            with open(ab_summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary_for": "ab_prior",
                        "mode": "org_script_fallback_pytorch",
                        "tensorflow_available": False,
                        "data_dir": args.data_dir,
                        "txt_path": args.txt_path,
                        "item_num": item_num,
                        "seq_size": seq_size,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "best_l2": best_l2,
                        "best_val_hit@10": best_val,
                        "best_history": best_history,
                        "records": records,
                        "prior_file": args.output_prior_file,
                        "prior_file_ab": ab_out,
                    },
                    f,
                    indent=2,
                )
            print(f"wrote AB training summary: {ab_summary_path}")
        return

    set_seed(args.seed)
    if os.path.dirname(args.output_model_path):
        os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    if os.path.dirname(args.output_prior_file):
        os.makedirs(os.path.dirname(args.output_prior_file), exist_ok=True)

    if args.data_dir:
        train_df, val_df, test_df, target_col, item_num, seq_size = load_from_preprocessed(args.data_dir)
    else:
        train_df, val_df, test_df, target_col, item_num, seq_size = load_from_txt(args.txt_path, maxlen=args.maxlen)

    model_args = SimpleNamespace(
        maxlen=seq_size,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        l2_emb=args.l2_emb,
        lr=args.lr,
    )

    tf.compat.v1.reset_default_graph()
    model = Model(usernum=1, itemnum=item_num, args=model_args)
    saver = tf.compat.v1.train.Saver()

    history = []
    best_hit10 = -1.0
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        steps_per_epoch = max(1, len(train_df) // max(1, args.batch_size))
        for ep in range(1, args.epochs + 1):
            losses = []
            for _ in range(steps_per_epoch):
                u, seq, pos, neg = make_batch(train_df, target_col, item_num, args.batch_size)
                loss, _ = sess.run(
                    [model.loss, model.train_op],
                    {
                        model.u: u,
                        model.input_seq: seq,
                        model.pos: pos,
                        model.neg: neg,
                        model.is_training: True,
                    },
                )
                losses.append(float(loss))
            val_hit1, val_hit10 = evaluate_hit10(sess, model, val_df, target_col, item_num)
            avg_loss = float(np.mean(losses)) if losses else 0.0
            history.append({"epoch": ep, "train_loss": avg_loss, "val_hit@1": val_hit1, "val_hit@10": val_hit10})
            print(f"[org][epoch {ep}] loss={avg_loss:.4f} val_hit@1={val_hit1:.4f} val_hit@10={val_hit10:.4f}")
            if val_hit10 > best_hit10:
                best_hit10 = val_hit10
                saver.save(sess, args.output_model_path)

        print(f"saved SASRec(org) checkpoint: {args.output_model_path}")

        if args.data_dir:
            build_prior_csv_org(sess, model, args.data_dir, args.stage, args.cans_num, args.output_prior_file)
        else:
            rows = []
            all_items = list(range(1, item_num))
            for i, row in test_df.reset_index(drop=True).iterrows():
                seq = np.array([row["seq"]])
                logits = model.predict(sess, [int(row.get("user_id", 0))], seq, all_items)[0]
                pred = all_items[int(np.argmax(logits))]
                rows.append({"id": i, "generate": str(pred), "real": str(int(row[target_col]))})
            pd.DataFrame(rows).to_csv(args.output_prior_file, index=False)
            print(f"wrote txt-mode prior file: {args.output_prior_file} ({len(rows)} rows)")

        ab_out = ""
        if args.ab_data_dir:
            ab_out = args.output_prior_file_ab or args.output_prior_file.replace(".csv", "_ab.csv")
            if os.path.dirname(ab_out):
                os.makedirs(os.path.dirname(ab_out), exist_ok=True)
            build_prior_csv_ab_org(sess, model, args.ab_data_dir, ab_out, args.ab_train_cans_num)

    summary_path = args.output_prior_file.replace(".csv", "_training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "org_sasrec_tf1",
                "data_dir": args.data_dir,
                "txt_path": args.txt_path,
                "item_num": item_num,
                "seq_size": seq_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "best_val_hit@10": best_hit10,
                "history": history,
                "prior_file": args.output_prior_file,
                "prior_file_ab": (args.output_prior_file_ab or args.output_prior_file.replace(".csv", "_ab.csv")) if args.ab_data_dir else "",
            },
            f,
            indent=2,
        )
    print(f"wrote training summary: {summary_path}")

    if args.ab_data_dir:
        ab_out = args.output_prior_file_ab or args.output_prior_file.replace(".csv", "_ab.csv")
        ab_summary_path = ab_out.replace(".csv", "_training_summary.json")
        with open(ab_summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary_for": "ab_prior",
                    "mode": "org_sasrec_tf1",
                    "data_dir": args.data_dir,
                    "txt_path": args.txt_path,
                    "item_num": item_num,
                    "seq_size": seq_size,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "best_val_hit@10": best_hit10,
                    "history": history,
                    "prior_file": args.output_prior_file,
                    "prior_file_ab": ab_out,
                },
                f,
                indent=2,
            )
        print(f"wrote AB training summary: {ab_summary_path}")


if __name__ == "__main__":
    main()
