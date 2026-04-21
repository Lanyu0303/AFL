import os
import random

import pandas as pd
import torch.utils.data as data


class AmazonDataset(data.Dataset):
    def __init__(self, data_dir="./data/amazon", stage="test", cans_num=20, sep=", ", no_augment=True, use_item_meta=False, item_meta_path="", max_meta_len=96):
        self.data_dir = data_dir
        self.cans_num = cans_num
        self.stage = stage
        self.sep = sep
        self.aug = (stage == "train") and not no_augment
        self.padding_item_id = None
        self.use_item_meta = use_item_meta
        self.item_meta_path = item_meta_path
        self.max_meta_len = int(max_meta_len)
        self.item_meta = {}
        self.check_files()


    def _safe_trunc(self, text):
        text = str(text or "").strip()
        if len(text) <= self.max_meta_len:
            return text
        return text[: self.max_meta_len - 3] + "..."

    def _semantic_text(self, item_id):
        title = self.item_id2name.get(int(item_id), str(item_id))
        if not self.use_item_meta:
            return title
        row = self.item_meta.get(int(item_id), {})
        brand = self._safe_trunc(row.get("brand", ""))
        category = self._safe_trunc(row.get("category", ""))
        desc = self._safe_trunc(row.get("description", ""))
        parts = [f"title={title}"]
        if brand:
            parts.append(f"brand={brand}")
        if category:
            parts.append(f"cat={category}")
        if desc:
            parts.append(f"desc={desc}")
        return " | ".join(parts)

    def _load_item_meta(self):
        if not self.use_item_meta:
            return
        meta_path = self.item_meta_path or os.path.join(self.data_dir, "item_meta.df")
        if not os.path.exists(meta_path):
            return
        df = pd.read_pickle(meta_path)
        if "item_id" not in df.columns:
            return
        for _, row in df.iterrows():
            iid = int(row["item_id"])
            self.item_meta[iid] = {
                "brand": str(row.get("brand", "") or ""),
                "category": str(row.get("category", "") or ""),
                "description": str(row.get("description", "") or ""),
            }

    def __len__(self):
        return len(self.session_data["seq"])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp["seq_unpad"], temp["next"])
        cans_name = [self.item_id2name[can] for can in candidates]
        seq_semantic = [self._semantic_text(x) for x in temp["seq_unpad"]]
        cans_semantic = [self._semantic_text(x) for x in candidates]
        return {
            "id": i,
            "seq": temp["seq"],
            "seq_name": temp["seq_title"],
            "len_seq": temp["len_seq"],
            "seq_str": self.sep.join(temp["seq_title"]),
            "seq_semantic_str": self.sep.join(seq_semantic),
            "cans": candidates,
            "cans_name": cans_name,
            "cans_str": self.sep.join(cans_name),
            "cans_semantic_str": self.sep.join(cans_semantic),
            "len_cans": self.cans_num,
            "item_id": temp["next"],
            "item_name": temp["next_item_name"],
            "correct_answer": temp["next_item_name"],
        }

    def negative_sampling(self, seq_unpad, next_item):
        canset = [i for i in self.item_id2name.keys() if i not in seq_unpad and i != next_item]
        if len(canset) < self.cans_num - 1:
            raise ValueError(
                f"[AmazonDataset] not enough negative items: need {self.cans_num - 1}, got {len(canset)}"
            )
        candidates = random.sample(canset, self.cans_num - 1) + [next_item]
        random.shuffle(candidates)
        return candidates

    def check_files(self):
        self.item_id2name = self.get_item_id2name()
        self.padding_item_id = max(self.item_id2name.keys()) + 1
        self._load_item_meta()
        if self.stage == "train":
            filename = "train_data.df"
        elif self.stage == "val":
            filename = "Val_data.df"
        elif self.stage == "test":
            filename = "Test_data.df"
        else:
            raise ValueError(f"Unsupported stage: {self.stage}")
        data_path = os.path.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)

    def get_item_id2name(self):
        item_id2name = {}
        item_path = os.path.join(self.data_dir, "id2name.txt")
        if not os.path.exists(item_path):
            fallback_path = os.path.join(self.data_dir, "id2name_AB_llara.txt")
            if os.path.exists(fallback_path):
                item_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Missing id2name mapping in {self.data_dir}. "
                    f"Expected one of: id2name.txt, id2name_AB_llara.txt"
                )
        with open(item_path, "r", encoding="utf-8") as f:
            for l in f.readlines():
                ll = l.strip("\n").split("::", maxsplit=1)
                item_id2name[int(ll[0])] = ll[1].strip()
        return item_id2name

    def session_data4frame(self, datapath, item_id2name):
        session_data = pd.read_pickle(datapath)
        session_data = session_data[session_data["len_seq"] >= 3]

        def remove_padding(xx):
            x = xx[:]
            for _ in range(50):
                try:
                    x.remove(self.padding_item_id)
                except ValueError:
                    break
            return x

        session_data["seq_unpad"] = session_data["seq"].apply(remove_padding)
        session_data["seq_title"] = session_data["seq_unpad"].apply(
            lambda x: [item_id2name[x_i] for x_i in x]
        )
        session_data["next_item_name"] = session_data["next"].apply(lambda x: item_id2name[x])
        return session_data
