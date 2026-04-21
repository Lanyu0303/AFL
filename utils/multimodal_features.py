import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class MultiModalConfig:
    use_text_feat: bool = False
    use_image_feat: bool = False
    use_meta_feat: bool = False
    fusion_type: str = "concat"  # concat | gated
    text_feat_path: str = ""
    image_feat_path: str = ""
    meta_feat_path: str = ""


class MultiModalItemEncoder(nn.Module):
    """Feature-centric item encoder with backward-compatible ID-only fallback.

    Inputs are item IDs [B, T]. Output is fused item representation [B, T, H].
    Feature files are expected to be aligned by item_id row index.
    """

    def __init__(
        self,
        hidden_size: int,
        item_num: int,
        id_embedding: nn.Embedding,
        config: Optional[MultiModalConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.id_embedding = id_embedding
        self.config = config or MultiModalConfig()

        self.text_features = self._load_feature_matrix(
            self.config.text_feat_path,
            enabled=self.config.use_text_feat,
            feature_name="text",
        )
        self.image_features = self._load_feature_matrix(
            self.config.image_feat_path,
            enabled=self.config.use_image_feat,
            feature_name="image",
        )
        self.meta_features = self._load_feature_matrix(
            self.config.meta_feat_path,
            enabled=self.config.use_meta_feat,
            feature_name="meta",
        )

        active_dims = [hidden_size]
        if self.text_features is not None:
            active_dims.append(self.text_features.shape[1])
        if self.image_features is not None:
            active_dims.append(self.image_features.shape[1])
        if self.meta_features is not None:
            active_dims.append(self.meta_features.shape[1])

        concat_in = int(sum(active_dims))
        self.concat_proj = nn.Linear(concat_in, hidden_size)

        if self.config.fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(concat_in, hidden_size),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

    def _load_feature_matrix(self, path: str, enabled: bool, feature_name: str):
        if not enabled:
            return None
        if not path:
            raise ValueError(f"{feature_name} feature enabled but no path provided")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{feature_name} feature file not found: {path}")

        arr = np.load(path, allow_pickle=True)
        if hasattr(arr, "item") and not isinstance(arr, np.ndarray):
            arr = arr.item()

        if isinstance(arr, dict):
            raise ValueError(
                f"{feature_name} feature path {path} contains dict. "
                "Please align to dense matrix with row index == item_id first."
            )

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{feature_name} matrix must be 2D, got shape={arr.shape}")
        if arr.shape[0] < self.item_num + 1:
            raise ValueError(
                f"{feature_name} matrix rows {arr.shape[0]} < required {self.item_num + 1}"
            )

        tensor = torch.tensor(arr[: self.item_num + 1], dtype=torch.float32)
        self.register_buffer(f"{feature_name}_buffer", tensor, persistent=True)
        return tensor

    def _lookup(self, features: torch.Tensor, item_ids: torch.Tensor):
        flat = item_ids.reshape(-1)
        out = features[flat]
        return out.view(item_ids.shape[0], item_ids.shape[1], -1)

    def forward(self, item_ids: torch.Tensor):
        parts = [self.id_embedding(item_ids)]

        if self.text_features is not None:
            parts.append(self._lookup(self.text_features, item_ids))
        if self.image_features is not None:
            parts.append(self._lookup(self.image_features, item_ids))
        if self.meta_features is not None:
            parts.append(self._lookup(self.meta_features, item_ids))

        if len(parts) == 1:
            return parts[0]

        fused_in = torch.cat(parts, dim=-1)
        concat_proj = self.concat_proj(fused_in)

        if self.gate is None:
            return concat_proj

        gate = self.gate(fused_in)
        return gate * concat_proj + (1.0 - gate) * parts[0]


def load_item_meta(item_meta_path: str) -> pd.DataFrame:
    if not item_meta_path:
        raise ValueError("item_meta_path is required")
    if not os.path.exists(item_meta_path):
        raise FileNotFoundError(f"item_meta file not found: {item_meta_path}")
    return pd.read_pickle(item_meta_path)


def summarize_feature_coverage(item_meta_path: str, out_json: str) -> Dict[str, float]:
    df = load_item_meta(item_meta_path)
    total = max(len(df), 1)
    stats = {
        "num_items": int(len(df)),
        "title_coverage": float(df.get("title", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum() / total),
        "description_coverage": float(df.get("description", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum() / total),
        "brand_coverage": float(df.get("brand", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum() / total),
        "category_coverage": float(df.get("category", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum() / total),
        "image_url_coverage": float(df.get("image_url", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum() / total),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats
