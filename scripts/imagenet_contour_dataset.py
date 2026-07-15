"""ImageNet 轮廓在线提取 Dataset（流式，不落盘、不全量预读）。

imagenet 在 `datasets/imagenet`（junction → D:\\data\\imagenet）是 HuggingFace
WebDataset 的 Parquet 分片（train 294 / val 14 / test 28，每 shard ~4358 行，
每 shard 5 个 row group × ~871 行）。schema: image (dict{bytes: JPEG, path})
+ label (int)。

训练时按需从 Parquet 读图、实时提取边缘（canny/sobel/hed），不写盘——省去百万张
PNG 的磁盘开销，且可随时换提取方法。

流式策略（避免全量 OOM）：只把索引（~128 万条 (file,row_group,row) 三元组，
~40MB）放内存，__getitem__ 按需读单行；row group 的 image 列用 LRU 缓存
（内存上界 = cap × ~101MB），其余靠 OS page cache。

worker-safe：__init__ 只保留可 pickle 字段（文件路径 + 索引 + 配置），
ParquetFile / extractor / LRU 在每个 worker 首次 __getitem__ 惰性重建
（Windows spawn 下 ParquetFile、cv2.dnn Net 等不可 pickle）。

全量语义：``max_images<=0`` = 不采样、用全部行；``shards<=0`` = 全部 shard。
默认即全量（训练集所有图都参与）。
"""
from __future__ import annotations

import io
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark.video.extractors import build_extractor, list_extractors  # noqa: E402
from benchmark.learned import _img_to_tensor  # noqa: E402

DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(REPO / "datasets")))

_IMAGE_EXTS = (".png", ".tiff", ".tif", ".jpg", ".jpeg")

# imagenet parquet 文件名前缀：split 名 → 文件前缀（val 对应 validation-*.parquet）。
_SPLIT_PREFIX = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


class _RowGroupLRU:
    """row group 的 image 列 LRU 缓存（按组数封顶，内存上界 = cap × ~101MB）。"""

    def __init__(self, cap: int = 128):
        self.cap = max(1, cap)
        self._d: "OrderedDict[tuple[int, int], object]" = OrderedDict()

    def get(self, key, loader):
        col = self._d.get(key)
        if col is not None:
            self._d.move_to_end(key)
            return col
        col = loader(key)
        self._d[key] = col
        self._d.move_to_end(key)
        while len(self._d) > self.cap:
            self._d.popitem(last=False)
        return col


class ImageNetContourDataset(Dataset):
    """Parquet → 流式按需读单行 → 实时边缘提取 → [0,1] 3 通道张量（不落地、不全量预读）。"""

    def __init__(
        self,
        split: str = "train",
        method: str = "canny",
        max_images: int = 0,
        size: int = 128,
        shards: int = 0,
        rg_cache_cap: int = 128,
    ):
        if method not in list_extractors():
            raise KeyError(f"Unknown extractor '{method}'. Available: {list_extractors()}")
        # 仅存可 pickle 字段（路径 + 配置）；ParquetFile/extractor/LRU 在 worker 惰性建
        self.method = method
        self.size = size
        self.rg_cache_cap = max(1, rg_cache_cap)

        data_dir = DATASETS_DIR / "imagenet" / "data"
        prefix = _SPLIT_PREFIX.get(split, split)
        files = sorted(data_dir.glob(f"{prefix}-*.parquet"))
        if not files:
            raise RuntimeError(f"无 imagenet parquet: {data_dir}/{prefix}-*.parquet（dataset_id=imagenet-{split}）")
        # shards<=0 → 全部 shard；否则取前 N 个
        self._file_paths = [str(f) for f in (files if not shards or shards <= 0 else files[: max(1, int(shards))])]

        # 全索引：跨所有选中 shard 的每个 row group 的每一行 (file_idx, row_group, row_in_group)
        # 只读 metadata（不持有 ParquetFile 对象），用于算索引与 __len__
        import pyarrow.parquet as pq
        full_index: list[tuple[int, int, int]] = []
        for fi, fp in enumerate(self._file_paths):
            md = pq.ParquetFile(fp).metadata
            for rg in range(md.num_row_groups):
                n = md.row_group(rg).num_rows
                for r in range(n):
                    full_index.append((fi, rg, r))

        # max_images<=0 → 不采样（全量）；否则等间隔确定性采样
        if max_images and max_images > 0 and len(full_index) > max_images:
            step = len(full_index) / max_images
            self._index = [full_index[int(i * step)] for i in range(max_images)]
        else:
            self._index = full_index
        self._n = len(self._index)
        if self._n == 0:
            raise RuntimeError(f"imagenet 索引为空（split={split}, shards={shards}）")

        # 惰性字段（worker 进程内首次 __getitem__ 时建）
        self._pq = None
        self._extractor = None
        self._rg_cache = None

    def _lazy_init(self) -> None:
        if self._pq is not None:
            return
        import pyarrow.parquet as pq
        self._pq = [pq.ParquetFile(fp) for fp in self._file_paths]
        self._extractor = build_extractor(self.method)
        self._rg_cache = _RowGroupLRU(cap=self.rg_cache_cap)

    def __len__(self) -> int:
        return self._n

    def _load_col(self, fi: int, rg: int):
        return self._pq[fi].read_row_group(rg, columns=["image"]).column("image")

    def __getitem__(self, i: int) -> torch.Tensor:
        if i < 0:
            i %= self._n
        self._lazy_init()
        fi, rg, r = self._index[i]
        col = self._rg_cache.get((fi, rg), lambda k: self._load_col(*k))
        cell = col[r].as_py()
        b = cell["bytes"] if isinstance(cell, dict) else cell
        # color BGR (extractors expect cv2 BGR; hed/pidinet/yoloe26 trained on
        # color BSDS/COCO — pass color, let each extractor decide gray-vs-color)
        img = Image.open(io.BytesIO(b)).convert("RGB")
        arr = np.ascontiguousarray(np.array(img, dtype=np.uint8)[..., ::-1])
        edges = self._extractor.extract(arr)  # uint8 HxW
        if edges.dtype != np.uint8:
            edges = edges.astype(np.uint8)
        t, _ = _img_to_tensor(edges)  # (1,3,H,W) float [0,1]（min-max + 复制3通道）
        if t.shape[2] != self.size or t.shape[3] != self.size:
            t = F.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return torch.clamp(t.squeeze(0), 0.0, 1.0)  # (3,H,W)


def split_from_dataset_id(dataset_id: str) -> str:
    """imagenet-train / imagenet-val / imagenet-test -> split name（id 用连字符单段）。"""
    parts = dataset_id.split("-", 1)
    return parts[1] if len(parts) == 2 else "train"


class ContourPNGDataset(Dataset):
    """读预提取的 frame_*.png 边缘图（extract_imagenet_contour.py 产物），min-max 归一化
    到 [0,1] 3 通道。训练快路径：无 JPEG 解码/无在线提边缘，PNG 多 worker 读且 OS 缓存友好。"""

    def __init__(self, frames_dir: str, size: int = 128, max_images: int = 0):
        self.size = size
        root = Path(frames_dir)
        files = sorted(
            p for p in root.rglob("*")
            if p.name.lower().startswith("frame_") and p.suffix.lower() in _IMAGE_EXTS
        )
        if max_images and max_images > 0:
            files = files[: max_images]
        self._files = [str(p) for p in files]
        if not self._files:
            raise RuntimeError(f"无轮廓 PNG: {root}（先跑 scripts/extract_imagenet_contour.py）")
        self._n = len(self._files)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> torch.Tensor:
        if i < 0:
            i %= self._n
        arr = np.array(Image.open(self._files[i]))
        t, _ = _img_to_tensor(arr)  # (1,3,H,W) float [0,1]（min-max + 复制3通道）
        if t.shape[2] != self.size or t.shape[3] != self.size:
            t = F.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return torch.clamp(t.squeeze(0), 0.0, 1.0)  # (3,H,W)


def preextracted_contour_dir(split: str, method: str) -> Path:
    """预提取 PNG 目录：datasets/contour/imagenet_<split>_<method>/。"""
    return DATASETS_DIR / "contour" / f"imagenet_{split}_{method}"
