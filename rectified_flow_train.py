"""Rectified-Flow training + inference on top of MSFlow (fusion-latent).

This script implements the workflow you described:

Train:
  1) Take a normal image x (training data are normal-only in MVTec/VisA).
  2) Create a pseudo-anomaly image x' using CutPaste (3-way): either
     - CutPaste (patch)
     - CutPaste-Scar (thin rotated patch)
  3) Run MSFlow (parallel+fusion) to obtain fusion latents z(x) and z(x').
  4) Train a rectified-flow velocity field v_theta per scale that transports
     z(x') -> z(x).

Test:
  1) For each test image, get z and z_rect (transport z -> "more normal").
  2) Compute logp maps: logp = -0.5 * mean(z**2, channel).
  3) diff = logp_rect - logp.
  4) Run post_process() twice: on outputs_list and outputs_list_diff.
  5) final anomaly map = anomaly_score_map_add + anomaly_score_map_add_diff.

Notes
-----
* We freeze MSFlow (extractor + flows). We only train the rectified-flow nets.
* The rectified-flow objective used here is the standard linear path loss:
    x_t = (1-t) x0 + t x1, target v* = x1 - x0, minimize ||v_theta(x_t,t) - v*||^2.
  With this objective, Euler integration from t=0..1 starting at x0 yields x1.
* Velocity model: small fully-convolutional residual network per scale.

Run examples
------------
Train rectified flow (requires a trained MSFlow checkpoint):

  python rectified_flow_train.py \
    --dataset mvtec --class-name bottle --msflow-ckpt /path/to/msflow_ckpt.pt \
    --rf-epochs 10 --rf-lr 1e-4 --batch-size 8

Test with rectified scoring:

  python rectified_flow_train.py \
    --dataset mvtec --class-name bottle --mode test \
    --msflow-ckpt /path/to/msflow_ckpt.pt --rf-ckpt /path/to/rf_ckpt.pt
"""

from __future__ import annotations

import os
import sys

# This repository has a local copy.py helper. Temporarily hide the script
# directory while importing stdlib/third-party libraries that need stdlib copy.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REMOVED_SYS_PATH_ENTRIES = []
for _path_value in ("", _SCRIPT_DIR):
    while _path_value in sys.path:
        _idx = sys.path.index(_path_value)
        _REMOVED_SYS_PATH_ENTRIES.append((_idx, _path_value))
        sys.path.pop(_idx)

import math
import time
import datetime
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

for _idx, _path_value in sorted(_REMOVED_SYS_PATH_ENTRIES):
    sys.path.insert(min(_idx, len(sys.path)), _path_value)
del _REMOVED_SYS_PATH_ENTRIES, _SCRIPT_DIR

from datasets import MVTecDataset, VisADataset, POSCODataset
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from post_process import post_process
from utils import positionalencoding2d, load_weights, save_weights, t2np
from evaluations import eval_det_loc
from utils import Score_Observer
from noise import pnoise2
import numpy as np


# ------------------------- CutPaste (3-way) -------------------------

def _denorm(x: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """(B,3,H,W) or (3,H,W) normalized -> [0,1] approx."""
    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    if x.dim() == 4:
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)
    return x * std_t + mean_t


def _renorm(x01: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """[0,1] tensor -> normalized."""
    mean_t = torch.as_tensor(mean, device=x01.device, dtype=x01.dtype).view(-1, 1, 1)
    std_t = torch.as_tensor(std, device=x01.device, dtype=x01.dtype).view(-1, 1, 1)
    if x01.dim() == 4:
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)
    return (x01 - mean_t) / std_t


def _rand_uniform(a: float, b: float) -> float:
    return float(torch.empty(1).uniform_(a, b).item())


class CutPaste3Way:
    """CutPaste (3-way) augmentation producing pseudo anomalies.

    We sample one of two variants per image:
      - CutPaste (patch): rectangular patch cut & paste.
      - CutPaste-Scar: long-thin patch cut & paste with random rotation.

    Hyperparameters follow CutPaste appendix (approx):
      - patch area ratio in [0.02, 0.15]
      - aspect ratio from (0.3,1) U (1,3.3)
      - scar: width in [2,16], length in [10,25], rotation in [-45,45]
      - color jitter intensity <= 0.1 on the patch before pasting
    """

    def __init__(
        self,
        img_mean: List[float],
        img_std: List[float],
        p_scar: float = 0.5,
        area_ratio: Tuple[float, float] = (0.02, 0.15),
        aspect_ratio_low_high: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.3, 1.0), (1.0, 3.3)),
        scar_w: Tuple[int, int] = (2, 16),
        scar_l: Tuple[int, int] = (10, 25),
        scar_deg: Tuple[float, float] = (-45.0, 45.0),
        jitter: float = 0.1,
    ):
        self.img_mean = img_mean
        self.img_std = img_std
        self.p_scar = p_scar
        self.area_ratio = area_ratio
        self.aspect_ratio_low_high = aspect_ratio_low_high
        self.scar_w = scar_w
        self.scar_l = scar_l
        self.scar_deg = scar_deg
        self.jitter = jitter

    @staticmethod
    def _color_jitter_patch(patch: torch.Tensor, jitter: float) -> torch.Tensor:
        # patch: (3,h,w) in [0,1]
        if jitter <= 0:
            return patch
        # torchvision functional jitter helpers expect PIL or Tensor in [0,1]
        # Apply brightness/contrast/saturation/hue with random order.
        ops = [
            lambda x: TF.adjust_brightness(x, 1.0 + _rand_uniform(-jitter, jitter)),
            lambda x: TF.adjust_contrast(x, 1.0 + _rand_uniform(-jitter, jitter)),
            lambda x: TF.adjust_saturation(x, 1.0 + _rand_uniform(-jitter, jitter)),
            # Hue in [-0.5,0.5] in TF; we keep small.
            lambda x: TF.adjust_hue(x, _rand_uniform(-jitter, jitter)),
        ]
        # shuffle ops
        perm = torch.randperm(len(ops)).tolist()
        out = patch
        for i in perm:
            out = ops[i](out)
        return out.clamp(0.0, 1.0)

    @staticmethod
    def _sample_rect_patch(H: int, W: int, area_ratio: Tuple[float, float], aspect_choices: Tuple[Tuple[float, float], Tuple[float, float]]):
        area = _rand_uniform(area_ratio[0], area_ratio[1]) * (H * W)
        if torch.rand(1).item() < 0.5:
            ar = _rand_uniform(aspect_choices[0][0], aspect_choices[0][1])
        else:
            ar = _rand_uniform(aspect_choices[1][0], aspect_choices[1][1])
        h = int(round(math.sqrt(area / ar)))
        w = int(round(h * ar))
        h = max(1, min(h, H - 1))
        w = max(1, min(w, W - 1))
        y0 = int(torch.randint(0, H - h + 1, (1,)).item())
        x0 = int(torch.randint(0, W - w + 1, (1,)).item())
        return y0, x0, h, w

    def _cutpaste_patch(self, x01: torch.Tensor) -> torch.Tensor:
        # x01: (3,H,W) in [0,1]
        _, H, W = x01.shape
        y0, x0, h, w = self._sample_rect_patch(H, W, self.area_ratio, self.aspect_ratio_low_high)
        patch = x01[:, y0 : y0 + h, x0 : x0 + w].clone()
        patch = self._color_jitter_patch(patch, self.jitter)
        # paste location
        py = int(torch.randint(0, H - h + 1, (1,)).item())
        px = int(torch.randint(0, W - w + 1, (1,)).item())
        out = x01.clone()
        out[:, py : py + h, px : px + w] = patch
        return out

    def _cutpaste_scar(self, x01: torch.Tensor) -> torch.Tensor:
        _, H, W = x01.shape
        w = int(torch.randint(self.scar_w[0], self.scar_w[1] + 1, (1,)).item())
        h = int(torch.randint(self.scar_l[0], self.scar_l[1] + 1, (1,)).item())
        w = max(1, min(w, W - 1))
        h = max(1, min(h, H - 1))
        y0 = int(torch.randint(0, H - h + 1, (1,)).item())
        x0 = int(torch.randint(0, W - w + 1, (1,)).item())
        patch = x01[:, y0 : y0 + h, x0 : x0 + w].clone()
        patch = self._color_jitter_patch(patch, self.jitter)
        deg = _rand_uniform(self.scar_deg[0], self.scar_deg[1])
        patch = TF.rotate(patch, deg, interpolation=TF.InterpolationMode.BILINEAR, expand=True)
        # After expand, patch size changes
        _, ph, pw = patch.shape
        # If patch is too big, center crop to fit
        ph = min(ph, H)
        pw = min(pw, W)
        patch = TF.center_crop(patch, [ph, pw])
        # paste location
        py = int(torch.randint(0, H - ph + 1, (1,)).item())
        px = int(torch.randint(0, W - pw + 1, (1,)).item())
        out = x01.clone()
        out[:, py : py + ph, px : px + pw] = patch
        return out

    def __call__(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_norm is normalized image tensor (3,H,W). Returns normalized pseudo-anomaly."""
        x01 = _denorm(x_norm, self.img_mean, self.img_std).clamp(0.0, 1.0)
        if torch.rand(1).item() < self.p_scar:
            x01p = self._cutpaste_scar(x01)
        else:
            x01p = self._cutpaste_patch(x01)
        return _renorm(x01p, self.img_mean, self.img_std)

class DRAEMAnomaly:
    """DRAEM-style Perlin noise anomaly generator (image space)."""
    
    def __init__(self, img_mean, img_std, beta_range=(0.1, 1.0)):
        self.img_mean = img_mean
        self.img_std = img_std
        self.beta_range = beta_range
    
    def _perlin_mask(self, H, W):
        """Perlin noise -> binary mask"""
        scale = np.random.uniform(0.05, 0.15)  
        thresh = np.random.uniform(0.3, 0.7)  
        noise = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                noise[i, j] = pnoise2(i * scale, j * scale)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return (noise > thresh).astype(np.float32)  # (H, W)
    
    def __call__(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_norm: (3, H, W) normalized. Returns normalized pseudo-anomaly."""
        x01 = _denorm(x_norm, self.img_mean, self.img_std).clamp(0, 1)
        _, H, W = x01.shape
        
        mask = self._perlin_mask(H, W)  # (H, W), 0 or 1
        mask_t = torch.from_numpy(mask).to(x01.device)  # (H, W)
        
        texture = x01.clone()
        aug_ops = [
            lambda x: TF.adjust_brightness(x, np.random.uniform(0.5, 1.5)),
            lambda x: TF.adjust_contrast(x, np.random.uniform(0.5, 1.5)),
            lambda x: TF.adjust_saturation(x, np.random.uniform(0, 2.0)),
            lambda x: TF.adjust_hue(x, np.random.uniform(-0.3, 0.3)),
        ]
        chosen = np.random.choice(len(aug_ops), 3, replace=False)
        for i in chosen:
            texture = aug_ops[i](texture).clamp(0, 1)
        
        beta = np.random.uniform(*self.beta_range)
        mask3 = mask_t.unsqueeze(0)  
        out = (1 - mask3) * x01 + (1 - beta) * mask3 * x01 + beta * mask3 * texture
        out = out.clamp(0, 1)
        
        return _renorm(out, self.img_mean, self.img_std)

# ------------------------- Diffusion result pairs -------------------------

DIFFUSION_REQUIRED_BUNDLE_FILES = {
    "background.png",
    "foreground.png",
    "segmentation.png",
    "location.png",
}
DIFFUSION_SIZE_BUCKETS = {"small", "large"}
SAMPLE_NUMBER_PATTERN = re.compile(r"(?:^|_)(\d{6,7})(?:\D*$|$)")
OBJECT_LABEL_PATTERN = re.compile(r"(object_\d+)")


@dataclass(frozen=True)
class DiffusionPair:
    sample_key: str
    sample_name: str
    object_label: str
    size_bucket: str
    input_dir: str
    normal_path: str
    anomaly_path: str


def _normalize_ext(ext: str) -> str:
    return ext if ext.startswith(".") else f".{ext}"


def _sample_sort_key(sample_key: str, sample_name: str):
    matches = SAMPLE_NUMBER_PATTERN.findall(sample_name)
    sample_number = int(matches[-1]) if matches else float("inf")
    return sample_number, sample_key


def _object_label_for_sample(sample_name: str) -> str:
    match = OBJECT_LABEL_PATTERN.search(sample_name)
    return match.group(1) if match else "unknown"


def _size_bucket_for_rel_parent(rel_parent: str) -> str:
    if not rel_parent:
        return "all"
    first = Path(rel_parent).parts[0]
    return first if first in DIFFUSION_SIZE_BUCKETS else "all"


def _evenly_spaced(items: List[DiffusionPair], limit: Optional[int]) -> List[DiffusionPair]:
    if limit is None or limit < 0 or len(items) <= limit:
        return list(items)
    if limit <= 0:
        return []
    if limit == 1:
        return [items[0]]

    max_idx = len(items) - 1
    selected: List[DiffusionPair] = []
    seen = set()
    for i in range(limit):
        idx = int(round(i * max_idx / (limit - 1)))
        while idx in seen and idx < max_idx:
            idx += 1
        while idx in seen and idx > 0:
            idx -= 1
        if idx not in seen:
            selected.append(items[idx])
            seen.add(idx)
    return selected


def _discover_diffusion_inputs(input_root: Path) -> List[Tuple[str, str, str, Path]]:
    samples: List[Tuple[str, str, str, Path]] = []
    for root, dirs, files in os.walk(input_root):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        if DIFFUSION_REQUIRED_BUNDLE_FILES.issubset(set(files)):
            sample_dir = Path(root)
            rel_dir = os.path.relpath(sample_dir, input_root)
            sample_name = sample_dir.name
            rel_parent = os.path.dirname(rel_dir)
            if rel_parent == ".":
                rel_parent = ""
            sample_key = os.path.join(rel_parent, sample_name) if rel_parent else sample_name
            samples.append((sample_key, sample_name, rel_parent, sample_dir))
            dirs[:] = []
    return sorted(samples, key=lambda item: _sample_sort_key(item[0], item[1]))


def _build_diffusion_result_index(
    result_root: Path,
    flat_output: bool,
    output_ext: str,
) -> Tuple[Dict[str, Path], Dict[str, List[Path]], int]:
    by_key: Dict[str, Path] = {}
    by_name: Dict[str, List[Path]] = {}
    if not result_root.is_dir():
        return by_key, by_name, 0

    ext = _normalize_ext(output_ext).lower()
    result_count = 0
    for root, dirs, files in os.walk(result_root):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        root_path = Path(root)
        if flat_output:
            for name in files:
                if name.startswith("."):
                    continue
                path = root_path / name
                if path.suffix.lower() != ext:
                    continue
                rel_parent = os.path.relpath(root_path, result_root)
                if rel_parent == ".":
                    rel_parent = ""
                sample_name = path.stem
                sample_key = os.path.join(rel_parent, sample_name) if rel_parent else sample_name
                by_key.setdefault(sample_key, path)
                by_name.setdefault(sample_name, []).append(path)
                result_count += 1
        elif "results_highres.png" in files:
            path = root_path / "results_highres.png"
            rel_parent = os.path.relpath(root_path.parent, result_root)
            if rel_parent == ".":
                rel_parent = ""
            sample_name = root_path.name
            sample_key = os.path.join(rel_parent, sample_name) if rel_parent else sample_name
            by_key.setdefault(sample_key, path)
            by_name.setdefault(sample_name, []).append(path)
            result_count += 1
    return by_key, by_name, result_count


def _result_path_for_sample(
    result_by_key: Dict[str, Path],
    result_by_name: Dict[str, List[Path]],
    sample_key: str,
    sample_name: str,
) -> Optional[Path]:
    if sample_key in result_by_key:
        return result_by_key[sample_key]
    candidates = result_by_name.get(sample_name, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def _limit_diffusion_pairs_per_object(
    pairs: List[DiffusionPair],
    max_per_object: Optional[int],
) -> List[DiffusionPair]:
    if max_per_object is None or max_per_object < 0:
        return list(pairs)

    groups: Dict[str, Dict[str, List[DiffusionPair]]] = {}
    for pair in pairs:
        groups.setdefault(pair.object_label, {}).setdefault(pair.size_bucket, []).append(pair)

    selected_ids = set()
    for buckets in groups.values():
        for bucket_pairs in buckets.values():
            bucket_pairs.sort(key=lambda pair: _sample_sort_key(pair.sample_key, pair.sample_name))

        active_buckets = [name for name, items in sorted(buckets.items()) if items]
        if not active_buckets:
            continue
        if active_buckets == ["all"]:
            selected = _evenly_spaced(buckets["all"], max_per_object)
        else:
            quota = {
                name: min(len(buckets[name]), max_per_object // len(active_buckets))
                for name in active_buckets
            }
            remainder = max_per_object - sum(quota.values())
            while remainder > 0:
                added = False
                for name in active_buckets:
                    if quota[name] < len(buckets[name]):
                        quota[name] += 1
                        remainder -= 1
                        added = True
                        if remainder == 0:
                            break
                if not added:
                    break
            selected = []
            for name in active_buckets:
                selected.extend(_evenly_spaced(buckets[name], quota[name]))

        for pair in selected:
            selected_ids.add((pair.sample_key, pair.normal_path, pair.anomaly_path))

    return [
        pair for pair in pairs
        if (pair.sample_key, pair.normal_path, pair.anomaly_path) in selected_ids
    ]



def _load_diffusion_metadata(input_root: Path) -> Dict[str, dict]:
    metadata_path = input_root / "metadata.jsonl"
    records: Dict[str, dict] = {}
    if not metadata_path.exists():
        return records
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[Warning] invalid json at {metadata_path}:{line_no}: {exc}")
                continue
            sample = record.get("sample_id") or record.get("sample")
            if sample:
                records.setdefault(sample, record)
    return records


def _metadata_size_bucket(record: Optional[dict], fallback: str) -> str:
    if not record:
        return fallback
    value = record.get("placement_size_dir") or record.get("placement_size_label")
    return value if value in DIFFUSION_SIZE_BUCKETS else fallback


def _metadata_object_label(record: Optional[dict], sample_name: str) -> str:
    if record and record.get("object_label"):
        return str(record["object_label"])
    return _object_label_for_sample(sample_name)

def _build_diffusion_pairs(
    input_root: Path,
    result_root: Path,
    flat_output: bool,
    output_ext: str,
    max_per_object: Optional[int],
) -> Tuple[List[DiffusionPair], Dict[str, int]]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Diffusion input root not found: {input_root}")
    if not result_root.is_dir():
        raise FileNotFoundError(f"Diffusion result root not found: {result_root}")

    input_samples = _discover_diffusion_inputs(input_root)
    result_by_key, result_by_name, result_count = _build_diffusion_result_index(
        result_root,
        flat_output=flat_output,
        output_ext=output_ext,
    )
    metadata = _load_diffusion_metadata(input_root)

    pairs: List[DiffusionPair] = []
    pending_results = 0
    for sample_key, sample_name, rel_parent, input_dir in input_samples:
        record = metadata.get(sample_name)
        result_path = _result_path_for_sample(result_by_key, result_by_name, sample_key, sample_name)
        if result_path is None and record and record.get('anomaly_path'):
            candidate = Path(record['anomaly_path']).expanduser().resolve(strict=False)
            if candidate.exists():
                result_path = candidate
        if result_path is None or not result_path.exists():
            pending_results += 1
            continue
        normal_path = input_dir / "background.png"
        size_bucket = _metadata_size_bucket(record, _size_bucket_for_rel_parent(rel_parent))
        pairs.append(
            DiffusionPair(
                sample_key=sample_key,
                sample_name=sample_name,
                object_label=_metadata_object_label(record, sample_name),
                size_bucket=size_bucket,
                input_dir=str(input_dir),
                normal_path=str(normal_path),
                anomaly_path=str(result_path),
            )
        )

    selected_pairs = _limit_diffusion_pairs_per_object(pairs, max_per_object)
    stats = {
        "input_bundles": len(input_samples),
        "result_images": result_count,
        "completed_pairs": len(pairs),
        "pending_results": pending_results,
        "selected_pairs": len(selected_pairs),
    }
    return selected_pairs, stats


class DiffusionPairDataset(TorchDataset):
    """Normal/background and generated anomaly pairs from TALE POSCO outputs."""

    def __init__(
        self,
        input_root: str,
        result_root: str,
        input_size: Tuple[int, int],
        img_mean: List[float],
        img_std: List[float],
        *,
        flat_output: bool = True,
        output_ext: str = ".jpg",
        max_per_object: Optional[int] = 290,
        reverse: bool = False,
    ):
        self.input_root = Path(input_root).expanduser().resolve(strict=False)
        self.result_root = Path(result_root).expanduser().resolve(strict=False)
        self.pairs, self.stats = _build_diffusion_pairs(
            self.input_root,
            self.result_root,
            flat_output=flat_output,
            output_ext=output_ext,
            max_per_object=max_per_object,
        )
        if reverse:
            self.pairs = list(reversed(self.pairs))
        if not self.pairs:
            raise ValueError(
                f"No completed diffusion pairs found: input_root={self.input_root}, result_root={self.result_root}"
            )

        self.transform = T.Compose([
            T.Resize(input_size, InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(img_mean, img_std),
        ])
        self._print_summary(max_per_object)

    def _print_summary(self, max_per_object: Optional[int]) -> None:
        print(f"Diffusion pair train dataset: input_root={self.input_root}")
        print(f"Diffusion pair train dataset: result_root={self.result_root}")
        print(
            "Diffusion pairs: "
            f"input_bundles={self.stats['input_bundles']} "
            f"result_images={self.stats['result_images']} "
            f"completed_pairs={self.stats['completed_pairs']} "
            f"pending_results={self.stats['pending_results']} "
            f"selected_pairs={self.stats['selected_pairs']} "
            f"max_per_object={max_per_object}"
        )

        counts: Dict[str, Dict[str, int]] = {}
        for pair in self.pairs:
            counts.setdefault(pair.object_label, {}).setdefault(pair.size_bucket, 0)
            counts[pair.object_label][pair.size_bucket] += 1
        for object_label in sorted(counts):
            bucket_counts = counts[object_label]
            total = sum(bucket_counts.values())
            detail = ", ".join(f"{name}:{bucket_counts[name]}" for name in sorted(bucket_counts))
            print(f"  {object_label}: total={total} ({detail})")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        normal = Image.open(pair.normal_path).convert("RGB")
        anomaly = Image.open(pair.anomaly_path).convert("RGB")
        return self.transform(normal), self.transform(anomaly), pair.sample_key

# ------------------------- Rectified Flow nets -------------------------

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding + MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0,1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, tdim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.to_scale_shift = nn.Linear(tdim, channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: (B, tdim)
        h = self.conv1(F.silu(self.norm1(x)))
        ss = self.to_scale_shift(t_emb)  # (B, 2C)
        scale, shift = ss.chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return x + h


class SmallConvVelocityNet(nn.Module):
    """Small fully-convolutional velocity net for one scale."""

    def __init__(self, channels: int, tdim: int = 64, depth: int = 4):
        super().__init__()
        self.tdim = tdim
        self.time = TimeEmbedding(tdim)
        self.in_proj = nn.Conv2d(channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMResBlock(channels, tdim * 4) for _ in range(depth)])
        self.out_proj = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W), t: (B,)
        t_emb = self.time(t)  # (B, tdim*4)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out_proj(h)


class MultiScaleRF(nn.Module):
    """Per-scale velocity nets for MSFlow fusion latents.

    Supports either:
      (A) shared hyperparams for all stages: tdim, depth
      (B) stage-wise hyperparams: tdims[i], depths[i]
    """

    def __init__(
        self,
        channels_list: List[int],
        tdim: int = 64,
        depth: int = 4,
        tdims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
    ):
        super().__init__()
        if tdims is not None or depths is not None:
            assert tdims is not None and depths is not None, "Provide both tdims and depths for stage-wise RF."
            assert len(tdims) == len(channels_list), f"tdims length {len(tdims)} must match stages {len(channels_list)}"
            assert len(depths) == len(channels_list), f"depths length {len(depths)} must match stages {len(channels_list)}"
            self.nets = nn.ModuleList(
                [SmallConvVelocityNet(c, tdim=int(ti), depth=int(di)) for c, ti, di in zip(channels_list, tdims, depths)]
            )
        else:
            self.nets = nn.ModuleList([SmallConvVelocityNet(c, tdim=tdim, depth=depth) for c in channels_list])

    def forward(self, z_list: List[torch.Tensor], t: torch.Tensor) -> List[torch.Tensor]:
        return [net(z, t) for net, z in zip(self.nets, z_list)]


# ------------------------- MSFlow forward (copied from train.py) -------------------------

def msflow_forward(c, extractor, parallel_flows, fusion_flow, image, *, return_pre_fusion: bool = False):
    h_list = extractor(image)
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    z_pre_list = []
    parallel_jac_list = []
    for h, parallel_flow, c_cond in zip(h_list, parallel_flows, c.c_conds):
        y = pool_layer(h)
        B, _, H, W = y.shape
        cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        z, jac = parallel_flow(y, [cond, ])
        z_pre_list.append(z)
        parallel_jac_list.append(jac)

    # Fusion happens AFTER the per-scale parallel flows.
    z_fused_list, fuse_jac = fusion_flow(z_pre_list)
    jac = fuse_jac + sum(parallel_jac_list)

    if return_pre_fusion:
        return z_pre_list, z_fused_list, jac
    return z_fused_list, jac


# ------------------------- Rectified Flow training / sampling -------------------------

def rf_loss(
    rf: MultiScaleRF,
    z0_list: List[torch.Tensor],
    z1_list: List[torch.Tensor],
) -> torch.Tensor:
    """Rectified-flow linear-path loss across all scales."""
    B = z0_list[0].shape[0]
    t = torch.rand(B, device=z0_list[0].device)
    loss = 0.0
    for z0, z1, net in zip(z0_list, z1_list, rf.nets):
        # x_t = (1-t)x0 + t x1
        t_b = t.view(B, 1, 1, 1)
        zt = (1 - t_b) * z0 + t_b * z1
        v_target = (z1 - z0)
        v_pred = net(zt, t)
        loss = loss + F.mse_loss(v_pred, v_target)
    return loss / len(z0_list)



@torch.no_grad()
def rf_transport(
    rf: MultiScaleRF,
    z0_list: List[torch.Tensor],
    steps: int = 16,
) -> List[torch.Tensor]:
    """Euler integrate from t=0..1 starting at z0."""
    z = [t.clone() for t in z0_list]
    B = z[0].shape[0]
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((B,), k / steps, device=z[0].device, dtype=torch.float32)
        v_list = rf(z, t)
        z = [zi + dt * vi for zi, vi in zip(z, v_list)]
    return z


def _make_outputs_list_from_z(z_list: List[torch.Tensor]) -> Tuple[List[List[torch.Tensor]], List[List[int]]]:
    """Create MSFlow-style outputs_list + size_list from z_list."""
    outputs_list = [list() for _ in z_list]
    size_list = []
    for lvl, z in enumerate(z_list):
        size_list.append(list(z.shape[-2:]))
        logp = -0.5 * torch.mean(z ** 2, dim=1)  # (B,H,W)
        outputs_list[lvl].append(logp)
    return outputs_list, size_list


def _outputs_list_diff(outputs_a: List[List[torch.Tensor]], outputs_b: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Assumes same structure: per-level list with a single (B,H,W) tensor."""
    out = []
    for la, lb in zip(outputs_a, outputs_b):
        assert len(la) == len(lb) == 1, "This helper expects per-level lists with a single batch tensor."
        out.append([lb[0] - la[0]])  # logp_rect - logp
    return out

def joint_minmax(S, D, eps=1e-8):
    m = min(S.min(), D.min())
    M = max(S.max(), D.max())
    S_n = (S - m) / (M - m + eps)
    D_n = (D - m) / (M - m + eps)
    return S_n, D_n

def minmax_norm(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

@torch.no_grad()
def eval_rf_epoch(
    c,
    epoch: int,
    loader: DataLoader,
    extractor,
    parallel_flows,
    fusion_flow,
    rf_model: MultiScaleRF,
    ode_steps: int,
    det_auroc_obs: Score_Observer,
    loc_auroc_obs: Score_Observer,
    loc_pro_obs: Score_Observer,
    pro_eval: bool = False,
):
    """Evaluate with rectified-flow scoring.

    Builds two MSFlow-style outputs_list:
      - outputs_list from z
      - outputs_list_diff from (logp_rect - logp)
    Then calls post_process twice and sums the two add-maps.
    """
    rf_model = rf_model.eval()
    parallel_flows = [pf.eval() for pf in parallel_flows]
    fusion_flow = fusion_flow.eval()

    gt_label_list: List[int] = []
    gt_mask_list: List[np.ndarray] = []
    outputs_list = [list() for _ in parallel_flows]
    outputs_list_diff = [list() for _ in parallel_flows]
    size_list: List[List[int]] = []
    diff_l01_list: List[List[torch.Tensor]] = [[], []]  # store L0/L1 diff maps (CPU)

    start = time.time()
    for idx, (image, label, mask) in enumerate(loader):
        image = image.to(c.device, non_blocking=True)
        gt_label_list.extend(t2np(label))
        gt_mask_list.extend(t2np(mask))

        # Get both pre-fusion and fused latents.
        z_pre_list, z_fused_list, _ = msflow_forward(
            c, extractor, parallel_flows, fusion_flow, image, return_pre_fusion=True
        )

        # Rectified flow is applied on *fusion* stages [0, 1].
        z_rf_in = [z_fused_list[i] for i in [0, 1]]
        z_fused_rect_l01 = rf_transport(rf_model, z_rf_in, steps=ode_steps)

        # Build a full fused rectified list (only stages [0,1] are rectified).
        z_fused_rect_full = [z.clone() for z in z_fused_list]
        z_fused_rect_full[0] = z_fused_rect_l01[0]
        z_fused_rect_full[1] = z_fused_rect_l01[1]

        # Baseline outputs (ALL stages) for detection
        for lvl, z in enumerate(z_fused_list):
            if idx == 0:
                size_list.append(list(z.shape[-2:]))
            logp = -0.5 * torch.mean(z ** 2, dim=1)   # (B,H,W)
            outputs_list[lvl].append(logp)
        
        # Diff maps (ONLY stages [0, 1]) for localization (store CPU tensors)
        diff_maps = []
        for lvl in [0, 1]:
            z = z_fused_list[lvl]
            zr = z_fused_rect_full[lvl]
            logp = -0.5 * torch.mean(z ** 2, dim=1)
            logp_r = -0.5 * torch.mean(zr ** 2, dim=1)
            diff_maps.append((logp_r - logp).detach().cpu())  # (B,H,W)
        
        diff_l01_list[0].append(diff_maps[0])
        diff_l01_list[1].append(diff_maps[1])

    fps = len(loader.dataset) / max(time.time() - start, 1e-6)
    print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"), f"Epoch {epoch} RF-eval FPS: {fps:.1f}")

    anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

    # Localization map from RAW diff (L0+L1), no normalization
    gt_full = np.asarray(gt_mask_list)
    if gt_full.ndim == 4 and gt_full.shape[1] == 1:
        gt_full = gt_full[:, 0]
    elif gt_full.ndim != 3:
        gt_full = np.squeeze(gt_full)
    H0, W0 = gt_full.shape[-2], gt_full.shape[-1]

    d0 = torch.cat(diff_l01_list[0], dim=0).unsqueeze(1)  # (N,1,h,w)
    d1 = torch.cat(diff_l01_list[1], dim=0).unsqueeze(1)
    if d0.shape[-2:] != (H0, W0):
        d0 = F.interpolate(d0, size=(H0, W0), mode='bilinear', align_corners=False)
    if d1.shape[-2:] != (H0, W0):
        d1 = F.interpolate(d1, size=(H0, W0), mode='bilinear', align_corners=False)

    # anomaly_score_map_add_final = minmax_norm((d0[:,0] + d1[:,0]).numpy())
    anomaly_score_map_add_final = minmax_norm((d0[:,0] + d1[:,0]).numpy()) * 5 + anomaly_score_map_add

    # Detection score: keep original MSFlow anomaly_score
    anomaly_score_final = anomaly_score

    det_auroc, loc_auroc, loc_pro_auc, best_det_auroc, best_loc_auroc, best_loc_pro = \
        eval_det_loc(
            det_auroc_obs,
            loc_auroc_obs,
            loc_pro_obs,
            epoch,
            gt_label_list,
            anomaly_score_final,
            gt_mask_list,
            anomaly_score_map_add_final,
            anomaly_score_map_mul,
            pro_eval,
        )

    return det_auroc, loc_auroc, loc_pro_auc, best_det_auroc, best_loc_auroc, best_loc_pro


# ------------------------- Main train/test loops -------------------------


def build_args():
    parser = argparse.ArgumentParser(description='Rectified-Flow on top of MSFlow')
    parser.add_argument('--dataset', default='mvtec', choices=['mvtec', 'visa', 'posco'])
    parser.add_argument('--class-name', default='bottle', type=str)
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str)
    parser.add_argument('--pool-type', default='avg', type=str)
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, nargs='+')

    # MSFlow ckpt (auto-resolved by default)
    # If you trained MSFlow with the official script, checkpoints are saved like:
    #   work_dirs/<msflow_version>/<dataset>/<class_name>/best_loc_auroc.pt
    # You can either provide --msflow-ckpt explicitly, or let this script resolve
    # it from --msflow-work-dir/--msflow-version.
    parser.add_argument('--msflow-ckpt', default='', type=str,
                        help='(optional) explicit path to a trained MSFlow checkpoint. If empty, auto-resolve.')
    parser.add_argument('--msflow-work-dir', default='work_dirs', type=str,
                        help='Root work dir that contains MSFlow checkpoints (default: work_dirs).')
    parser.add_argument('--msflow-version', default='msflow_wide_resnet50_2_avgpool_pl258', type=str,
                        help='MSFlow experiment folder name under work_dirs (default matches your example).')
    parser.add_argument('--msflow-ckpt-name', default='best_loc_auroc.pt', type=str,
                        help='MSFlow checkpoint filename inside class folder (default: best_loc_auroc.pt).')

    # RF training
    parser.add_argument('--rf-epochs', default=10, type=int)
    parser.add_argument('--rf-lr', default=1e-4, type=float)
    parser.add_argument('--rf-depth', default=4, type=int)
    parser.add_argument('--rf-tdim', default=64, type=int)
    # Stage-wise RF hyperparams (for L0/L1 when you slice stages). Example:
    #   --rf-tdims 32 16  --rf-depths 3 2
    # If provided, these override --rf-tdim/--rf-depth.
    parser.add_argument('--rf-tdims', default=None, type=int, nargs='+', help='Per-stage tdim list (e.g., 32 16).')
    parser.add_argument('--rf-depths', default=None, type=int, nargs='+', help='Per-stage depth list (e.g., 3 2).')
    parser.add_argument('--rf-steps', default=16, type=int, help='ODE steps for test-time transport.')
    parser.add_argument('--rf-ckpt', default='', type=str, help='path to RF checkpoint for testing or resume.')
    parser.add_argument('--eval-every', default=1, type=int,
                        help='Evaluate every N epochs. Use 0 to disable evaluation during RF training.')
    parser.add_argument('--skip-eval', action='store_true', default=False,
                        help='Skip dataset evaluation during RF training.')

    # Optional TALE/POSCO diffusion pair training source.
    parser.add_argument('--train-source', default='dataset', choices=['dataset', 'diffusion'],
                        help='dataset: create pseudo anomalies online. diffusion: use background/result image pairs.')
    parser.add_argument('--diffusion-input-root', default='', type=str,
                        help='Root with input bundles containing background.png, e.g. diffusion_result/input.')
    parser.add_argument('--diffusion-result-root', default='', type=str,
                        help='Root with generated results, e.g. diffusion_result/result.')
    parser.add_argument('--diffusion-output-ext', default='.jpg', type=str,
                        help='Generated result extension when --diffusion-flat-output is enabled.')
    parser.add_argument('--diffusion-flat-output', action=argparse.BooleanOptionalAction, default=True,
                        help='Match output_dir/<size>/<sample>.jpg. Use --no-diffusion-flat-output for <sample>/results_highres.png.')
    parser.add_argument('--diffusion-max-per-object', default=290, type=int,
                        help='Cap completed diffusion pairs per object, balanced across small/large. Use -1 for no cap.')
    parser.add_argument('--diffusion-reverse', action='store_true', default=False,
                        help='Reverse selected diffusion pair order after balancing.')

    # paths
    parser.add_argument('--data-path', default='', type=str)
    # work-dir for RF checkpoints/logs
    parser.add_argument('--work-dir', default='./work_dirs', type=str)
    parser.add_argument('--pro-eval', action='store_true', default=False)

    return parser


def init_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_defaults(c, args):
    # minimal config fields expected by MSFlow code
    c.dataset = args.dataset
    c.class_name = args.class_name
    c.extractor = args.extractor
    c.pool_type = args.pool_type
    c.parallel_blocks = args.parallel_blocks
    c.batch_size = args.batch_size
    c.workers = args.workers
    c.pro_eval = args.pro_eval
    c.pro_eval_interval = 4

    # device
    c.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset paths
    if args.data_path:
        c.data_path = args.data_path
    elif c.dataset == 'mvtec':
        c.data_path = './data/MVTec'
    elif c.dataset == 'visa':
        c.data_path = './data/VisA_pytorch/1cls'
    elif c.dataset == 'posco':
        c.data_path = './data/posco'
    else:
        raise ValueError(f'Unsupported dataset: {c.dataset}')

    # image size rule from original main.py
    c.input_size = (256, 256) if c.class_name == 'transistor' else (512, 512)

    # normalization used in datasets.py
    c.img_mean = [0.485, 0.456, 0.406]
    c.img_std = [0.229, 0.224, 0.225]

    # conditioning dims for flows (default.py defines these; we keep identical by importing default)
    # If default.py provides c.c_conds, we'll keep it; otherwise set a safe default.
    if not hasattr(c, 'c_conds'):
        # Typical MSFlow uses [128, 256, 512] cond channels for 3 scales.
        c.c_conds = [128, 256, 512]

    # workdir
    c.version_name = f"rf_on_msflow_{c.extractor}_{c.pool_type}pool_pl{''.join([str(x) for x in c.parallel_blocks])}"
    c.ckpt_dir = os.path.join(args.work_dir, c.version_name, c.dataset, c.class_name)
    os.makedirs(c.ckpt_dir, exist_ok=True)

    return c


def resolve_msflow_ckpt(args) -> str:
    """Resolve MSFlow checkpoint path.

    Priority:
      1) --msflow-ckpt if provided
      2) work_dirs/<msflow_version>/<dataset>/<class_name>/<msflow_ckpt_name>
    """
    if args.msflow_ckpt:
        return args.msflow_ckpt
    return os.path.join(
        args.msflow_work_dir,
        args.msflow_version,
        args.dataset,
        args.class_name,
        args.msflow_ckpt_name,
    )


def load_msflow_frozen(c, msflow_ckpt_path: str):
    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    parallel_flows, fusion_flow = build_msflow_model(c, output_channels)
    parallel_flows = [pf.to(c.device).eval() for pf in parallel_flows]
    fusion_flow = fusion_flow.to(c.device).eval()

    # load weights
    _ = load_weights(parallel_flows, fusion_flow, msflow_ckpt_path)

    # freeze
    for p in extractor.parameters():
        p.requires_grad_(False)
    for pf in parallel_flows:
        for p in pf.parameters():
            p.requires_grad_(False)
    for p in fusion_flow.parameters():
        p.requires_grad_(False)

    return extractor, parallel_flows, fusion_flow


def select_dataset_class(name: str):
    if name == 'mvtec':
        return MVTecDataset
    if name == 'visa':
        return VisADataset
    if name == 'posco':
        return POSCODataset
    raise ValueError(f'Unsupported dataset: {name}')


def _build_train_loader(c, args):
    if args.train_source == 'diffusion':
        if not args.diffusion_input_root:
            raise ValueError('--diffusion-input-root is required when --train-source diffusion')
        if not args.diffusion_result_root:
            raise ValueError('--diffusion-result-root is required when --train-source diffusion')
        train_dataset = DiffusionPairDataset(
            args.diffusion_input_root,
            args.diffusion_result_root,
            c.input_size,
            c.img_mean,
            c.img_std,
            flat_output=args.diffusion_flat_output,
            output_ext=args.diffusion_output_ext,
            max_per_object=args.diffusion_max_per_object,
            reverse=args.diffusion_reverse,
        )
    else:
        Dataset = select_dataset_class(c.dataset)
        train_dataset = Dataset(c, is_train=True)

    return DataLoader(
        train_dataset,
        batch_size=c.batch_size,
        shuffle=True,
        num_workers=c.workers,
        pin_memory=True,
    )


def _build_optional_eval_loader(c, args):
    if args.skip_eval or args.eval_every <= 0:
        return None
    Dataset = select_dataset_class(c.dataset)
    try:
        test_dataset = Dataset(c, is_train=False)
    except (AssertionError, FileNotFoundError) as exc:
        if args.train_source == 'diffusion':
            print(f'[Warning] skip RF eval because test dataset is unavailable: {exc}')
            return None
        raise
    return DataLoader(
        test_dataset,
        batch_size=c.batch_size,
        shuffle=False,
        num_workers=c.workers,
        pin_memory=True,
    )


def train_rf(args):
    import default as c
    c = resolve_defaults(c, args)
    init_seeds(args.seed)

    train_loader = _build_train_loader(c, args)

    msflow_ckpt = resolve_msflow_ckpt(args)
    assert os.path.isfile(msflow_ckpt), f"MSFlow checkpoint not found: {msflow_ckpt}"
    print(f"[MSFlow] load: {msflow_ckpt}")
    extractor, parallel_flows, fusion_flow = load_msflow_frozen(c, msflow_ckpt)

    test_loader = _build_optional_eval_loader(c, args)
    det_auroc_obs = Score_Observer('Det.AUROC', args.rf_epochs)
    loc_auroc_obs = Score_Observer('Loc.AUROC', args.rf_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', args.rf_epochs)

    # Build RF nets after we see z channels (depends on MSFlow extractor choices)
    rf_model: Optional[MultiScaleRF] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    anomaly_gen = None
    if args.train_source == 'dataset':
        anomaly_gen = DRAEMAnomaly(c.img_mean, c.img_std, beta_range=(0.1, 1.0))

    for epoch in range(args.rf_epochs):
        if rf_model is not None:
            rf_model.train()
        epoch_loss = 0.0
        n = 0
        for batch in train_loader:
            if args.train_source == 'diffusion':
                img, img_pseudo, _ = batch
                img = img.to(c.device, non_blocking=True)
                img_pseudo = img_pseudo.to(c.device, non_blocking=True)
            else:
                img, y, _ = batch
                # train set is normal-only, but keep safe
                if y.sum().item() != 0:
                    img = img[y == 0]
                    if img.numel() == 0:
                        continue
                img = img.to(c.device, non_blocking=True)
                assert anomaly_gen is not None
                p_identity = 0.10
                img_pseudo_list = []
                for x in img:
                    if torch.rand(1).item() < p_identity:
                        img_pseudo_list.append(x)
                    else:
                        img_pseudo_list.append(anomaly_gen(x))
                img_pseudo = torch.stack(img_pseudo_list, dim=0)

            # --- MSFlow forward ---
            # We APPLY rectified flow on the *fusion* latents (after fusion_flow).
            with torch.no_grad():
                _, z_fused_list, _ = msflow_forward(c, extractor, parallel_flows, fusion_flow, img, return_pre_fusion=True)
                _, z_fused_p_list, _ = msflow_forward(c, extractor, parallel_flows, fusion_flow, img_pseudo, return_pre_fusion=True)

            # Use only fusion stages [0, 1] for rectified flow (train)
            z_list = [z_fused_list[i] for i in [0, 1]]
            z_p_list = [z_fused_p_list[i] for i in [0, 1]]

            if rf_model is None:
                channels_list = [z.shape[1] for z in z_list]
                rf_tdims = args.rf_tdims
                rf_depths = args.rf_depths
                if rf_tdims is not None or rf_depths is not None:
                    rf_model = MultiScaleRF(channels_list, tdims=rf_tdims, depths=rf_depths).to(c.device)
                else:
                    rf_model = MultiScaleRF(channels_list, tdim=args.rf_tdim, depth=args.rf_depth).to(c.device)
                optimizer = torch.optim.Adam(rf_model.parameters(), lr=args.rf_lr)
                if args.rf_ckpt and os.path.isfile(args.rf_ckpt):
                    ckpt = torch.load(args.rf_ckpt, map_location='cpu')
                    rf_model.load_state_dict(ckpt['rf_model'])
                    if 'optimizer' in ckpt:
                        optimizer.load_state_dict(ckpt['optimizer'])
                    print(f"[RF] resumed from {args.rf_ckpt}")

            assert rf_model is not None and optimizer is not None
            optimizer.zero_grad(set_to_none=True)

            loss = rf_loss(rf_model, z_p_list, z_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rf_model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss.item()) * img.shape[0]
            n += img.shape[0]

        avg_loss = epoch_loss / max(n, 1)
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"), f"Epoch {epoch} RF train loss: {avg_loss:.6e} samples={n}")

        if rf_model is not None:
            ckpt_last = os.path.join(c.ckpt_dir, "rf_last.pt")
            torch.save({'rf_model': rf_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, ckpt_last)
            print(f"[RF] saved: {ckpt_last}")

        should_eval = (
            rf_model is not None
            and test_loader is not None
            and args.eval_every > 0
            and ((epoch + 1) % args.eval_every == 0)
        )
        if should_eval:
            eval_rf_epoch(
                c,
                epoch,
                test_loader,
                extractor,
                parallel_flows,
                fusion_flow,
                rf_model,
                args.rf_steps,
                det_auroc_obs,
                loc_auroc_obs,
                loc_pro_obs,
                pro_eval=c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0),
            )


def main():
    parser = build_args()
    args = parser.parse_args()
    train_rf(args)


if __name__ == '__main__':
    main()
