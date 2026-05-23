from __future__ import annotations

import os
import re
import sys
import argparse
import datetime
import time
import sysconfig
import importlib.util
from functools import lru_cache
from typing import List, Optional


def ensure_stdlib_copy_module():
    copy_path = os.path.join(sysconfig.get_paths()['stdlib'], 'copy.py')
    spec = importlib.util.spec_from_file_location('copy', copy_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['copy'] = module


ensure_stdlib_copy_module()

import cv2
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import default as c
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from post_process import post_process
from utils import load_weights
from rectified_flow_train_posco import MultiScaleRF, msflow_forward, rf_transport, minmax_norm



def option_was_provided(argv: List[str], *names: str) -> bool:
    for arg in argv:
        if arg in names:
            return True
        if any(arg.startswith(name + '=') for name in names):
            return True
    return False


def expand_path_string(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.expandvars(os.path.expanduser(path))


def apply_keep_mask_to_anomaly_map(
    anomaly_map,
    mask_dir,
    folder_name,
    threshold=10,
    erode_pixels=0,
    close_kernel=7,
):
    """
    mask white area -> keep anomaly score
    mask black area -> force anomaly score to 0

    close_kernel > 0:
      fill small black holes inside the white ROI area.

    erode_pixels > 0:
      shrink white ROI area slightly. Default should normally be 0 because
      erosion can remove valid white ROI boundary pixels.
    """
    mask_path = os.path.join(mask_dir, f"{folder_name}_mask.jpg")

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    h, w = anomaly_map.shape[:2]

    # Resize mask to anomaly map size
    mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

    # white area = valid ROI
    keep = (mask_img > threshold).astype(np.uint8)

    # Fill small black holes inside the white ROI.
    # This fixes tiny black dots/noise inside mask white regions.
    if close_kernel is not None and close_kernel > 0:
        if close_kernel % 2 == 0:
            close_kernel += 1
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel)

    # Optional: shrink valid ROI slightly to remove boundary artifacts.
    # Use 0 if you do not want to remove any valid white ROI pixels.
    if erode_pixels > 0:
        kernel = np.ones((erode_pixels, erode_pixels), np.uint8)
        keep = cv2.erode(keep, kernel, iterations=1)

    debug_dir = "./debug_mask_check"
    os.makedirs(debug_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(debug_dir, f"{folder_name}_keep_mask_used.png"),
        (keep * 255).astype(np.uint8)
    )

    filtered = anomaly_map.copy()
    filtered[keep == 0] = 0.0

    return filtered

class PoscoTestFolderDataset(Dataset):
    """
    POSCO test dataset for one subfolder/class.

    Expected structure:
      data/posco/test/
        normal/<folder_name>/*.png
        abnormal/<folder_name>/*.png

    Example for folder_name='02':
      data/posco/test/normal/02/*.png
      data/posco/test/abnormal/02/*.png
    """
    def __init__(self, data_root: str, folder_name: str, input_size=(512, 512), img_mean=None, img_std=None,
                 apply_test_mask: bool = False, mask_dir: str = './mask', mask_threshold: int = 10):
        self.data_root = data_root
        self.folder_name = folder_name
        self.input_size = input_size
        self.apply_test_mask = apply_test_mask
        self.mask_dir = mask_dir
        self.mask_threshold = mask_threshold
        self.img_info_list = self._collect_images(data_root, folder_name)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(img_mean, img_std)

    @staticmethod
    def _collect_images(data_root: str, folder_name: str):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        assert os.path.isdir(data_root), f"POSCO test folder not found: {data_root}"

        img_info = []
        for label_name in ['normal', 'abnormal']:
            folder_path = os.path.join(data_root, label_name, folder_name)
            if not os.path.isdir(folder_path):
                print(f"[Warning] Missing test folder, skip: {folder_path}")
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(exts):
                    img_path = os.path.join(folder_path, fname)
                    img_info.append((img_path, label_name, folder_name, fname))

        assert len(img_info) > 0, (
            f"No test images found for folder {folder_name!r}. Expected images under:\n"
            f"  {os.path.join(data_root, 'normal', folder_name)}\n"
            f"  {os.path.join(data_root, 'abnormal', folder_name)}"
        )
        return img_info

    def __len__(self):
        return len(self.img_info_list)

    def __getitem__(self, idx):
        img_path, label_name, folder_name, fname = self.img_info_list[idx]
        original_img = Image.open(img_path).convert('RGB')

        # IMPORTANT:
        # Do NOT apply the ROI mask to the model input.
        # The model sees the original test image. The ROI mask is applied only
        # later to the anomaly map before bbox generation.
        model_img = original_img.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)

        x = self.normalize(self.to_tensor(model_img))
        return x, img_path, label_name, folder_name, fname


class PoscoFlatTestDataset(Dataset):
    """
    POSCO flat test dataset for the whole POSCO model.

    Expected structure:
      data/posco/test/
        normal/*.jpg|jpeg|png|bmp|tif|tiff|webp
        abnormal/*.jpg|jpeg|png|bmp|tif|tiff|webp

    This also supports nested folders under normal/abnormal, but outputs are grouped
    only by label name: normal or abnormal.
    """
    def __init__(self, data_root: str, input_size=(512, 512), img_mean=None, img_std=None,
                 apply_test_mask: bool = False, mask_dir: str = './mask', mask_threshold: int = 10):
        self.data_root = data_root
        self.input_size = input_size
        self.apply_test_mask = apply_test_mask
        self.mask_dir = mask_dir
        self.mask_threshold = mask_threshold
        self.img_info_list = self._collect_images(data_root)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(img_mean, img_std)

    @staticmethod
    def _collect_images(data_root: str):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        assert os.path.isdir(data_root), f"POSCO test root not found: {data_root}"

        img_info = []
        for label_name in ['normal', 'abnormal']:
            label_dir = os.path.join(data_root, label_name)
            assert os.path.isdir(label_dir), f"Missing POSCO test folder: {label_dir}"

            for dirpath, _, filenames in os.walk(label_dir):
                rel_dir = os.path.relpath(dirpath, label_dir)
                rel_prefix = '' if rel_dir == '.' else rel_dir.replace(os.sep, '_') + '_'
                for fname in sorted(filenames):
                    if not fname.lower().endswith(exts):
                        continue
                    img_path = os.path.join(dirpath, fname)
                    # Keep flat output safe even when nested input folders have same filenames.
                    save_fname = rel_prefix + fname
                    folder_name = rel_dir.split(os.sep)[0] if rel_dir != '.' else None
                    img_info.append((img_path, label_name, folder_name, save_fname))

        assert len(img_info) > 0, (
            f"No POSCO test images found. Expected images under:\n"
            f"  {os.path.join(data_root, 'normal')}\n"
            f"  {os.path.join(data_root, 'abnormal')}"
        )
        normal_count = sum(1 for _, label, _, _ in img_info if label == 'normal')
        abnormal_count = sum(1 for _, label, _, _ in img_info if label == 'abnormal')
        print(f"[INFO] POSCO flat test images: normal={normal_count}, abnormal={abnormal_count}")
        return img_info

    def __len__(self):
        return len(self.img_info_list)

    def __getitem__(self, idx):
        img_path, label_name, folder_name, save_fname = self.img_info_list[idx]
        original_img = Image.open(img_path).convert('RGB')

        # IMPORTANT:
        # Do NOT apply the ROI mask to the model input.
        # The model sees the original test image. The ROI mask is applied only
        # later to the anomaly map before bbox generation.
        if self.apply_test_mask and folder_name is None:
            raise ValueError(
                f"Cannot choose mask for flat test image without subfolder: {img_path}. "
                "Use data/posco/test/{normal,abnormal}/02/*.jpg style folders, "
                "or use --visualize-by-folder."
            )
        model_img = original_img.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)

        x = self.normalize(self.to_tensor(model_img))
        return x, img_path, label_name, folder_name, save_fname

def build_msflow(cfg, ckpt_path: str):
    extractor, output_channels = build_extractor(cfg)
    extractor = extractor.to(cfg.device).eval()

    parallel_flows, fusion_flow = build_msflow_model(cfg, output_channels)
    parallel_flows = [pf.to(cfg.device).eval() for pf in parallel_flows]
    fusion_flow = fusion_flow.to(cfg.device).eval()

    print(f"[INFO] Loading MSFlow checkpoint: {ckpt_path}")
    load_weights(parallel_flows, fusion_flow, ckpt_path)

    return extractor, parallel_flows, fusion_flow


def build_rf_from_batch(device, rf_ckpt_path: str, z_fused_list: List[torch.Tensor], rf_tdims, rf_depths):
    channels_list = [z_fused_list[i].shape[1] for i in [0, 1]]
    rf_model = MultiScaleRF(
        channels_list,
        tdims=rf_tdims,
        depths=rf_depths,
    ).to(device).eval()

    print(f"[INFO] Loading Rectified-Flow checkpoint: {rf_ckpt_path}")
    ckpt = torch.load(rf_ckpt_path, map_location='cpu')
    state = ckpt['rf_model'] if isinstance(ckpt, dict) and 'rf_model' in ckpt else ckpt
    rf_model.load_state_dict(state)
    return rf_model


def squeeze_score_map(anomaly_map: np.ndarray) -> np.ndarray:
    score_map = np.asarray(anomaly_map, dtype=np.float32)
    if score_map.ndim != 2:
        score_map = np.squeeze(score_map)
    return score_map


def minmax_score_map(anomaly_map: np.ndarray) -> np.ndarray:
    score_map = squeeze_score_map(anomaly_map)
    if score_map.size == 0:
        return score_map
    finite = np.isfinite(score_map)
    if not finite.any():
        return np.zeros_like(score_map, dtype=np.float32)
    vmin = float(np.nanmin(score_map))
    vmax = float(np.nanmax(score_map))
    denom = vmax - vmin
    if denom <= 1e-8:
        return np.zeros_like(score_map, dtype=np.float32)
    return ((score_map - vmin) / denom).astype(np.float32)


def bbox_score_map(anomaly_map: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'raw':
        return squeeze_score_map(anomaly_map)
    if mode == 'minmax':
        return minmax_score_map(anomaly_map)
    raise ValueError(f'Unsupported bbox score mode: {mode}')


def save_score_maps(out_dir: str, stem: str, raw_map: np.ndarray, selected_map: np.ndarray,
                    mode: str, threshold: float):
    raw_map = squeeze_score_map(raw_map)
    selected_map = squeeze_score_map(selected_map)
    np.save(os.path.join(out_dir, f'{stem}_score_raw.npy'), raw_map)
    np.save(os.path.join(out_dir, f'{stem}_score_{mode}.npy'), selected_map)
    with open(os.path.join(out_dir, f'{stem}_score_stats.txt'), 'w') as f:
        f.write(f'bbox_score_mode: {mode}\n')
        f.write(f'threshold: {threshold}\n')
        maps_to_log = [('raw', raw_map)]
        if mode != 'raw':
            maps_to_log.append((mode, selected_map))
        for name, score_map in maps_to_log:
            f.write(f'{name}_min: {float(np.nanmin(score_map)):.8f}\n')
            f.write(f'{name}_max: {float(np.nanmax(score_map)):.8f}\n')
            f.write(f'{name}_mean: {float(np.nanmean(score_map)):.8f}\n')
            f.write(f'{name}_std: {float(np.nanstd(score_map)):.8f}\n')


def anomaly_map_to_bboxes(anomaly_map: np.ndarray, threshold=0.5, min_area=50):
    score_map = squeeze_score_map(anomaly_map)
    binary = (score_map >= threshold).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    bboxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def draw_bboxes_on_image(img: Image.Image, bboxes, color='red', width=3):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in bboxes:
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return out


def image_name_to_gt_txt_candidates(fname: str) -> List[str]:
    """Return GT txt filename candidates for exact-frame and sequence-level GT files."""
    stem, _ = os.path.splitext(os.path.basename(fname))
    stripped = re.sub(r'_\d+$', '', stem)

    candidates = []
    for candidate in (stem + '.txt', stripped + '.txt'):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def image_name_to_gt_txt(fname: str) -> str:
    """Backward-compatible sequence-level GT name for a frame image."""
    return image_name_to_gt_txt_candidates(fname)[-1]


@lru_cache(maxsize=16)
def build_gt_name_index(gt_dir: str):
    index = {}
    if not os.path.isdir(gt_dir):
        return index
    for dirpath, _, filenames in os.walk(gt_dir):
        for name in filenames:
            if not name.lower().endswith('.txt'):
                continue
            index.setdefault(name.lower(), []).append(os.path.join(dirpath, name))
    return index


def choose_indexed_gt(paths, folder_name: Optional[str], label_name: Optional[str]) -> Optional[str]:
    if not paths:
        return None

    preferred = sorted(paths)
    if folder_name is not None:
        folder_matches = [p for p in preferred if str(folder_name) in os.path.normpath(p).split(os.sep)]
        if folder_matches:
            preferred = folder_matches
    if label_name is not None:
        label_matches = [p for p in preferred if str(label_name) in os.path.normpath(p).split(os.sep)]
        if label_matches:
            preferred = label_matches

    return preferred[0] if len(preferred) == 1 else None


def gt_path_candidates(gt_dir: str, folder_name: Optional[str], fname: str, label_name: Optional[str] = None):
    gt_names = image_name_to_gt_txt_candidates(fname)
    candidates = []
    for gt_name in gt_names:
        if label_name is not None and folder_name is not None:
            candidates.append(os.path.join(gt_dir, str(label_name), str(folder_name), gt_name))
        if folder_name is not None:
            candidates.append(os.path.join(gt_dir, str(folder_name), gt_name))
        if label_name is not None:
            candidates.append(os.path.join(gt_dir, str(label_name), gt_name))
        candidates.append(os.path.join(gt_dir, gt_name))
    return gt_names, candidates


def find_gt_path(gt_dir: str, folder_name: Optional[str], fname: str, label_name: Optional[str] = None) -> Optional[str]:
    """Find matching GT txt for an original image filename or use an exact GT file path."""
    if os.path.isfile(gt_dir):
        return gt_dir

    gt_names, candidates = gt_path_candidates(gt_dir, folder_name, fname, label_name)

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Fallback for GT directories that have an extra nesting level. Only use it
    # when the basename resolves unambiguously after folder/label preference.
    index = build_gt_name_index(os.path.abspath(gt_dir))
    for gt_name in gt_names:
        indexed = choose_indexed_gt(index.get(gt_name.lower(), []), folder_name, label_name)
        if indexed is not None:
            return indexed
    return None


def describe_gt_lookup(gt_dir: str, folder_name: Optional[str], fname: str, label_name: Optional[str] = None) -> str:
    if os.path.isfile(gt_dir):
        return f'image={fname}	label={label_name}	folder={folder_name}	exact_gt_file={gt_dir}'
    gt_names, candidates = gt_path_candidates(gt_dir, folder_name, fname, label_name)
    tried = ' | '.join(candidates[:12])
    if len(candidates) > 12:
        tried += ' | ...'
    return (
        f'image={fname}	label={label_name}	folder={folder_name}	'
        f'gt_names={",".join(gt_names)}	tried={tried}'
    )


def write_missing_gt_log(path: str, records):
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for record in records:
            f.write(record + '\n')


def load_gt_boxes(gt_dir: str, folder_name: Optional[str], fname: str, label_name: Optional[str] = None):
    """
    Load GT boxes for an image.

    Expected txt format per line:
      x1 y1 x2 y2

    Returns:
      gt_boxes: list[(x1, y1, x2, y2)]
      gt_path: matched txt path or None
    """
    gt_path = find_gt_path(gt_dir, folder_name, fname, label_name)
    if gt_path is None:
        return [], None

    gt_boxes = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, parts[:4])
            except ValueError:
                continue
            gt_boxes.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))

    return gt_boxes, gt_path


def compute_iou(box_a, box_b) -> float:
    """Compute IoU for boxes in (x1, y1, x2, y2) format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return float(inter_area / union)


def match_boxes_for_f1(pred_boxes, gt_boxes, iou_threshold: float = 0.2):
    """
    One-to-one greedy matching for object detection F1.

    TP: prediction matched to one unmatched GT with IoU >= threshold.
    FP: prediction not matched to any GT.
    FN: GT not matched by any prediction.
    """
    pairs = []
    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            iou = compute_iou(pred, gt)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))

    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_preds = set()
    matched_gts = set()
    matches = []

    for iou, pi, gi in pairs:
        if pi in matched_preds or gi in matched_gts:
            continue
        matched_preds.add(pi)
        matched_gts.add(gi)
        matches.append((pi, gi, iou))

    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matches


def compute_prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def save_detection_eval_txt(out_dir: str, fname: str, gt_path: Optional[str], pred_boxes, gt_boxes, tp, fp, fn, matches):
    stem, _ = os.path.splitext(fname)
    eval_path = os.path.join(out_dir, f'{stem}_eval.txt')
    precision, recall, f1 = compute_prf(tp, fp, fn)

    with open(eval_path, 'w') as f:
        f.write(f'image: {fname}\n')
        f.write(f'gt_file: {gt_path if gt_path is not None else "None"}\n')
        f.write(f'TP: {tp}\n')
        f.write(f'FP: {fp}\n')
        f.write(f'FN: {fn}\n')
        f.write(f'precision: {precision:.6f}\n')
        f.write(f'recall: {recall:.6f}\n')
        f.write(f'f1: {f1:.6f}\n')
        f.write('\n[pred_boxes] x1 y1 x2 y2\n')
        for box in pred_boxes:
            f.write(f'{box[0]} {box[1]} {box[2]} {box[3]}\n')
        f.write('\n[gt_boxes] x1 y1 x2 y2\n')
        for box in gt_boxes:
            f.write(f'{box[0]} {box[1]} {box[2]} {box[3]}\n')
        f.write('\n[matches] pred_index gt_index iou\n')
        for pi, gi, iou in matches:
            f.write(f'{pi} {gi} {iou:.6f}\n')


def save_f1_summary(out_path: str, total_tp: int, total_fp: int, total_fn: int,
                    evaluated_images: int, missing_gt_images: int, matched_gt_images: int = 0,
                    empty_gt_files: int = 0, total_pred_boxes: int = 0, total_gt_boxes: int = 0):
    precision, recall, f1 = compute_prf(total_tp, total_fp, total_fn)
    with open(out_path, 'w') as f:
        f.write(f'evaluated_images: {evaluated_images}\n')
        f.write(f'matched_gt_images: {matched_gt_images}\n')
        f.write(f'missing_gt_images_skipped: {missing_gt_images}\n')
        f.write(f'empty_gt_files: {empty_gt_files}\n')
        f.write(f'total_pred_boxes: {total_pred_boxes}\n')
        f.write(f'total_gt_boxes: {total_gt_boxes}\n')
        f.write(f'TP: {total_tp}\n')
        f.write(f'FP: {total_fp}\n')
        f.write(f'FN: {total_fn}\n')
        f.write(f'micro_precision: {precision:.6f}\n')
        f.write(f'micro_recall: {recall:.6f}\n')
        f.write(f'micro_f1: {f1:.6f}\n')
    return precision, recall, f1


def save_heatmap_outputs(anomaly_map: np.ndarray,
                         out_dir: str,
                         fname: str,
                         save_size=(1920, 1080),
                         mask_dir: Optional[str] = None,
                         folder_name: Optional[str] = None,
                         mask_threshold: int = 10):
    """Save pure heatmap.

    If mask_dir/folder_name are given, the black mask area is forced to black
    AFTER resizing/colorizing the heatmap. This prevents interpolation from
    leaking heatmap colors outside the ROI boundary.
    """
    os.makedirs(out_dir, exist_ok=True)

    target_w, target_h = save_size

    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.ndim != 2:
        amap = np.squeeze(amap)

    # Normalize each image map to [0, 1] for visualization.
    amap = amap - np.nanmin(amap)
    denom = np.nanmax(amap) + 1e-8
    amap = amap / denom
    amap_u8 = (amap * 255).clip(0, 255).astype(np.uint8)
    amap_u8 = cv2.resize(amap_u8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # OpenCV colormap is BGR, convert to RGB for PIL saving.
    heatmap_bgr = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    heatmap_pil = Image.fromarray(heatmap_rgb)

    stem, ext = os.path.splitext(fname)
    ext = ext if ext else '.jpg'
    heatmap_pil.save(os.path.join(out_dir, f"{stem}_heatmap{ext}"))

def save_outputs(img_tensor: torch.Tensor,
                 anomaly_map: np.ndarray,
                 out_dir: str,
                 fname: str,
                 threshold: float,
                 min_area: int,
                 save_size=(1920, 1080),
                 save_heatmap: bool = True,
                 original_image_path: Optional[str] = None,
                 mask_dir: Optional[str] = None,
                 folder_name: Optional[str] = None,
                 mask_threshold: int = 10,
                 bbox_score_mode: str = 'raw',
                 save_score_map: bool = False):
    """Save bbox image and heatmap image together in the same output folder.

    If original_image_path is given, bboxes are drawn on the original unmasked test image.
    The model input remains unmasked; bbox visualization is drawn on the original image.
    """
    os.makedirs(out_dir, exist_ok=True)

    if original_image_path is not None:
        img_pil = Image.open(original_image_path).convert('RGB')
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_u8 = ((img_tensor.cpu() * std + mean).clamp(0, 1) * 255).byte()
        img_pil = Image.fromarray(img_u8.permute(1, 2, 0).numpy())

    score_for_bbox = bbox_score_map(anomaly_map, bbox_score_mode)
    bboxes = anomaly_map_to_bboxes(score_for_bbox, threshold=threshold, min_area=min_area)

    target_w, target_h = save_size
    resized_img = img_pil.resize((target_w, target_h), Image.BILINEAR)

    # anomaly_map is produced at model input resolution, usually 512x512.
    amap = np.asarray(anomaly_map)
    if amap.ndim != 2:
        amap = np.squeeze(amap)
    map_h, map_w = amap.shape[-2], amap.shape[-1]

    scale_x = target_w / map_w
    scale_y = target_h / map_h
    scaled_bboxes = [
        (int(x0 * scale_x), int(y0 * scale_y), int(x1 * scale_x), int(y1 * scale_y))
        for x0, y0, x1, y1 in bboxes
    ]

    stem, ext = os.path.splitext(fname)
    ext = ext if ext else '.jpg'

    if save_score_map:
        save_score_maps(out_dir, stem, anomaly_map, score_for_bbox, bbox_score_mode, threshold)

    boxed = draw_bboxes_on_image(resized_img, scaled_bboxes, color='red', width=6)
    boxed.save(os.path.join(out_dir, f"{stem}_bbox{ext}"))

    # Save predicted bbox coordinates used for drawing/evaluation.
    pred_txt_path = os.path.join(out_dir, f"{stem}_pred_boxes.txt")
    with open(pred_txt_path, 'w') as f:
        for x1, y1, x2, y2 in scaled_bboxes:
            f.write(f"{x1} {y1} {x2} {y2}\n")

    if save_heatmap:
        save_heatmap_outputs(
            anomaly_map=anomaly_map,
            out_dir=out_dir,
            fname=fname,
            save_size=save_size,
            mask_dir=mask_dir,
            folder_name=folder_name,
            mask_threshold=mask_threshold,
        )

    return scaled_bboxes



@torch.no_grad()
def get_final_localization_map(cfg, extractor, parallel_flows, fusion_flow, rf_model, imgs, rf_steps: int):
    _, z_fused_list, _ = msflow_forward(cfg, extractor, parallel_flows, fusion_flow, imgs, return_pre_fusion=True)

    size_list = [list(z.shape[-2:]) for z in z_fused_list]
    outputs_list = []
    for z in z_fused_list:
        logp = -0.5 * torch.mean(z ** 2, dim=1)
        outputs_list.append([logp])
    _, anomaly_score_map_add, _ = post_process(cfg, size_list, outputs_list)

    z_rf_in = [z_fused_list[i] for i in [0, 1]]
    z_fused_rect_l01 = rf_transport(rf_model, z_rf_in, steps=rf_steps)

    diff_maps = []
    for lvl in [0, 1]:
        z = z_fused_list[lvl]
        zr = z_fused_rect_l01[lvl]
        logp = -0.5 * torch.mean(z ** 2, dim=1)
        logp_r = -0.5 * torch.mean(zr ** 2, dim=1)
        diff_maps.append((logp_r - logp).detach().cpu())

    d0 = diff_maps[0].unsqueeze(1)
    d1 = diff_maps[1].unsqueeze(1)
    if d0.shape[-2:] != tuple(cfg.input_size):
        d0 = F.interpolate(d0, size=cfg.input_size, mode='bilinear', align_corners=False)
    if d1.shape[-2:] != tuple(cfg.input_size):
        d1 = F.interpolate(d1, size=cfg.input_size, mode='bilinear', align_corners=False)

    rf_map = minmax_norm((d0[:, 0] + d1[:, 0]).numpy())
    return rf_map + anomaly_score_map_add


def setup_cfg(args, folder_name: Optional[str] = None):
    c.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c.input_size = (512, 512)
    c.img_mean = [0.485, 0.456, 0.406]
    c.img_std = [0.229, 0.224, 0.225]
    c.extractor = args.extractor
    c.pool_type = args.pool_type
    c.parallel_blocks = args.parallel_blocks
    c.c_conds = args.c_conds
    c.clamp_alpha = args.clamp_alpha
    c.dataset = 'posco'
    c.class_name = 'posco'
    c.posco_train_subdir = None
    if folder_name is not None:
        c.class_name = folder_name
        c.posco_train_subdir = folder_name
    return c


def discover_folder_names(args) -> List[str]:
    if args.folder_names:
        return list(args.folder_names)

    explicit_msflow = getattr(args, 'explicit_msflow_ckpt', False)
    explicit_rf = getattr(args, 'explicit_rf_ckpt', False)

    msflow_base = os.path.join(args.msflow_work_dir, args.msflow_version, 'posco')
    rf_base = os.path.join(args.rf_work_dir, args.rf_version, 'posco')
    normal_base = os.path.join(args.data_root, 'normal')
    abnormal_base = os.path.join(args.data_root, 'abnormal')

    # Discover folders from only the paths that are needed. Explicit checkpoint
    # paths are global inputs, so they should not constrain folder discovery.
    candidate_sets = []
    discovery_bases = []
    if not explicit_msflow:
        discovery_bases.append(('MSFlow checkpoints', msflow_base))
    if not explicit_rf:
        discovery_bases.append(('RF checkpoints', rf_base))
    discovery_bases.append(('abnormal test folders', abnormal_base))

    for label, base in discovery_bases:
        if os.path.isdir(base):
            candidate_sets.append({d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))})
        else:
            print(f"[Warning] {label} folder not found while discovering subfolders: {base}")

    if not candidate_sets:
        raise FileNotFoundError('Could not discover any folder names. Use --folder-names 01 02 ...')

    folder_names = sorted(set.intersection(*candidate_sets)) if len(candidate_sets) > 1 else sorted(candidate_sets[0])

    valid = []
    skipped = []
    for folder in folder_names:
        msflow_ckpt = args.msflow_ckpt if explicit_msflow else os.path.join(msflow_base, folder, args.msflow_ckpt_name)
        rf_ckpt = args.rf_ckpt if explicit_rf else os.path.join(rf_base, folder, args.rf_ckpt_name)
        has_normal = os.path.isdir(os.path.join(normal_base, folder))
        has_abnormal = os.path.isdir(os.path.join(abnormal_base, folder))
        if os.path.isfile(msflow_ckpt) and os.path.isfile(rf_ckpt) and (has_normal or has_abnormal):
            valid.append(folder)
        else:
            skipped.append((folder, msflow_ckpt, rf_ckpt, has_normal, has_abnormal))

    if skipped:
        print('[Warning] Some folders were skipped because checkpoint or test folder was missing:')
        for folder, ms_ckpt, rf_ckpt, has_normal, has_abnormal in skipped:
            print(f"  - {folder}: msflow={os.path.isfile(ms_ckpt)}, rf={os.path.isfile(rf_ckpt)}, "
                  f"normal={has_normal}, abnormal={has_abnormal}")

    if not valid:
        raise FileNotFoundError('No valid folders found with both checkpoints and test images.')
    return valid

def run_one_folder(args, folder_name: str):
    cfg = setup_cfg(args, folder_name)

    if getattr(args, 'explicit_msflow_ckpt', False):
        msflow_ckpt = args.msflow_ckpt
    else:
        msflow_ckpt = os.path.join(
            args.msflow_work_dir, args.msflow_version, 'posco', folder_name, args.msflow_ckpt_name
        )

    if getattr(args, 'explicit_rf_ckpt', False):
        rf_ckpt = args.rf_ckpt
    else:
        rf_ckpt = os.path.join(
            args.rf_work_dir, args.rf_version, 'posco', folder_name, args.rf_ckpt_name
        )

    assert os.path.isfile(msflow_ckpt), f"MSFlow checkpoint not found: {msflow_ckpt}"
    assert os.path.isfile(rf_ckpt), f"RF checkpoint not found: {rf_ckpt}"

    dataset = PoscoTestFolderDataset(
        args.data_root,
        folder_name=folder_name,
        input_size=cfg.input_size,
        img_mean=cfg.img_mean,
        img_std=cfg.img_std,
        apply_test_mask=args.apply_test_mask,
        mask_dir=args.mask_dir,
        mask_threshold=args.mask_threshold,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print('\n' + '=' * 80)
    print(f"[INFO] Visualize folder: {folder_name}")
    print(f"[INFO] Test images: {len(dataset)}")
    print(f"[INFO] MSFlow: {msflow_ckpt}")
    print(f"[INFO] RF:     {rf_ckpt}")
    print('=' * 80)

    extractor, parallel_flows, fusion_flow = build_msflow(cfg, msflow_ckpt)

    print('[INFO] Initializing RF model...')
    init_imgs, *_ = next(iter(loader))
    init_imgs = init_imgs.to(cfg.device, non_blocking=True)
    with torch.no_grad():
        _, z_fused_list, _ = msflow_forward(cfg, extractor, parallel_flows, fusion_flow, init_imgs, return_pre_fusion=True)
    rf_model = build_rf_from_batch(cfg.device, rf_ckpt, z_fused_list, args.rf_tdims, args.rf_depths)

    seen_dirs = set()
    total_processed = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    evaluated_images = 0
    missing_gt_images = 0
    missing_gt_records = []
    matched_gt_images = 0
    empty_gt_files = 0
    total_pred_boxes = 0
    total_gt_boxes = 0

    for imgs, img_paths, label_names, folder_names, fnames in loader:
        imgs = imgs.to(cfg.device, non_blocking=True)
        final_maps = get_final_localization_map(cfg, extractor, parallel_flows, fusion_flow, rf_model, imgs, args.rf_steps)
        total_processed += imgs.shape[0]

        for b in range(imgs.shape[0]):
            fname = fnames[b]
            final_map = final_maps[b]
            if torch.is_tensor(final_map):
                final_map = final_map.detach().cpu().numpy()

            # After inference, force anomaly score to 0 in the black masked-out area before bbox generation.
            if args.apply_test_mask:
                final_map = apply_keep_mask_to_anomaly_map(
                    anomaly_map=final_map,
                    mask_dir=args.mask_dir,
                    folder_name=folder_name,
                    threshold=args.mask_threshold,
                    erode_pixels=args.mask_erode_pixels,
                    close_kernel=args.mask_close_kernel,
                )

            # Save separately to avoid name collision between normal/abnormal.
            out_dir = os.path.join(args.output_dir, folder_name, label_names[b])
            if out_dir not in seen_dirs:
                os.makedirs(out_dir, exist_ok=True)
                seen_dirs.add(out_dir)

            pred_boxes = save_outputs(
                img_tensor=imgs[b],
                anomaly_map=final_map,
                out_dir=out_dir,
                fname=fname,
                threshold=args.threshold,
                min_area=args.min_area,
                save_heatmap=args.save_heatmap,
                original_image_path=img_paths[b],
                mask_dir=args.mask_dir if args.apply_test_mask else None,
                folder_name=folder_name if args.apply_test_mask else None,
                mask_threshold=args.mask_threshold,
                bbox_score_mode=args.bbox_score_mode,
                save_score_map=args.save_score_map,
            )

            if args.eval_f1:
                total_pred_boxes += len(pred_boxes)
                gt_source = args.gt_path if args.gt_path is not None else args.gt_dir
                gt_boxes, gt_path = load_gt_boxes(gt_source, folder_name, fname, label_names[b])
                if gt_path is None:
                    missing_gt_images += 1
                    missing_gt_records.append(describe_gt_lookup(gt_source, folder_name, fname, label_names[b]))
                    if args.count_missing_gt_as_empty:
                        tp, fp, fn, matches = match_boxes_for_f1(
                            pred_boxes, [], iou_threshold=args.iou_threshold
                        )
                        total_tp += tp
                        total_fp += fp
                        total_fn += fn
                        evaluated_images += 1
                        save_detection_eval_txt(out_dir, fname, gt_path, pred_boxes, [], tp, fp, fn, matches)
                else:
                    matched_gt_images += 1
                    total_gt_boxes += len(gt_boxes)
                    if len(gt_boxes) == 0:
                        empty_gt_files += 1
                    tp, fp, fn, matches = match_boxes_for_f1(
                        pred_boxes, gt_boxes, iou_threshold=args.iou_threshold
                    )
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    evaluated_images += 1
                    save_detection_eval_txt(out_dir, fname, gt_path, pred_boxes, gt_boxes, tp, fp, fn, matches)


    if args.eval_f1:
        summary_path = os.path.join(args.output_dir, folder_name, 'f1_summary.txt')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        write_missing_gt_log(os.path.join(args.output_dir, folder_name, 'missing_gt.txt'), missing_gt_records)
        precision, recall, f1 = save_f1_summary(
            summary_path, total_tp, total_fp, total_fn, evaluated_images, missing_gt_images,
            matched_gt_images, empty_gt_files, total_pred_boxes, total_gt_boxes
        )
        print(f'[F1] folder={folder_name} TP={total_tp} FP={total_fp} FN={total_fn} '
              f'precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} '
              f'evaluated_images={evaluated_images} matched_gt={matched_gt_images} '
              f'missing_gt={missing_gt_images} pred_boxes={total_pred_boxes} gt_boxes={total_gt_boxes}')

    del extractor, parallel_flows, fusion_flow, rf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_model(args):
    cfg = setup_cfg(args, None)
    assert os.path.isfile(args.msflow_ckpt), f"MSFlow checkpoint not found: {args.msflow_ckpt}"
    assert os.path.isfile(args.rf_ckpt), f"RF checkpoint not found: {args.rf_ckpt}"

    dataset = PoscoFlatTestDataset(
        args.data_root,
        input_size=cfg.input_size,
        img_mean=cfg.img_mean,
        img_std=cfg.img_std,
        apply_test_mask=args.apply_test_mask,
        mask_dir=args.mask_dir,
        mask_threshold=args.mask_threshold,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    print(f"[INFO] Found {len(dataset)} images under {args.data_root}")
    extractor, parallel_flows, fusion_flow = build_msflow(cfg, args.msflow_ckpt)

    print('[INFO] Initializing RF model...')
    init_imgs, _, _, _, _ = next(iter(loader))
    init_imgs = init_imgs.to(cfg.device, non_blocking=True)
    with torch.no_grad():
        _, z_fused_list, _ = msflow_forward(cfg, extractor, parallel_flows, fusion_flow, init_imgs, return_pre_fusion=True)
    rf_model = build_rf_from_batch(cfg.device, args.rf_ckpt, z_fused_list, args.rf_tdims, args.rf_depths)

    seen_dirs = set()
    total_processed = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    evaluated_images = 0
    missing_gt_images = 0
    missing_gt_records = []
    matched_gt_images = 0
    empty_gt_files = 0
    total_pred_boxes = 0
    total_gt_boxes = 0

    for imgs, img_paths, label_names, folder_names, fnames in loader:
        imgs = imgs.to(cfg.device, non_blocking=True)
        final_maps = get_final_localization_map(cfg, extractor, parallel_flows, fusion_flow, rf_model, imgs, args.rf_steps)
        total_processed += imgs.shape[0]

        for b in range(imgs.shape[0]):
            final_map = final_maps[b]
            if torch.is_tensor(final_map):
                final_map = final_map.detach().cpu().numpy()

            # After inference, force anomaly score to 0 in the black masked-out area before bbox generation.
            if args.apply_test_mask:
                final_map = apply_keep_mask_to_anomaly_map(
                    anomaly_map=final_map,
                    mask_dir=args.mask_dir,
                    folder_name=folder_names[b],
                    threshold=args.mask_threshold,
                    erode_pixels=args.mask_erode_pixels,
                    close_kernel=args.mask_close_kernel,
                )

            out_dir = os.path.join(args.output_dir, label_names[b])
            if out_dir not in seen_dirs:
                os.makedirs(out_dir, exist_ok=True)
                seen_dirs.add(out_dir)

            pred_boxes = save_outputs(
                imgs[b],
                final_map,
                out_dir,
                fnames[b],
                args.threshold,
                args.min_area,
                save_heatmap=args.save_heatmap,
                original_image_path=img_paths[b],
                mask_dir=args.mask_dir if args.apply_test_mask else None,
                folder_name=folder_names[b] if args.apply_test_mask else None,
                mask_threshold=args.mask_threshold,
                bbox_score_mode=args.bbox_score_mode,
                save_score_map=args.save_score_map,
            )

            if args.eval_f1:
                total_pred_boxes += len(pred_boxes)
                eval_folder_name = folder_names[b]
                gt_fname = os.path.basename(img_paths[b])
                gt_source = args.gt_path if args.gt_path is not None else args.gt_dir
                gt_boxes, gt_path = load_gt_boxes(gt_source, eval_folder_name, gt_fname, label_names[b])
                if gt_path is None:
                    missing_gt_images += 1
                    missing_gt_records.append(describe_gt_lookup(gt_source, eval_folder_name, gt_fname, label_names[b]))

                    # Only normal images without GT are treated as empty/no-object images.
                    # Abnormal images without GT are skipped because GT annotation is incomplete.
                    if label_names[b] == "normal":
                        tp, fp, fn, matches = match_boxes_for_f1(
                            pred_boxes, [], iou_threshold=args.iou_threshold
                        )
                        total_tp += tp
                        total_fp += fp
                        total_fn += fn
                        evaluated_images += 1
                        save_detection_eval_txt(
                            out_dir, fnames[b], gt_path, pred_boxes, [], tp, fp, fn, matches
                        )
                    else:
                        # abnormal image without GT annotation -> skip
                        pass

                else:
                    matched_gt_images += 1
                    total_gt_boxes += len(gt_boxes)
                    if len(gt_boxes) == 0:
                        empty_gt_files += 1
                    tp, fp, fn, matches = match_boxes_for_f1(
                        pred_boxes, gt_boxes, iou_threshold=args.iou_threshold
                    )
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    evaluated_images += 1
                    save_detection_eval_txt(
                        out_dir, fnames[b], gt_path, pred_boxes, gt_boxes, tp, fp, fn, matches
                    )

    if args.eval_f1:
        summary_path = os.path.join(args.output_dir, 'f1_summary.txt')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        write_missing_gt_log(os.path.join(args.output_dir, 'missing_gt.txt'), missing_gt_records)
        precision, recall, f1 = save_f1_summary(
            summary_path, total_tp, total_fp, total_fn, evaluated_images, missing_gt_images,
            matched_gt_images, empty_gt_files, total_pred_boxes, total_gt_boxes
        )
        print(f'[F1] TP={total_tp} FP={total_fp} FN={total_fn} '
              f'precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} '
              f'evaluated_images={evaluated_images} matched_gt={matched_gt_images} '
              f'missing_gt={missing_gt_images} pred_boxes={total_pred_boxes} gt_boxes={total_gt_boxes}')


def main():
    parser = argparse.ArgumentParser(description='Visualize POSCO bounding boxes from MSFlow+RF localization map')
    parser.add_argument('--data_root', '--data-root', dest='data_root', type=str, default='./data/posco/test',
                        help='POSCO test root containing normal/*.jpg and abnormal/*.jpg')
    parser.add_argument('--output_dir', '--output-dir', dest='output_dir', type=str, default='./results2',
                        help='Where to save images with bounding boxes')
    parser.add_argument('--save_heatmap', action='store_true', default=True,
                        help='Save pure heatmap image next to bbox image in the same folder. Default: True')
    parser.add_argument('--no_save_heatmap', dest='save_heatmap', action='store_false',
                        help='Disable heatmap saving')
    parser.add_argument('--apply-test-mask', action='store_true', default=True,
                        help='Use folder-specific mask only after inference: bbox/anomaly scores outside ROI are suppressed. Model input remains the original image.')
    parser.add_argument('--mask-dir', '--mask_dir', dest='mask_dir', type=str, default='./mask',
                        help='Directory containing 02_mask.jpg, 04_mask.jpg, ...')
    parser.add_argument('--mask-threshold', type=int, default=10,
                        help='Pixels <= threshold in mask are treated as black masked-out area.')
    parser.add_argument('--mask-erode-pixels', type=int, default=0,
                        help='Shrink white ROI before bbox generation. Use 0 to keep ROI unchanged.')
    parser.add_argument('--mask-close-kernel', type=int, default=7,
                        help='Fill small black holes inside white ROI. Use 0 to disable.')
    parser.add_argument('--gt-dir', '--gt_dir', dest='gt_dir', type=str, default='./gt',
                        help='Directory containing GT txt files. Supports either gt/<name>.txt or gt/<folder>/<name>.txt.')
    parser.add_argument('--gt-path', '--gt_path', dest='gt_path', type=str, default=None,
                        help='Exact GT txt file path. If set, this file is used directly instead of matching from --gt-dir.')
    parser.add_argument('--eval-f1', action='store_true', default=False,
                        help='Compute micro precision/recall/F1 using GT txt files and predicted boxes.')
    parser.add_argument('--iou-threshold', type=float, default=0.2,
                        help='IoU threshold for TP matching. Default: 0.2')
    parser.add_argument('--count-missing-gt-as-empty', action='store_true', default=False,
                        help='If enabled, images without matching GT txt are treated as no-object images; predictions become FP. Default: skip missing GT images.')

    # Old single-model mode arguments.
    parser.add_argument('--msflow_ckpt', '--msflow-ckpt', dest='msflow_ckpt', type=str,
                        default='work_dirs/msflow_wide_resnet50_2_avgpool_pl258/posco/posco/last.pt')
    parser.add_argument('--rf_ckpt', '--rf-ckpt', dest='rf_ckpt', type=str,
                        default='work_dirs/rf_on_msflow_wide_resnet50_2_avgpool_pl258/posco/posco/rf_last.pt')

    # New folder-by-folder mode arguments.
    parser.add_argument('--visualize-by-folder', action='store_true', default=False,
                        help='Run each POSCO subfolder with its matching MSFlow and RF checkpoints.')
    parser.add_argument('--folder-names', type=str, nargs='+', default=None,
                        help='Optional folder names to run, e.g., --folder-names 01 02 05. If omitted, auto-discover.')
    parser.add_argument('--msflow-work-dir', '--msflow_work_dir', dest='msflow_work_dir', type=str, default='work_dirs')
    parser.add_argument('--msflow-version', type=str, default='msflow_wide_resnet50_2_avgpool_pl258')
    parser.add_argument('--msflow-ckpt-name', type=str, default='last.pt')
    parser.add_argument('--rf-work-dir', '--rf_work_dir', dest='rf_work_dir', type=str, default='work_dirs')
    parser.add_argument('--rf-version', type=str, default='rf_on_msflow_wide_resnet50_2_avgpool_pl258')
    parser.add_argument('--rf-ckpt-name', type=str, default='rf_last.pt')

    parser.add_argument('--threshold', type=float, default=2.5)
    parser.add_argument('--bbox-score-mode', '--bbox_score_mode', dest='bbox_score_mode',
                        choices=['raw', 'minmax'], default='raw',
                        help='Score map used for bbox thresholding. raw keeps legacy absolute scores; minmax uses per-image 0..1 scores like the heatmap.')
    parser.add_argument('--save-score-map', '--save_score_map', dest='save_score_map', action='store_true', default=False,
                        help='Save raw and bbox score maps as .npy plus per-image score stats.')
    parser.add_argument('--min_area', type=int, default=80,
                        help='Minimum connected region area to keep')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rf_steps', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--rf-tdims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--rf-depths', type=int, nargs='+', default=[3, 3])
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str)
    parser.add_argument('--pool-type', default='avg', type=str)
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, nargs='+')
    parser.add_argument('--c-conds', default=[64, 64, 64], type=int, nargs='+')
    parser.add_argument('--clamp-alpha', default=1.9, type=float)
    raw_argv = sys.argv[1:]
    args = parser.parse_args()
    args.explicit_msflow_ckpt = option_was_provided(raw_argv, '--msflow_ckpt', '--msflow-ckpt')
    args.explicit_rf_ckpt = option_was_provided(raw_argv, '--rf_ckpt', '--rf-ckpt')
    args.explicit_threshold = option_was_provided(raw_argv, '--threshold')
    if args.bbox_score_mode == 'minmax' and not args.explicit_threshold:
        args.threshold = 0.5
    for path_attr in (
        'data_root', 'output_dir', 'mask_dir', 'gt_dir', 'gt_path',
        'msflow_ckpt', 'rf_ckpt', 'msflow_work_dir', 'rf_work_dir',
    ):
        setattr(args, path_attr, expand_path_string(getattr(args, path_attr)))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.visualize_by_folder:
        folder_names = discover_folder_names(args)
        print(f"[INFO] visualize-by-folder enabled. Found {len(folder_names)} folder(s): {folder_names}")
        for folder_name in folder_names:
            run_one_folder(args, folder_name)
    else:
        run_single_model(args)


if __name__ == '__main__':
    main()
