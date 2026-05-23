from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
PRED_SUFFIX = '_pred_boxes.txt'
Box = Tuple[int, int, int, int]


def expand_path(path: Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def strip_frame_suffix(stem: str) -> str:
    return re.sub(r'_\d+$', '', stem)


def gt_file_name(image_path: Path, sequence_level: bool) -> str:
    stem = image_path.stem
    if sequence_level:
        stem = strip_frame_suffix(stem)
    return stem + '.txt'


def safe_int_box(values: List[float]) -> Optional[Box]:
    if len(values) < 4:
        return None
    x1, y1, x2, y2 = values[:4]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def read_pred_boxes(path: Path) -> List[Box]:
    boxes: List[Box] = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = []
            for token in line.replace(',', ' ').split():
                try:
                    values.append(float(token))
                except ValueError:
                    continue
            box = safe_int_box(values)
            if box is not None:
                boxes.append(box)
    return boxes


def write_gt_boxes(path: Path, boxes: List[Box], overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for x1, y1, x2, y2 in boxes:
            f.write(f'{x1} {y1} {x2} {y2}\n')
    return True


def output_gt_dir(gt_out_dir: Path, label: str, folder: Optional[str], layout: str) -> Path:
    if layout == 'label-folder':
        out = gt_out_dir / label
        if folder:
            out = out / folder
        return out
    if layout == 'folder':
        return gt_out_dir / folder if folder else gt_out_dir
    if layout == 'flat':
        return gt_out_dir
    raise ValueError(f'Unsupported layout: {layout}')


def iter_label_images(data_root: Path, label: str) -> Iterable[Tuple[Path, Optional[str], str]]:
    label_dir = data_root / label
    if not label_dir.is_dir():
        raise FileNotFoundError(f'label directory not found: {label_dir}')

    for image_path in sorted(label_dir.rglob('*')):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel_dir = image_path.parent.relative_to(label_dir)
        folder = None if rel_dir == Path('.') else rel_dir.parts[0]
        rel_prefix = '' if rel_dir == Path('.') else '_'.join(rel_dir.parts) + '_'
        save_stem = Path(rel_prefix + image_path.name).stem
        yield image_path, folder, save_stem


def pred_candidates_for_image(results_dir: Path, label: str, folder: Optional[str], image_path: Path, save_stem: str) -> List[Path]:
    orig_stem = image_path.stem
    candidates = [results_dir / label / f'{save_stem}{PRED_SUFFIX}']
    if folder:
        candidates.extend([
            results_dir / folder / label / f'{orig_stem}{PRED_SUFFIX}',
            results_dir / folder / label / f'{save_stem}{PRED_SUFFIX}',
        ])
    candidates.append(results_dir / label / f'{orig_stem}{PRED_SUFFIX}')

    seen = set()
    unique = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def find_pred_for_image(results_dir: Path, label: str, folder: Optional[str], image_path: Path, save_stem: str) -> Optional[Path]:
    for candidate in pred_candidates_for_image(results_dir, label, folder, image_path, save_stem):
        if candidate.is_file():
            return candidate
    return None


def infer_folder_from_pred(results_dir: Path, pred_path: Path, label: str) -> Optional[str]:
    rel = pred_path.relative_to(results_dir)
    parts = rel.parts
    if len(parts) >= 3 and parts[-2] == label:
        return parts[-3]
    return None


def iter_pred_files(results_dir: Path, label: str) -> Iterable[Tuple[Path, Optional[str], str]]:
    for pred_path in sorted(results_dir.rglob(f'*{PRED_SUFFIX}')):
        parts = pred_path.relative_to(results_dir).parts
        if len(parts) < 2 or parts[-2] != label:
            continue
        stem = pred_path.name[:-len(PRED_SUFFIX)]
        folder = infer_folder_from_pred(results_dir, pred_path, label)
        yield pred_path, folder, stem


def build_from_data_root(args) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    missing: List[str] = []

    for image_path, folder, save_stem in iter_label_images(args.data_root, args.label):
        pred_path = find_pred_for_image(args.results_dir, args.label, folder, image_path, save_stem)
        if pred_path is None:
            tried = ' | '.join(str(p) for p in pred_candidates_for_image(args.results_dir, args.label, folder, image_path, save_stem))
            missing.append(f'{image_path}\tfolder={folder}\ttried={tried}')
            continue

        boxes = read_pred_boxes(pred_path)
        if args.skip_empty and not boxes:
            continue

        gt_dir = output_gt_dir(args.gt_out_dir, args.label, folder, args.layout)
        gt_path = gt_dir / gt_file_name(image_path, args.sequence_level)
        written = write_gt_boxes(gt_path, boxes, args.overwrite)
        rows.append({
            'image_path': str(image_path),
            'pred_path': str(pred_path),
            'gt_path': str(gt_path),
            'label': args.label,
            'folder': folder or '',
            'box_count': len(boxes),
            'written': int(written),
            'mode': 'data_root',
        })

    return rows, missing


def build_from_results_only(args) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    missing: List[str] = []

    for pred_path, folder, stem in iter_pred_files(args.results_dir, args.label):
        boxes = read_pred_boxes(pred_path)
        if args.skip_empty and not boxes:
            continue

        gt_dir = output_gt_dir(args.gt_out_dir, args.label, folder, args.layout)
        gt_name = strip_frame_suffix(stem) + '.txt' if args.sequence_level else stem + '.txt'
        gt_path = gt_dir / gt_name
        written = write_gt_boxes(gt_path, boxes, args.overwrite)
        rows.append({
            'image_path': '',
            'pred_path': str(pred_path),
            'gt_path': str(gt_path),
            'label': args.label,
            'folder': folder or '',
            'box_count': len(boxes),
            'written': int(written),
            'mode': 'results_only',
        })

    return rows, missing


def write_manifest(gt_out_dir: Path, rows: List[dict], missing: List[str]) -> None:
    gt_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = gt_out_dir / 'manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['image_path', 'pred_path', 'gt_path', 'label', 'folder', 'box_count', 'written', 'mode'],
        )
        writer.writeheader()
        writer.writerows(rows)

    if missing:
        (gt_out_dir / 'missing_pred_boxes.txt').write_text('\n'.join(missing) + '\n', encoding='utf-8')


def print_summary(rows: List[dict], missing: List[str], gt_out_dir: Path) -> None:
    total = len(rows)
    written = sum(int(row['written']) for row in rows)
    empty = sum(1 for row in rows if int(row['box_count']) == 0)
    total_boxes = sum(int(row['box_count']) for row in rows)
    print(f'gt_out_dir: {gt_out_dir}')
    print(f'gt_files_seen: {total}')
    print(f'gt_files_written: {written}')
    print(f'empty_gt_files: {empty}')
    print(f'total_boxes: {total_boxes}')
    print(f'missing_pred_files: {len(missing)}')
    print(f'manifest: {gt_out_dir / "manifest.csv"}')
    if missing:
        print(f'missing_pred_log: {gt_out_dir / "missing_pred_boxes.txt"}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create POSCO GT txt files from visualize_bboxes.py abnormal *_pred_boxes.txt outputs.'
    )
    parser.add_argument('--results-dir', '--results_dir', dest='results_dir', type=Path, required=True,
                        help='visualize_bboxes.py output directory containing abnormal *_pred_boxes.txt files.')
    parser.add_argument('--gt-out-dir', '--gt_out_dir', dest='gt_out_dir', type=Path, required=True,
                        help='New GT directory to create.')
    parser.add_argument('--data-root', '--data_root', dest='data_root', type=Path, default=None,
                        help='Optional POSCO test root. If set, GT filenames are generated from original abnormal image names.')
    parser.add_argument('--label', type=str, default='abnormal', choices=['abnormal', 'normal'],
                        help='Which label output to convert. Default: abnormal.')
    parser.add_argument('--layout', choices=['label-folder', 'folder', 'flat'], default='label-folder',
                        help='GT output layout. Default creates gt_out/abnormal/<folder>/<image>.txt.')
    parser.add_argument('--sequence-level', action='store_true', default=False,
                        help='Strip trailing frame suffix like _000000 from GT txt names.')
    parser.add_argument('--skip-empty', action='store_true', default=False,
                        help='Skip GT files when the predicted box file has no boxes. Default writes empty txt files.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing GT txt files.')
    args = parser.parse_args()

    args.results_dir = expand_path(args.results_dir)
    args.gt_out_dir = expand_path(args.gt_out_dir)
    if args.data_root is not None:
        args.data_root = expand_path(args.data_root)

    if not args.results_dir.is_dir():
        raise FileNotFoundError(f'results-dir not found: {args.results_dir}')
    if args.data_root is not None and not args.data_root.is_dir():
        raise FileNotFoundError(f'data-root not found: {args.data_root}')

    if args.data_root is not None:
        rows, missing = build_from_data_root(args)
    else:
        rows, missing = build_from_results_only(args)

    write_manifest(args.gt_out_dir, rows, missing)
    print_summary(rows, missing, args.gt_out_dir)


if __name__ == '__main__':
    main()
