from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
Box = Tuple[int, int, int, int]


def expand_path(path: Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def strip_frame_suffix(stem: str) -> str:
    return re.sub(r'_\d+$', '', stem)


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def parse_box_line(line: str) -> Optional[Box]:
    values = []
    for token in line.strip().replace(',', ' ').split():
        try:
            values.append(float(token))
        except ValueError:
            continue
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


def valid_box_count(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if parse_box_line(line) is not None:
                count += 1
    return count


def file_has_boxes(path: Path) -> bool:
    return valid_box_count(path) > 0


def build_name_index(source_gt_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(source_gt_dir.rglob('*.txt')):
        if file_has_boxes(path):
            index[path.name.lower()].append(path)
    return index


def target_gt_path(gt_dir: Path, label: str, folder: Optional[str], image_path: Path, layout: str, sequence_level: bool) -> Path:
    stem = strip_frame_suffix(image_path.stem) if sequence_level else image_path.stem
    name = stem + '.txt'
    if layout == 'label-folder':
        out = gt_dir / label
        if folder:
            out = out / folder
        return out / name
    if layout == 'folder':
        return (gt_dir / folder / name) if folder else (gt_dir / name)
    if layout == 'flat':
        return gt_dir / name
    raise ValueError(f'Unsupported layout: {layout}')


def iter_expected_targets_from_data(data_root: Path, target_gt_dir: Path, label: str, layout: str, sequence_level: bool):
    label_dir = data_root / label
    if not label_dir.is_dir():
        raise FileNotFoundError(f'label directory not found: {label_dir}')
    for image_path in sorted(label_dir.rglob('*')):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel_dir = image_path.parent.relative_to(label_dir)
        folder = None if rel_dir == Path('.') else rel_dir.parts[0]
        yield target_gt_path(target_gt_dir, label, folder, image_path, layout, sequence_level), label, folder


def infer_label_folder_from_target(target_gt_dir: Path, target_path: Path, labels: List[str]) -> Tuple[Optional[str], Optional[str]]:
    rel = target_path.relative_to(target_gt_dir)
    parts = rel.parts
    label = None
    folder = None
    if parts and parts[0] in labels:
        label = parts[0]
        if len(parts) >= 3:
            folder = parts[1]
    elif len(parts) >= 2:
        folder = parts[0]
    return label, folder


def iter_existing_targets(target_gt_dir: Path, labels: List[str]):
    for target_path in sorted(target_gt_dir.rglob('*.txt')):
        label, folder = infer_label_folder_from_target(target_gt_dir, target_path, labels)
        if label is not None and label not in labels:
            continue
        yield target_path, label, folder


def source_candidates(source_gt_dir: Path, target_gt_dir: Path, target_path: Path, label: Optional[str], folder: Optional[str], index) -> List[Path]:
    rel = target_path.relative_to(target_gt_dir)
    name = target_path.name
    stripped_name = strip_frame_suffix(target_path.stem) + '.txt'
    names = [name]
    if stripped_name not in names:
        names.append(stripped_name)

    candidates: List[Path] = []
    candidates.append(source_gt_dir / rel)
    for candidate_name in names:
        if label and folder:
            candidates.append(source_gt_dir / label / folder / candidate_name)
        if folder:
            candidates.append(source_gt_dir / folder / candidate_name)
        if label:
            candidates.append(source_gt_dir / label / candidate_name)
        candidates.append(source_gt_dir / candidate_name)
        candidates.extend(index.get(candidate_name.lower(), []))

    return unique_paths(candidates)


def find_source_gt(source_gt_dir: Path, target_gt_dir: Path, target_path: Path, label: Optional[str], folder: Optional[str], index) -> Optional[Path]:
    for candidate in source_candidates(source_gt_dir, target_gt_dir, target_path, label, folder, index):
        if candidate.is_file() and file_has_boxes(candidate):
            return candidate
    return None


def should_fill_target(target_path: Path, overwrite_existing: bool) -> Tuple[bool, str, int]:
    if not target_path.exists():
        return True, 'missing', 0
    box_count = valid_box_count(target_path)
    if box_count == 0:
        return True, 'empty', 0
    if overwrite_existing:
        return True, 'overwrite_existing', box_count
    return False, 'non_empty_keep', box_count


def copy_gt(source: Path, target: Path, dry_run: bool) -> int:
    box_count = valid_box_count(source)
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return box_count


def write_manifest(out_path: Path, rows: List[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'target_path', 'source_path', 'action', 'reason', 'label', 'folder',
                'old_box_count', 'new_box_count', 'candidate_count',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def collect_targets(args):
    labels = args.labels
    if args.data_root is not None:
        targets = []
        for label in labels:
            targets.extend(iter_expected_targets_from_data(args.data_root, args.target_gt_dir, label, args.layout, args.sequence_level))
        existing = {str(path) for path, _, _ in targets}
        for item in iter_existing_targets(args.target_gt_dir, labels):
            if str(item[0]) not in existing:
                targets.append(item)
        return targets
    return list(iter_existing_targets(args.target_gt_dir, labels))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fill missing or empty GT txt files from another existing GT directory.'
    )
    parser.add_argument('--target-gt-dir', '--target_gt_dir', dest='target_gt_dir', type=Path, required=True,
                        help='GT directory to modify, e.g. gt_from_pred.')
    parser.add_argument('--source-gt-dir', '--source_gt_dir', dest='source_gt_dir', type=Path, required=True,
                        help='Existing manually-filled GT directory used as source.')
    parser.add_argument('--data-root', '--data_root', dest='data_root', type=Path, default=None,
                        help='Optional test data root. If set, missing target GT files are also created for these images.')
    parser.add_argument('--labels', nargs='+', default=['abnormal'], choices=['abnormal', 'normal'],
                        help='Labels to process. Default: abnormal.')
    parser.add_argument('--layout', choices=['label-folder', 'folder', 'flat'], default='label-folder',
                        help='Expected target layout when --data-root is used. Default: label-folder.')
    parser.add_argument('--sequence-level', action='store_true', default=False,
                        help='Strip trailing frame suffix like _000000 from expected target names when --data-root is used.')
    parser.add_argument('--overwrite-existing', action='store_true', default=False,
                        help='Also overwrite non-empty target files. Default only fills missing or empty target files.')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Do not copy files; only write manifest and print summary.')
    args = parser.parse_args()

    args.target_gt_dir = expand_path(args.target_gt_dir)
    args.source_gt_dir = expand_path(args.source_gt_dir)
    if args.data_root is not None:
        args.data_root = expand_path(args.data_root)

    if not args.target_gt_dir.exists():
        args.target_gt_dir.mkdir(parents=True, exist_ok=True)
    if not args.target_gt_dir.is_dir():
        raise FileNotFoundError(f'target-gt-dir is not a directory: {args.target_gt_dir}')
    if not args.source_gt_dir.is_dir():
        raise FileNotFoundError(f'source-gt-dir not found: {args.source_gt_dir}')
    if args.data_root is not None and not args.data_root.is_dir():
        raise FileNotFoundError(f'data-root not found: {args.data_root}')

    index = build_name_index(args.source_gt_dir)
    targets = collect_targets(args)

    rows: List[dict] = []
    copied = 0
    skipped = 0
    missing_source = 0

    for target_path, label, folder in targets:
        fill, reason, old_count = should_fill_target(target_path, args.overwrite_existing)
        candidates = source_candidates(args.source_gt_dir, args.target_gt_dir, target_path, label, folder, index)
        if not fill:
            skipped += 1
            rows.append({
                'target_path': str(target_path),
                'source_path': '',
                'action': 'skip',
                'reason': reason,
                'label': label or '',
                'folder': folder or '',
                'old_box_count': old_count,
                'new_box_count': old_count,
                'candidate_count': len(candidates),
            })
            continue

        source = find_source_gt(args.source_gt_dir, args.target_gt_dir, target_path, label, folder, index)
        if source is None:
            missing_source += 1
            rows.append({
                'target_path': str(target_path),
                'source_path': '',
                'action': 'missing_source',
                'reason': reason,
                'label': label or '',
                'folder': folder or '',
                'old_box_count': old_count,
                'new_box_count': old_count,
                'candidate_count': len(candidates),
            })
            continue

        new_count = copy_gt(source, target_path, args.dry_run)
        copied += 1
        rows.append({
            'target_path': str(target_path),
            'source_path': str(source),
            'action': 'would_copy' if args.dry_run else 'copied',
            'reason': reason,
            'label': label or '',
            'folder': folder or '',
            'old_box_count': old_count,
            'new_box_count': new_count,
            'candidate_count': len(candidates),
        })

    manifest_name = 'fill_empty_gt_manifest_dry_run.csv' if args.dry_run else 'fill_empty_gt_manifest.csv'
    manifest_path = args.target_gt_dir / manifest_name
    write_manifest(manifest_path, rows)

    print(f'target_gt_dir: {args.target_gt_dir}')
    print(f'source_gt_dir: {args.source_gt_dir}')
    print(f'targets_seen: {len(targets)}')
    print(f'copied: {copied}')
    print(f'skipped_non_empty: {skipped}')
    print(f'missing_source: {missing_source}')
    print(f'dry_run: {args.dry_run}')
    print(f'manifest: {manifest_path}')


if __name__ == '__main__':
    main()
