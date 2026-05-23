from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# This repository has a local copy.py file, which can shadow Python's stdlib
# copy module when PIL imports defusedxml. Keep this standalone script isolated
# from the repository import path before importing PIL.
_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in ('', str(_SCRIPT_DIR)):
    while _path in sys.path:
        sys.path.remove(_path)

from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
Box = Tuple[int, int, int, int]


def expand_path(path: Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def expand_optional_path(path: Optional[Path]) -> Optional[Path]:
    return None if path is None else expand_path(path)


def normalize_key(name: str) -> str:
    return re.sub(r'\s+', ' ', name.strip()).lower()


def strip_frame_suffix(stem: str) -> str:
    return re.sub(r'_\d+$', '', stem)


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def image_gt_names(image_path: Path) -> List[str]:
    stem = image_path.stem
    stripped = strip_frame_suffix(stem)
    return unique([f'{stem}.txt', f'{stripped}.txt'])


def iter_images(data_root: Path, labels: Optional[Sequence[str]], limit: Optional[int]) -> Iterable[Path]:
    roots: List[Path]
    if labels:
        roots = [data_root / label for label in labels if (data_root / label).is_dir()]
        if not roots:
            roots = [data_root]
    else:
        roots = [data_root]

    count = 0
    for root in roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            yield path
            count += 1
            if limit is not None and count >= limit:
                return


def infer_context(image_path: Path, data_root: Path) -> Tuple[Optional[str], Optional[str], Path]:
    rel = image_path.relative_to(data_root)
    parts = rel.parts
    label = parts[0] if len(parts) >= 2 else None
    folder = parts[1] if len(parts) >= 3 else image_path.parent.name
    return label, folder, rel


def build_gt_index(gt_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(gt_dir.rglob('*.txt')):
        index[normalize_key(path.name)].append(path)
    return index


def direct_gt_candidates(
    gt_dir: Path,
    rel_parent: Path,
    label: Optional[str],
    folder: Optional[str],
    names: Sequence[str],
) -> List[Path]:
    candidates: List[Path] = []
    for name in names:
        candidates.append(gt_dir / rel_parent / name)
        if label and folder:
            candidates.append(gt_dir / label / folder / name)
        if folder:
            candidates.append(gt_dir / folder / name)
        if label:
            candidates.append(gt_dir / label / name)
        candidates.append(gt_dir / name)
    return unique_paths(candidates)


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


def choose_index_match(matches: List[Path], label: Optional[str], folder: Optional[str]) -> Tuple[Optional[Path], List[Path]]:
    if not matches:
        return None, []

    preferred = matches
    if folder:
        folder_matches = [p for p in preferred if folder in p.parts]
        if folder_matches:
            preferred = folder_matches
    if label:
        label_matches = [p for p in preferred if label in p.parts]
        if label_matches:
            preferred = label_matches

    preferred = sorted(preferred)
    if len(preferred) == 1:
        return preferred[0], []
    return None, preferred


def find_gt_path(
    image_path: Path,
    data_root: Path,
    gt_dir: Path,
    gt_index: Dict[str, List[Path]],
) -> Tuple[Optional[Path], str, List[str]]:
    label, folder, rel = infer_context(image_path, data_root)
    names = image_gt_names(image_path)

    for candidate in direct_gt_candidates(gt_dir, rel.parent, label, folder, names):
        if candidate.is_file():
            return candidate, 'direct', [str(candidate)]

    ambiguous: List[Path] = []
    for name in names:
        chosen, tied = choose_index_match(gt_index.get(normalize_key(name), []), label, folder)
        if chosen is not None:
            return chosen, 'indexed', [str(chosen)]
        ambiguous.extend(tied)

    if ambiguous:
        return None, 'ambiguous', [str(p) for p in sorted(set(ambiguous))]

    return None, 'missing', names


def parse_numbers(line: str) -> List[float]:
    line = line.strip()
    if not line or line.startswith('#'):
        return []
    values: List[float] = []
    for token in line.replace(',', ' ').split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def parse_box(values: List[float], image_size: Tuple[int, int], box_format: str, normalized: bool) -> Optional[Box]:
    if box_format == 'xyxy-class':
        if len(values) < 5:
            return None
        x1, y1, x2, y2 = values[-4:]
    elif box_format == 'xywh':
        if len(values) < 4:
            return None
        x, y, w, h = values[:4]
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif box_format == 'yolo':
        if len(values) < 5:
            return None
        _, cx, cy, w, h = values[:5]
        x1, y1, x2, y2 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
        normalized = True if not normalized else normalized
    else:
        if len(values) < 4:
            return None
        x1, y1, x2, y2 = values[:4]

    width, height = image_size
    if normalized:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def load_boxes(gt_path: Path, image_size: Tuple[int, int], box_format: str, normalized: bool) -> List[Box]:
    boxes: List[Box] = []
    with gt_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            box = parse_box(parse_numbers(line), image_size, box_format, normalized)
            if box is not None:
                boxes.append(box)
    return boxes


def draw_gt_overlay(image: Image.Image, boxes: Sequence[Box], gt_name: str) -> Image.Image:
    base = image.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        draw.rectangle((x1, y1, x2, y2), fill=(0, 255, 0, 45), outline=(0, 210, 0, 255), width=4)
        text = f'GT {idx}'
        text_box = draw.textbbox((x1, y1), text, font=font)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        label_y = max(0, y1 - th - 6)
        draw.rectangle((x1, label_y, x1 + tw + 8, label_y + th + 6), fill=(0, 110, 0, 220))
        draw.text((x1 + 4, label_y + 3), text, fill=(255, 255, 255, 255), font=font)

    header = f'{gt_name} | boxes={len(boxes)}'
    text_box = draw.textbbox((8, 8), header, font=font)
    draw.rectangle((4, 4, text_box[2] + 12, text_box[3] + 12), fill=(0, 0, 0, 170))
    draw.text((8, 8), header, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(base, overlay).convert('RGB')


def output_path_for(output_dir: Path, rel: Path) -> Path:
    stem = rel.stem + '_gt_overlay.jpg'
    return output_dir / rel.parent / stem


def write_text_list(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Overlay POSCO GT boxes on matching test images.')
    parser.add_argument('--image-path', '--image_path', dest='image_path', type=Path, default=None,
                        help='Direct mode: exact test image path to overlay.')
    parser.add_argument('--gt-path', '--gt_path', dest='gt_path', type=Path, default=None,
                        help='Direct mode: exact GT txt path for --image-path.')
    parser.add_argument('--data-root', '--data_root', dest='data_root', type=Path, default=Path('./data/posco/test'))
    parser.add_argument('--gt-dir', '--gt_dir', dest='gt_dir', type=Path, default=Path('./gt'))
    parser.add_argument('--output-dir', '--output_dir', dest='output_dir', type=Path, default=Path('./gt_overlay'))
    parser.add_argument('--labels', nargs='*', default=['abnormal'],
                        help='Top-level test labels to scan. Use --labels all to scan every image under data-root.')
    parser.add_argument('--box-format', choices=['xyxy', 'xywh', 'xyxy-class', 'yolo'], default='xyxy')
    parser.add_argument('--normalized', action='store_true',
                        help='Scale xyxy/xywh coordinates from 0..1 to image pixels. YOLO is treated as normalized.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--save-empty', action='store_true',
                        help='Save overlay images even when the matched GT file has no valid boxes.')
    args = parser.parse_args()

    image_path = expand_optional_path(args.image_path)
    gt_path_arg = expand_optional_path(args.gt_path)
    data_root = expand_path(args.data_root)
    gt_dir = expand_path(args.gt_dir)
    output_dir = expand_path(args.output_dir)

    if image_path is not None or gt_path_arg is not None:
        if image_path is None or gt_path_arg is None:
            raise ValueError('Use --image-path and --gt-path together for direct overlay mode.')
        if not image_path.is_file():
            raise FileNotFoundError(f'image-path not found: {image_path}')
        if not gt_path_arg.is_file():
            raise FileNotFoundError(f'gt-path not found: {gt_path_arg}')

        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as img:
            boxes = load_boxes(gt_path_arg, img.size, args.box_format, args.normalized)
            overlay = draw_gt_overlay(img, boxes, gt_path_arg.name)
            out_path = output_dir / f'{image_path.stem}_gt_overlay.jpg'
            overlay.save(out_path, quality=95)

        rows = [{
            'image_path': str(image_path),
            'gt_path': str(gt_path_arg),
            'output_path': str(out_path),
            'label': '',
            'folder': '',
            'box_count': len(boxes),
            'match_status': 'direct_path',
        }]
        with (output_dir / 'matched.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['image_path', 'gt_path', 'output_path', 'label', 'folder', 'box_count', 'match_status'],
            )
            writer.writeheader()
            writer.writerows(rows)

        summary = [
            f'image_path: {image_path}',
            f'gt_path: {gt_path_arg}',
            f'output_path: {out_path}',
            f'box_count: {len(boxes)}',
            'box_format: xyxy_pixels' if args.box_format == 'xyxy' and not args.normalized else f'box_format: {args.box_format}',
        ]
        write_text_list(output_dir / 'summary.txt', summary)
        print('\n'.join(summary))
        print(f'overlay: {out_path}')
        return

    if not data_root.is_dir():
        raise FileNotFoundError(f'data-root not found: {data_root}')
    if not gt_dir.is_dir():
        raise FileNotFoundError(f'gt-dir not found: {gt_dir}')

    labels = None if args.labels and args.labels == ['all'] else args.labels
    gt_index = build_gt_index(gt_dir)

    rows = []
    missing_lines: List[str] = []
    ambiguous_lines: List[str] = []
    matched = 0
    saved = 0
    empty_gt = 0
    total = 0

    for image_path in iter_images(data_root, labels, args.limit):
        total += 1
        label, folder, rel = infer_context(image_path, data_root)
        gt_path, match_status, match_info = find_gt_path(image_path, data_root, gt_dir, gt_index)

        if gt_path is None:
            line = f'{image_path}\tstatus={match_status}\tcandidates={" | ".join(match_info)}'
            if match_status == 'ambiguous':
                ambiguous_lines.append(line)
            else:
                missing_lines.append(line)
            continue

        with Image.open(image_path) as img:
            boxes = load_boxes(gt_path, img.size, args.box_format, args.normalized)
            if not boxes:
                empty_gt += 1
                if not args.save_empty:
                    rows.append({
                        'image_path': str(image_path),
                        'gt_path': str(gt_path),
                        'output_path': '',
                        'label': label or '',
                        'folder': folder or '',
                        'box_count': 0,
                        'match_status': match_status,
                    })
                    matched += 1
                    continue

            overlay = draw_gt_overlay(img, boxes, gt_path.name)
            out_path = output_path_for(output_dir, rel)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(out_path, quality=95)

        matched += 1
        saved += 1
        rows.append({
            'image_path': str(image_path),
            'gt_path': str(gt_path),
            'output_path': str(out_path),
            'label': label or '',
            'folder': folder or '',
            'box_count': len(boxes),
            'match_status': match_status,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'matched.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['image_path', 'gt_path', 'output_path', 'label', 'folder', 'box_count', 'match_status'],
        )
        writer.writeheader()
        writer.writerows(rows)

    write_text_list(output_dir / 'missing_gt.txt', missing_lines)
    write_text_list(output_dir / 'ambiguous_gt.txt', ambiguous_lines)

    summary = [
        f'data_root: {data_root}',
        f'gt_dir: {gt_dir}',
        f'output_dir: {output_dir}',
        f'total_images_scanned: {total}',
        f'matched_gt_images: {matched}',
        f'saved_overlay_images: {saved}',
        f'empty_gt_files: {empty_gt}',
        f'missing_gt_images: {len(missing_lines)}',
        f'ambiguous_gt_images: {len(ambiguous_lines)}',
    ]
    write_text_list(output_dir / 'summary.txt', summary)

    print('\n'.join(summary))
    print(f'matched_csv: {output_dir / "matched.csv"}')
    print(f'missing_gt:  {output_dir / "missing_gt.txt"}')
    print(f'ambiguous:   {output_dir / "ambiguous_gt.txt"}')


if __name__ == '__main__':
    main()
