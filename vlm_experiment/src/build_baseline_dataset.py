"""按脚本内设置的批次相对目录构建冻结清单，不依赖交付目录名称格式。

图片与同名JSON留在原处；manifest保存图片/标注绝对路径、内容hash及两种坐标GT。
完整会话目录不可跨集合；任一眼hard整图排除，其他读取错误使构建失败。
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import re
from PIL import Image
SIDES = ('image_left_eye', 'image_right_eye')
STATES = ('open', 'closed', 'narrow', 'occluded')
LABELS = {'eye_' + state: state for state in STATES}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 标注左右指驾驶员自身左右；输出字段指图像左右。固定按ID映射，不按框位置重排。
EYE_ID_TO_SIDE = {'2': 'image_left_eye', '1': 'image_right_eye'}

# batch_dirs是相对--data-root的路径；None表示尚未提供真实目录名，运行前必须填写。
BUILD_SETTINGS = {
    "purpose": "formal_baseline",
    "splits": {
        "train": [
            # 67,
            # 68,
            # 70,
            71,
            # 75
        ],
        "val": [
            77
        ],
        "test": [
            79
        ]
    },
    "hard_policy": "exclude_image",
    "batch_dirs": {
        # "67": None,
        # "68": None,
        # "70": None,
        "71": "FA-The-71st-batch-delivery-20260204",
        "77": "FA-The-77st-batch-delivery-20260228",
        # "77": None,
        "79": "FA-The-79st-batch-delivery-20260314"
    }
}

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """先写同目录临时文件再替换，避免中断留下不完整的构建报告。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_gt(path):
    """返回像素GT；任一眼hard则排除整图，其他未知标签或非双眼标注报错。"""
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    items = [item for item in data['dataList'] if 'eye_status' in item.get('properties', {})]
    # hard是整图排除条件，先于双眼数量检查；它不等同于状态不可判定的occluded。
    if any(item['properties']['eye_status'] == 'hard' for item in items):
        return None
    if len(items) != 2:
        raise ValueError(f'Expected two localized eyes, got {len(items)}')
    eyes = {}
    for item in items:
        side = EYE_ID_TO_SIDE.get(str(item.get('id')))
        if side is None or side in eyes:
            raise ValueError('Expected unique eye annotation IDs 1 and 2')
        state = LABELS[item['properties']['eye_status']]
        (x1, y1), (x2, y2) = item['coordinates']
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
            raise ValueError('Nonfinite GT coordinates')
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError('Degenerate GT cannot be scored')
        eyes[side] = {'bbox': box, 'state': state}
    # 标注列表顺序不影响左右；框和状态始终归属同一ID，不用几何排序修正标注。
    return {side: eyes[side] for side in SIDES}


def normalize_gt(gt, width, height):
    """生成Norm1000整数训练目标；保留原像素GT供评分，避免GT量化误差。

    使用实际图片W/H，Python round采用ties-to-even；不旋转、裁剪或修改入参。
    """
    return {side: {'bbox': [round(v / size * 1000) for v, size in zip(gt[side]['bbox'], (width, height, width, height))],
                   'state': gt[side]['state']} for side in SIDES}


def build_dataset(root, out, observed_list=None):
    """按批次划分扫描全量图片，返回构建报告并写入新的out目录。

    BUILD_SETTINGS中的splits决定批次用途，batch_dirs明确各批次相对root的位置。
    配置错误在创建输出前抛出；样本错误汇总为failed报告，不发布manifest。
    observed_list记录已看过的诊断图片相对路径，用于防止污染最终测试集。
    """
    config = BUILD_SETTINGS
    root, out = Path(root).resolve(), Path(out)
    purpose = config['purpose']
    splits = config['splits']
    assignment = {}
    for split, batches in splits.items():
        for batch in batches:
            if batch in assignment:
                raise ValueError('Batch assigned twice: ' + str(batch))
            assignment[batch] = split
    batch_dirs = config.get('batch_dirs')
    if not isinstance(batch_dirs, dict):
        raise ValueError('Configure batch_dirs: batch ID -> directory relative to data root')
    folders = {}
    for batch in assignment:
        relative = batch_dirs.get(str(batch))
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f'Fill batch_dirs[{str(batch)!r}] with the actual relative directory')
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or path == Path('.'):
            raise ValueError(f'Batch directory must be a nonempty relative path: {relative}')
        folder = root / path
        resolved = folder.resolve()
        if (resolved != root and root not in resolved.parents) or resolved == root or not folder.is_dir():
            raise ValueError(f'Batch directory missing or outside data root: {relative}')
        # 不同批次不能指向同一目录或父子目录，包括符号链接别名，防止重复扫描。
        for existing in folders.values():
            other = existing.resolve()
            if resolved == other or other in resolved.parents or resolved in other.parents:
                raise ValueError(f'Batch directories overlap: {relative} and {existing.relative_to(root)}')
        folders[batch] = folder
    observed = set()
    if observed_list:
        observed = {line.strip() for line in Path(observed_list).read_text().splitlines() if line.strip()}
        if not observed:
            raise ValueError('Observed list is empty; provide actual diagnostic paths or a documented review note')
        if any(Path(p).is_absolute() or '..' in Path(p).parts for p in observed):
            raise ValueError('Observed list must use data-root-relative paths')
    # 输出必须是新目录，避免失败构建遗留或覆盖上一版冻结清单。
    out.mkdir(parents=True, exist_ok=False)
    records = {split: [] for split in splits}
    exclusions, errors = [], []
    # 三种身份分别记录所属split：人员只统计交叠，会话和相同内容用于阻断泄漏。
    persons, sessions, contents = defaultdict(set), defaultdict(set), defaultdict(set)
    seen_paths = set()
    counts = {split: Counter() for split in splits}
    scanned = Counter()
    for batch, folder in sorted(folders.items()):
        split = assignment[batch]
        # 仅扫描显式选中的批次，递归收集全部图片；排序固定清单顺序，不进行抽样。
        images = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not images:
            errors.append({'batch': batch, 'error': 'No images'})
        # 双向核对图片/标注：此处发现孤立JSON，逐图读取时发现缺失JSON。
        image_stems = {p.with_suffix('') for p in images}
        for annotation in folder.rglob('*.json'):
            if annotation.with_suffix('') not in image_stems:
                errors.append({'annotation': annotation.resolve().as_posix(), 'error': 'Annotation has no matching image'})
        for image in images:
            relative = image.relative_to(root).as_posix()
            seen_paths.add(relative)
            annotation = image.with_suffix('.json')
            scanned[split] += 1
            try:
                if root not in image.resolve().parents or root not in annotation.resolve().parents:
                    raise ValueError('Source path escapes data root')
                gt = read_gt(annotation)
                if gt is None:
                    exclusions.append({'image': image.resolve().as_posix(), 'split': split, 'batch': batch, 'reason': 'hard_on_at_least_one_eye'})
                    continue
                # 会话内图片与JSON同级；人员数字前缀只用于统计，不作为split隔离键。
                match = re.match(r'^(\d+)_', image.parent.name)
                if not match:
                    raise ValueError('Cannot extract person ID from session directory')
                person = match[1]
                with Image.open(image) as decoded:
                    width, height = decoded.size  # 保留存储像素方向；忽略JSON尺寸与EXIF旋转。
                image_hash = sha256_file(image)
                persons[person].add(split)
                # 完整场景目录为group；去掉交付批次前缀，识别同一场景重复交付。
                session_group = image.parent.relative_to(folder).as_posix()
                sessions[session_group].add(split)
                contents[image_hash].add(split)
                states = Counter(eye['state'] for eye in gt.values())
                counts[split].update(states)
                # 文件地址在构建时确定；相对路径仅用于稳定样本ID、会话分组和诊断清单匹配。
                records[split].append({'sample_id': hashlib.sha256(relative.encode()).hexdigest(),
                    'image': image.resolve().as_posix(), 'annotation': annotation.resolve().as_posix(),
                    'image_sha256': image_hash, 'annotation_sha256': sha256_file(annotation),
                    'batch': batch, 'split': split, 'person_id': person, 'session_group': session_group, 'session': image.parent.relative_to(root).as_posix(),
                    'scenario': image.parent.name.rsplit('_', 1)[-1], 'width': width, 'height': height,
                    'observed_diagnostic': relative in observed,
                    'gt_pixel': gt, 'target_norm1000': normalize_gt(gt, width, height)})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({'image': image.resolve().as_posix(), 'split': split, 'error': f'{type(exc).__name__}: {exc}'})
    # 同一人员不同会话可以跨集合；同一会话或完全相同图片跨集合则拒绝构建。
    for group, owners in sessions.items():
        if len(owners) > 1:
            errors.append({'session_group': group, 'error': 'Session directory crosses splits', 'splits': sorted(owners)})
    for digest, owners in contents.items():
        if len(owners) > 1:
            errors.append({'image_sha256': digest, 'error': 'Identical image crosses splits', 'splits': sorted(owners)})
    for split, rows in records.items():
        if not rows:
            errors.append({'split': split, 'error': 'Empty split after exclusions'})
        # 已用于调参/诊断的图片不能再作为最终测试样本。
        if split == 'test' and any(row['observed_diagnostic'] for row in rows):
            errors.append({'split': split, 'error': 'Previously observed sample enters held-out test'})
    if observed - seen_paths:
        errors.append({'error': 'Observed paths absent from selected batches', 'count': len(observed-seen_paths)})
    atomic_json(out/'exclusions.json', exclusions)
    atomic_json(out/'build_errors.json', errors)
    # 保留排除/错误报告便于定位，但有任何错误就不输出可供训练评估的清单。
    manifests = {}
    if not errors:
        for split, rows in records.items():
            path = out/(split+'.jsonl')
            with path.open('x', encoding='utf-8') as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            manifests[split] = {'file': path.name, 'sha256': sha256_file(path), 'images': len(rows),
                'states': {s: counts[split][s] for s in STATES},
                'persons': len({r['person_id'] for r in rows}), 'sessions': len({r['session'] for r in rows}),
                'scenarios': dict(Counter(r['scenario'] for r in rows)),
                'observed_diagnostic_images': sum(r['observed_diagnostic'] for r in rows)}
    report = {'schema_version': 4, 'path_format': 'absolute', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'failed' if errors else 'built', 'purpose': purpose, 'source_root_at_build': str(root),
        'annotation_id_to_side': EYE_ID_TO_SIDE,
        'config': config,  # 保存脚本设置快照用于追溯，不是需要读取的配置文件。
        'observed_list_sha256': sha256_file(observed_list) if observed_list else None,
        'diagnostic_overlap_reviewed': bool(observed_list or config.get('diagnostic_overlap_review')),
        'scanned_images': dict(scanned), 'excluded_images': len(exclusions),
        'excluded_by_split': dict(Counter(x['split'] for x in exclusions)),
        'error_count': len(errors), 'manifests': manifests,
        'person_overlap_across_splits': {person: sorted(owners) for person, owners in sorted(persons.items()) if len(owners)>1},
        'person_overlap_is_error': False, 'split_group': 'session_directory',
        'builder_sha256': sha256_file(__file__),
        'coordinate_contract': {'gt_pixel': 'original_image_pixel_xyxy', 'target_norm1000': 'integer_xyxy_0_1000', 'rounding': 'ties_to_even'},
        'scope': 'paired localized eyes; hard images excluded; complete session directories isolated; same-person different-session overlap allowed and reported; cross-scene evaluation, not unseen-person generalization'}
    atomic_json(out/'dataset.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', required=True, help='New directory')
    parser.add_argument('--observed-list', help='All known diagnostic image paths, one data-root-relative path per line')
    args = parser.parse_args()
    report = build_dataset(args.data_root, args.out, args.observed_list)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'built':
        raise SystemExit('Dataset build failed; see build_errors.json. No usable manifests published.')


if __name__ == '__main__':
    main()
