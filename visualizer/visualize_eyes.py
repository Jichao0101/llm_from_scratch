"""可视化同目录同名JSON的眼框和状态；结果统一写入visualizer/temp/并保留源目录结构。"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
# 固定按左右眼配色，不随状态变化：id=2图像左眼绿色，id=1图像右眼青色。
EYE_COLORS = {'2': 'lime', '1': 'cyan'}
VALID_STATES = {'eye_open', 'eye_closed', 'eye_narrow', 'eye_occluded', 'hard'}


def visualize_image(image_path, output):
    """按原图像素坐标画框，不旋转或改写原图/JSON；返回输出路径及诊断信息。

    这是标注排查工具：缺JSON、空标注、hard或非法框仍输出带说明的图片，
    不套用训练集的整图排除规则，也不为缺失眼睛补框。
    """
    annotation = image_path.with_suffix('.json')
    warnings = []
    notes = []
    with Image.open(image_path) as source:
        canvas = source.convert('RGB')
    draw = ImageDraw.Draw(canvas)
    font_size = max(14, min(canvas.size) // 65)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', font_size)
    except OSError:
        font = ImageFont.load_default()
    line_height = font_size + 8
    thickness = max(2, min(canvas.size) // 400)
    try:
        data = json.loads(annotation.read_text(encoding='utf-8-sig'))
        items = data.get('dataList', [])
        if not isinstance(items, list):
            raise ValueError('dataList must be a list')
    except (OSError, ValueError, AttributeError) as exc:
        items = []
        warnings.append('Cannot read annotation: ' + str(exc))
        notes.append('MISSING / INVALID JSON')

    eyes = [item for item in items if isinstance(item, dict)
            and isinstance(item.get('properties'), dict)
            and 'eye_status' in item['properties']]
    if not eyes:
        notes.append('NO eye_status ANNOTATIONS')
        warnings.append('No eye_status annotations (dataList entries: {})'.format(len(items)))
    seen_ids = set()
    for item in eyes:
        eye_id = str(item.get('id', '?'))
        state = str(item['properties']['eye_status'])
        color = EYE_COLORS.get(eye_id, 'orange')
        if eye_id not in EYE_COLORS:
            notes.append('INVALID EYE ID: ' + eye_id)
        elif eye_id in seen_ids:
            notes.append('DUPLICATE EYE ID: ' + eye_id)
        seen_ids.add(eye_id)
        if state not in VALID_STATES:
            notes.append('id={}: INVALID STATE: {}'.format(eye_id, state))
        try:
            (x1, y1), (x2, y2) = item['coordinates']
            values = (x1, y1, x2, y2)
            if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
                raise ValueError('nonfinite or nonnumeric coordinates')
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError('degenerate box')
            if not (0 <= box[0] < box[2] <= canvas.width and 0 <= box[1] < box[3] <= canvas.height):
                notes.append('id={}: BOX OUTSIDE IMAGE'.format(eye_id))
                warnings.append('id={}: box extends outside image'.format(eye_id))
            draw.rectangle(box, outline=color, width=thickness)
            # 正常标注仅在眼框上方显示原始状态，左右由框和文字的固定颜色区分。
            draw.text((max(0, min(box[0], canvas.width - 1)), max(0, box[1] - line_height)),
                      state, fill=color, font=font)
        except (KeyError, TypeError, ValueError) as exc:
            notes.append('id={}: INVALID / MISSING BOX'.format(eye_id))
            warnings.append('id={}: {}'.format(eye_id, exc))
    for eye_id in EYE_COLORS:
        if eye_id not in seen_ids:
            notes.append('id={}: MISSING EYE ANNOTATION'.format(eye_id))
    # 仅异常图片显示底部提示；正常图片无角落文字或横幅。
    if notes:
        top = max(0, canvas.height - len(notes) * line_height - 8)
        draw.rectangle((0, top, canvas.width, canvas.height), fill='black')
        for index, label in enumerate(notes):
            draw.text((5, top + 4 + index * line_height), label, fill='orange', font=font)
        warnings.extend(label for label in notes if label not in warnings)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    canvas.close()
    return output, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'),
                        help='路径配置JSON，默认使用脚本旁的config.json')
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    try:
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        source_root = Path(config['source_root']).expanduser()
        if not source_root.is_absolute():
            source_root = config_path.parent / source_root
        source_root = source_root.resolve()
        if not source_root.is_dir():
            raise ValueError('source_root must be an existing directory')
        paths = config.get('paths')
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p.strip() for p in paths):
            raise ValueError('paths must be a nonempty list of directory/image/annotation paths')
    except (OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
        parser.error('Invalid config {}: {}'.format(config_path, exc))

    output_root = Path(__file__).resolve().parent / 'temp'
    images = []
    for entry in paths:
        path = Path(entry).expanduser()
        # 相对路径以配置文件为基准，换工作目录启动时仍指向同一批数据。
        if not path.is_absolute():
            path = config_path.parent / path
        path = path.resolve()
        if not path.exists():
            parser.error('Path does not exist: ' + str(path))
        folder = path if path.is_dir() else path.parent
        if path == output_root or output_root in path.parents:
            parser.error('Use the original image directory, not generated temp images')
        if path.is_dir():
            # 仅处理当前目录，避免再次处理temp产物或混入其他会话目录。
            selected = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() == '.json':
            selected = sorted(p for p in folder.iterdir() if p.is_file()
                              and p.stem == path.stem and p.suffix.lower() in IMAGE_SUFFIXES)
            if len(selected) != 1:
                parser.error('Expected exactly one same-stem image for JSON: ' + str(path))
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            selected = [path]
        else:
            parser.error('Expected an image, annotation JSON, or directory: ' + str(path))
        if not selected:
            parser.error('No images found in: ' + str(folder))
        images.extend(selected)
    # 配置可混合目录和单文件；同一图片只输出一次。
    images = list(dict.fromkeys(image.resolve() for image in images))
    for image in images:
        if source_root not in image.parents:
            parser.error('Image outside source_root: ' + str(image))
        if image == output_root or output_root in image.parents:
            parser.error('Cannot use generated output as input: ' + str(image))
    saved = failed = warned = 0
    for image in images:
        try:
            # 输出只落在仓库temp下；保留源层级及原扩展名，避免不同目录/格式同名覆盖。
            relative = image.relative_to(source_root)
            output = output_root / relative.parent / (relative.name + '.png')
            output, warnings = visualize_image(image, output)
            saved += 1
            warned += bool(warnings)
            print('Saved: ' + str(output))
            for warning in warnings:
                print('  WARNING: ' + warning)
        except (OSError, ValueError) as exc:
            failed += 1
            print('FAILED: {}: {}'.format(image, exc))
    print('Saved: {}, annotation warnings: {}, image failures: {}'.format(saved, warned, failed))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
