"""从完整训练集冻结性能样本，复制原图并导出可整体迁移的样本目录。"""
import argparse
import json
from pathlib import Path
import random
import shutil
import tempfile

from s02_workload import load_dataset, load_fixed_samples


def export_samples(config_path):
    """一次选样并发布完整目录；将图像与标注保存为清单，来源统计单独保存；原图按字节复制。"""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if set(config) != {'dataset', 'out', 'sample_count', 'sample_seed'}:
        raise ValueError('Expected dataset, out, sample_count and sample_seed')
    if type(config['sample_count']) is not int or config['sample_count'] < 1:
        raise ValueError('sample_count must be a positive integer')
    if type(config['sample_seed']) is not int:
        raise ValueError('sample_seed must be an integer')
    for key in ('dataset', 'out'):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(key + ' must be a nonempty path')
        value = Path(config[key]).expanduser()
        config[key] = str((value if value.is_absolute() else config_path.parent / value).resolve())
    meta, rows = load_dataset(config['dataset'], 'train')
    if meta['purpose'] != 'formal_baseline':
        raise ValueError('Export requires a formal_baseline dataset')
    if config['sample_count'] > len(rows):
        raise ValueError('sample_count exceeds training set size')
    # 与原性能入口一致：先按ID排序，再按seed无放回抽样；输出保留抽样顺序。
    selected = random.Random(config['sample_seed']).sample(
        sorted(rows, key=lambda row: row['sample_id']), config['sample_count'])
    output = Path(config['out'])
    if output.exists():
        raise FileExistsError(output)
    annotation_fields = ('width', 'height', 'gt_pixel', 'target_norm1000')
    sources = [Path(row['image']) for row in selected]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if len({source.resolve() for source in sources}) != len(sources):
        raise ValueError('Selected samples must reference distinct image files')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    try:
        (temporary / 'images').mkdir()
        exported, provenance = [], []
        for index, (row, source) in enumerate(zip(selected, sources), 1):
            image = Path('images') / (f'{index:04d}' + source.suffix.lower())
            shutil.copyfile(source, temporary / image)
            exported.append({'sample_id': row['sample_id'], 'image': image.as_posix(),
                             **{key: row[key] for key in annotation_fields}})
            # 来源地址、SHA及分组信息只用于追溯，不参与推理或评分。
            provenance.append({key: value for key, value in row.items()
                               if key not in annotation_fields and key not in ('image', 'split')})
            provenance[-1]['source_image'] = row['image']
        (temporary / 'samples.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in exported), encoding='utf-8')
        info = {'purpose': 'performance_samples', 'source_split': 'train',
                'selected_images': len(exported), 'source_images': len(rows),
                'method': 'uniform_random_without_replacement',
                'sample_seed': config['sample_seed'], 'source_dataset': config['dataset'],
                'image_paths_relative_to': 'samples.jsonl', 'samples': provenance}
        (temporary / 'sample_info.json').write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        load_fixed_samples(temporary / 'samples.jsonl')
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().with_suffix('.json')),
                        help='JSON config (default: %(default)s)')
    args = parser.parse_args()
    print(export_samples(args.config))


if __name__ == '__main__':
    main()
