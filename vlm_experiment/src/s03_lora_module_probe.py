"""Inspect actual Qwen3.5 Linear modules before choosing a LoRA allowlist."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-9B')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--out', required=True, help='New JSON file; existing files are rejected')
    parser.add_argument('--rank', type=int, default=16)
    args = parser.parse_args()
    if args.rank < 1:
        parser.error('--rank must be positive')
    out = Path(args.out)
    if out.exists():
        parser.error('--out already exists')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    import torch
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': 0},
    )
    model.eval()
    modules = []
    groups = defaultdict(list)
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        row = {
            'name': name,
            'in_features': module.in_features,
            'out_features': module.out_features,
            'weight_dtype': str(module.weight.dtype),
            'candidate_adapter_parameters': args.rank * (module.in_features + module.out_features),
        }
        modules.append(row)
        pattern = '.'.join('*' if part.isdigit() else part for part in name.split('.'))
        groups[pattern].append(row)
    summary = []
    for pattern, rows in sorted(groups.items()):
        shapes = sorted({(r['in_features'], r['out_features']) for r in rows})
        item = {
            'pattern': pattern, 'count': len(rows), 'shapes': shapes,
            'example': rows[0]['name'],
            'candidate_adapter_parameters': sum(r['candidate_adapter_parameters'] for r in rows),
        }
        summary.append(item)
        print(f"{pattern}: count={len(rows)}, shapes={shapes}", flush=True)
        print(f"  example: {rows[0]['name']}", flush=True)
    report = {
        'model_path': args.model, 'model_class': type(model).__name__,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'rank_for_estimate': args.rank,
        'status': 'inventory_only_no_adapter_injected',
        'top_level_modules': [name for name, _ in model.named_children()],
        'linear_modules': modules, 'groups': summary,
        'note': 'Counts are hypothetical per-module estimates, not an approved target list. Non-Linear projections are outside this inventory.',
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(f'Report: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
