"""按冻结清单完整运行Qwen3.5 baseline；评估定位与状态分类精度，保存逐case预测及失败记录。"""
import argparse
import json
import itertools
import math
import os
from pathlib import Path
from PIL import Image, ImageDraw
SIDES = ('image_left_eye', 'image_right_eye')
STATES = ('open', 'closed', 'narrow', 'occluded')
PROMPT_TEMPLATE = '''Locate the driver's two eyes in this image.
image_left_eye is the eye with the smaller horizontal box-center coordinate;
image_right_eye has the larger coordinate. Use image positions, not anatomical left and right.
For each eye, output bbox and state. Return tight boxes around the eye regions,
not the eyebrows or the whole face.
bbox must be [xmin,ymin,xmax,ymax] with integer coordinates normalized to 0..1000.
state must be open, closed, narrow, or occluded. Use occluded when lighting, hair,
or other factors prevent reliable classification as open, closed, or narrow.
Return exactly one compact JSON object with exactly this structure:
{"image_left_eye":{"bbox":[100,200,130,220],"state":"open"},"image_right_eye":{"bbox":[150,200,180,220],"state":"closed"}}
Do not use a JSON array, bbox_2d, or label. Do not include explanations, markdown,
or any additional fields.'''


def atomic_json(path, value):
    """先写同目录临时文件再替换，避免中断留下半份case或summary。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def parse_prediction(raw, width, height):
    """严格验证norm1000，然后一次转像素；不夹紧、交换左右或补框。"""
    text = raw.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in ('```', '```json') or lines[-1].strip() != '```':
            raise ValueError('Invalid JSON fence')
        text = '\n'.join(lines[1:-1])
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != set(SIDES):
        raise ValueError('Expected exactly two eye fields')
    result = {}
    for side in SIDES:
        eye = value[side]
        if not isinstance(eye, dict) or set(eye) != {'bbox', 'state'} or eye['state'] not in STATES:
            raise ValueError('Invalid eye schema/state')
        box = eye['bbox']
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError('Expected four integer coordinates')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError('Invalid norm1000 bbox')
        result[side] = {'bbox': [v * size / 1000 for v, size in zip(box, (width, height, width, height))], 'state': eye['state']}
    return result


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def summarize(samples, cases):
    """固定清单是分母；缺失/非法预测计失败，状态按对应侧计分且不依赖IoU。"""
    n = len(samples)
    if not n:
        raise ValueError('Empty evaluation manifest')
    confusion = {state: {p: 0 for p in (*STATES, 'invalid')} for state in STATES}
    thresholds = {str(t): {'localization_tp': 0, 'same_side_tp': 0, 'end_to_end_eyes': 0, 'both_eyes': 0} for t in (0.5, 0.75)}
    valid = completed = 0
    same_side_ious = []
    area_ratios, center_errors = [], []
    errors = {}
    for sample in samples:
        case = cases.get(sample['sample_id'])
        completed += case is not None
        pred = case.get('prediction') if case else None
        gt = sample['gt_pixel']
        if pred:
            valid += 1
        else:
            reason = case.get('error_type', 'invalid') if case else 'not_run'
            errors[reason] = errors.get(reason, 0) + 1
        for side in SIDES:
            confusion[gt[side]['state']][pred[side]['state'] if pred else 'invalid'] += 1
            same_side_ious.append(iou(gt[side]['bbox'], pred[side]['bbox']) if pred else 0.0)
            if pred:
                a, b = gt[side]['bbox'], pred[side]['bbox']
                area_ratios.append(((b[2]-b[0])*(b[3]-b[1]))/((a[2]-a[0])*(a[3]-a[1])))
                center_errors.append(math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2))
        if not pred:
            continue
        matrix = [[iou(gt[g]['bbox'], pred[p]['bbox']) for p in SIDES] for g in SIDES]
        for text, counts in thresholds.items():
            threshold = float(text)
            # 定位允许一对一匹配；端到端仍要求模型原始左右字段正确。
            permutation = max(itertools.permutations(range(2)), key=lambda perm: (sum(matrix[i][perm[i]] >= threshold for i in range(2)), sum(matrix[i][perm[i]] for i in range(2))))
            counts['localization_tp'] += sum(matrix[i][permutation[i]] >= threshold for i in range(2))
            counts['same_side_tp'] += sum(matrix[i][i] >= threshold for i in range(2))
            joint = sum(matrix[i][i] >= threshold and gt[s]['state'] == pred[s]['state'] for i, s in enumerate(SIDES))
            counts['end_to_end_eyes'] += joint
            counts['both_eyes'] += joint == 2
    per_class = {}
    for state in STATES:
        support = sum(confusion[state].values())
        tp = confusion[state][state]
        fp = sum(confusion[g][state] for g in STATES if g != state)
        fn = support - tp
        per_class[state] = {'support': support, 'precision': tp/(tp+fp) if tp+fp else None,
                            'recall': tp/support if support else None,
                            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    for counts in thresholds.values():
        tp = counts['localization_tp']
        counts.update({'localization_precision': tp/(2*valid) if valid else 0.0, 'localization_recall': tp/(2*n),
                       'localization_fp': 2*valid-tp, 'localization_fn': 2*n-tp,
                       'same_side_recall': counts['same_side_tp']/(2*n),
                       'end_to_end_eye_rate': counts['end_to_end_eyes']/(2*n), 'both_eyes_rate': counts['both_eyes']/n})
    supported = [v['f1'] for v in per_class.values() if v['support']]
    return {'manifest_images': n, 'completed_images': completed, 'valid_prediction_images': valid,
            'inference_and_parse_success_rate': valid/n, 'errors': errors,
            'same_side_mean_iou_failures_zero': sum(same_side_ious)/(2*n),
            'mean_area_ratio_valid_predictions_only': sum(area_ratios)/len(area_ratios) if area_ratios else None,
            'mean_center_error_pixels_valid_predictions_only': sum(center_errors)/len(center_errors) if center_errors else None,
            'state_confusion_gt_rows': confusion, 'state_per_class': per_class,
            'state_accuracy_all_eyes': sum(confusion[s][s] for s in STATES)/(2*n),
            'state_macro_f1_supported_classes': sum(supported)/len(supported) if supported else None,
            'state_macro_f1_four_classes': sum(supported)/4 if len(supported)==4 else None,
            'four_class_coverage': all(v['support'] for v in per_class.values()), 'thresholds': thresholds}


def load_dataset(dataset, split):
    dataset = Path(dataset)
    meta = json.loads((dataset/'dataset.json').read_text())
    expected_coordinates = {'gt_pixel': 'original_image_pixel_xyxy', 'target_norm1000': 'integer_xyxy_0_1000', 'rounding': 'ties_to_even'}
    if meta.get('schema_version') != 4 or meta.get('path_format') != 'absolute' or meta.get('coordinate_contract') != expected_coordinates:
        raise ValueError('Expected schema v4 with absolute image paths and norm1000 contract; rebuild dataset')
    if meta['status'] != 'built':
        raise ValueError('Dataset is not successfully built')
    info = meta['manifests'][split]
    manifest = dataset/info['file']
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != info['images'] or not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest count/identity mismatch')
    if any(row['split'] != split for row in rows):
        raise ValueError('Wrong split in manifest')
    if any(not Path(row['image']).is_absolute() for row in rows):
        raise ValueError('Manifest images must use absolute paths; rebuild dataset')
    return meta, rows


def save_overlay(path, image, gt, prediction):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for label, eyes, color, offset in [('GT', gt, 'lime', 0), ('PRED', prediction, 'red', 16)]:
        if eyes is None:
            continue
        for side in SIDES:
            eye = eyes[side]
            draw.rectangle(eye['bbox'], outline=color, width=2)
            draw.text((eye['bbox'][0], max(0, eye['bbox'][1]+offset)), f"{label} {side}: {eye['state']}", fill=color)
    canvas.save(path)


def run(args):
    if args.context_budget < 1 or args.max_new_tokens < 1 or args.max_new_tokens >= args.context_budget:
        raise ValueError('Invalid context/generation budget')
    if args.split == 'test' and not args.final_test:
        raise ValueError('Held-out test requires --final-test; use val for baseline and model selection')
    dataset, out = Path(args.dataset), Path(args.out)
    meta, rows = load_dataset(dataset, args.split)
    if meta['purpose'] != 'formal_baseline' and not args.preflight_only:
        raise ValueError('Local structure-test manifests cannot produce a model baseline')
    if args.preflight_only:
        if args.resume:
            raise ValueError('Preflight is not resumable')
        out.mkdir(parents=True, exist_ok=False)
        errors = []
        for sample in rows:
            try:
                with Image.open(sample['image']) as image:
                    if image.size != (sample['width'], sample['height']):
                        raise ValueError('Image dimensions changed')
                    image.convert('RGB').load()
            except (OSError, ValueError) as exc:
                errors.append({'sample_id': sample['sample_id'], 'error': str(exc)})
        atomic_json(out/'preflight.json', {'checked': len(rows), 'errors': errors,
            'dataset_purpose': meta['purpose'], 'status': 'data_check_only', 'inference_performed': False})
        if errors:
            raise ValueError('Data preflight failed; see preflight.json')
        print(f'Data checked: {len(rows)} images; no model inference performed.')
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_path = Path(args.model).resolve()
    # 记录推理参数供结果解释和续跑使用；模型加载器负责检查可加载性。
    if not model_path.is_dir():
        raise ValueError('Expected a local model directory')
    config = {'dataset_path': str(dataset.resolve()), 'split': args.split,
        'model_path': str(model_path), 'prompt_template': PROMPT_TEMPLATE, 'coordinate_scale': 'norm1000',
        'context_budget': args.context_budget, 'max_new_tokens': args.max_new_tokens,
        'seed': args.seed, 'do_sample': False, 'num_beams': 1, 'enable_thinking': False,
        'dtype': 'bfloat16', 'attention_implementation': 'sdpa',
        'processor_policy': 'saved processor defaults; no EXIF transpose; no input truncation',
        'save_overlays': args.save_overlays}
    cases = {}
    allowed_ids = {row['sample_id'] for row in rows}
    if args.resume:
        if json.loads((out/'config.json').read_text()) != config:
            raise ValueError('Resume paths or inference settings differ; create a new run')
        for file in (out/'cases').glob('*.json'):
            case = json.loads(file.read_text())
            if case['sample_id'] not in allowed_ids or case['sample_id'] in cases:
                raise ValueError('Unexpected/duplicate saved case')
            cases[case['sample_id']] = case
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out/'cases').mkdir()
        atomic_json(out/'config.json', config)
    if args.save_overlays:
        (out/'overlays').mkdir(exist_ok=True)
    atomic_json(out/'summary.json', {'status': 'incomplete', 'expected_images': len(rows), 'completed_images': len(cases)})
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='sdpa').eval()
    try:
        for index, sample in enumerate(rows):
            sample_id = sample['sample_id']
            if sample_id in cases:
                continue  # 已记录的失败也保留；重试失败使用新的run，避免挑选性覆盖。
            case = {'sample_id': sample_id, 'image': sample['image'], 'prediction': None, 'raw_output': None}
            image = inputs = outputs = None
            stage = 'image'
            try:
                with Image.open(sample['image']) as source:
                    image = source.convert('RGB')
                if image.size != (sample['width'], sample['height']):
                    raise ValueError('Image dimensions mismatch')
                stage = 'processor'
                messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT_TEMPLATE}]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = processor(text=[prompt], images=[image], return_tensors='pt', padding=False, truncation=False)
                length = inputs['input_ids'].shape[-1]
                case['input_tokens'] = length
                if 'image_grid_thw' in inputs:
                    case['image_grid_thw'] = inputs['image_grid_thw'].tolist()
                if length + args.max_new_tokens > args.context_budget:
                    stage = 'context_budget'
                    raise ValueError('Input plus generation budget exceeds frozen limit; no truncation applied')
                inputs = {k: v.to(device='cuda:0', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
                stage = 'generate'
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                        num_beams=1, use_cache=True, return_dict_in_generate=False)
                tokens = outputs[0, length:].cpu().tolist()
                case['output_tokens'] = len(tokens)
                case['hit_token_limit'] = len(tokens) >= args.max_new_tokens
                raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                case['raw_output'] = raw
                stage = 'parse'
                case['prediction'] = parse_prediction(raw, sample['width'], sample['height'])
                case['status'] = 'valid'
            except Exception as exc:
                case.update(status='failed', error_type=stage, error=f'{type(exc).__name__}: {exc}')
            finally:
                inputs = outputs = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if args.save_overlays and image is not None:
                try:
                    save_overlay(out/'overlays'/(sample_id+'.jpg'), image, sample['gt_pixel'], case['prediction'])
                except Exception as exc:
                    case['overlay_error'] = str(exc)
            if image is not None:
                image.close()
            atomic_json(out/'cases'/(sample_id+'.json'), case)
            cases[sample_id] = case
            print(f"[{index+1}/{len(rows)}] {sample_id[:12]} {case['status']}", flush=True)
    finally:
        summary = summarize(rows, cases)
        complete = summary['completed_images'] == len(rows)
        summary.update({'status': 'evaluation_complete' if complete else 'incomplete',
            'baseline_ready_for_comparison': bool(complete and args.split=='val' and summary['four_class_coverage'] and meta['diagnostic_overlap_reviewed']),
            'diagnostic_overlap_reviewed': meta['diagnostic_overlap_reviewed'],
            'note': 'Completion is manifest coverage, not a quality pass. Missing diagnostic overlap review or class coverage prevents formal baseline readiness.'})
        atomic_json(out/'summary.json', summary)
        by_batch = {str(batch): summarize([r for r in rows if r['batch']==batch], cases) for batch in sorted({r['batch'] for r in rows})}
        atomic_json(out/'by_batch.json', by_batch)
    print('Results:', out.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default='/workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-9B')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--context-budget', type=int, default=5600)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--final-test', action='store_true')
    parser.add_argument('--save-overlays', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
