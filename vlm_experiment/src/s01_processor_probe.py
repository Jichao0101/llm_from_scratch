"""Offline, CPU-only input sizing for the existing full-image smoke sample."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import transformers
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    model = Path(args.model).resolve()
    sample_path = Path(args.sample).resolve()
    sample = json.loads(sample_path.read_text(encoding='utf-8'))
    if isinstance(sample, list):
        if len(sample) != 1:
            raise ValueError('Expected exactly one smoke sample')
        sample = sample[0]
    image_path = Path(sample['image'])
    if not image_path.is_absolute():
        image_path = sample_path.parent / image_path
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
    width, height = image.size
    instruction = (
        f"Locate the driver's two eyes in this image "
        f"(original width={width}, height={height}). "
        "image_left_eye is the eye with the smaller horizontal box-center "
        "coordinate; image_right_eye has the larger coordinate. "
        "Use image positions, not anatomical left and right. "
        "For each eye, output bbox and state. "
        "bbox must be [xmin,ymin,xmax,ymax], using integer pixel coordinates "
        "relative to the original image, with the origin at the top-left. "
        "state must be open, closed, narrow, or occluded. "
        "Use occluded when lighting, hair, or other factors prevent reliable "
        "classification as open, closed, or narrow. "
        "Return exactly one compact JSON object with keys "
        "image_left_eye and image_right_eye. "
        'Each value must have the structure {"bbox":[xmin,ymin,xmax,ymax],'
        '"state":"label"}. '
        "Do not include explanations, markdown, or extra fields."
    )
    messages = [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': instruction}
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=[prompt], images=[image], return_tensors='pt',
                       padding=False, truncation=False)
    grids = inputs['image_grid_thw'].tolist()
    ip = processor.image_processor
    patch, merge = int(ip.patch_size), int(ip.merge_size)
    visual_tokens = sum(int(t * h * w) // (merge * merge) for t, h, w in grids)
    image_token = getattr(processor, 'image_token', '<|image_pad|>')
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
    actual_image_tokens = int((inputs['input_ids'] == image_token_id).sum())
    if actual_image_tokens != visual_tokens:
        raise ValueError(f'Visual token mismatch: grid={visual_tokens}, input={actual_image_tokens}')
    length = int(inputs['attention_mask'].sum())
    # 只用于统计输出长度，不加入prompt
    expected = {
        side: {
            "bbox": [int(round(v)) for v in item["bbox"]],
            "state": item["state"],
        }
        for side, item in sample["expected_detection"].items()
    }

    # 比当前答案更长的状态名组合，用作额外长度检查
    longer_case = {
        side: {"bbox": item["bbox"], "state": "occluded"}
        for side, item in expected.items()
    }

    def output_token_count(obj):
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        # 计入回答结束标记
        ids = processor.tokenizer.encode(
            text + "<|im_end|>", add_special_tokens=False
        )
        return len(ids)

    expected_tokens = output_token_count(expected)
    longer_case_tokens = output_token_count(longer_case)

    # 留32 tokens容纳少量空白和坐标分词差异；不是严格最坏情况保证
    estimated_output_budget = max(expected_tokens, longer_case_tokens) + 32
    max_new_tokens = max(
        128,
        ((estimated_output_budget + 31) // 32) * 32,
    )

    context_budget = 5600
    report = {
        'status': 'processor_only_no_model_inference',
        'utc': datetime.now(timezone.utc).isoformat(),
        'transformers': transformers.__version__,
        'model': str(model), 'sample': str(sample_path), 'image': str(image_path),
        'sample_sha256': hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'instruction': instruction, 'enable_thinking': False,
        'resize_policy': 'local_processor_defaults_no_crop_no_truncation',
        'processor_size': getattr(ip, 'size', None),
        'original_wh': list(image.size), 'image_grid_thw': grids,
        'processed_wh': [[int(w * patch), int(h * patch)] for t, h, w in grids],
        'patch_size': patch, 'merge_size': merge,
        'visual_tokens': visual_tokens, 'input_tokens': length,
        'other_input_tokens': length - visual_tokens,
        'generation_reserve_tokens': max_new_tokens,
        'required_context_with_reserve': length + max_new_tokens,
        'fits_budget_context': length + max_new_tokens <= context_budget,
        'tensor_shapes': {k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')},
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / ('processor_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    text = json.dumps(report, ensure_ascii=False, indent=2)
    with target.open('x', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)
    print(f'\nSaved: {target}')


if __name__ == '__main__':
    main()