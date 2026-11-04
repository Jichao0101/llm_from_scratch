"""S02共享负载：冻结输入、离线加载、图像预处理与greedy生成；无测量或诊断。"""
import json
from pathlib import Path

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


GENERATION_SEED = 42
def load_dataset(dataset, split):
    dataset = Path(dataset)
    meta = json.loads((dataset/'dataset.json').read_text())
    if meta['status'] != 'built':
        raise ValueError('Dataset is not successfully built')
    info = meta['manifests'][split]
    manifest = dataset/info['file']
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must be nonempty with unique sample IDs')
    if any(row['split'] != split for row in rows):
        raise ValueError('Wrong split in manifest')
    if any(not Path(row['image']).is_absolute() for row in rows):
        raise ValueError('Manifest images must use absolute paths; rebuild dataset')
    return meta, rows



def load_fixed_samples(manifest):
    """读取可迁移的固定性能样本；image相对清单目录，运行时解析为绝对路径。"""
    manifest = Path(manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get('sample_id'), str) or not row['sample_id'] for row in rows):
        raise ValueError('Samples must be nonempty with nonempty string sample IDs')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Sample IDs must be unique')
    images = set()
    for row in rows:
        if not isinstance(row.get('image'), str) or not row['image']:
            raise ValueError('Each sample needs an image path')
        image = Path(row['image'])
        if image.is_absolute():
            raise ValueError('Sample images must be relative to the manifest directory')
        resolved = (manifest.parent / image).resolve()
        if not resolved.is_relative_to(manifest.parent):
            raise ValueError('Sample image must stay within the sample directory')
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved in images:
            raise ValueError('Sample image paths must be unique')
        images.add(resolved)
        row['image'] = str(resolved)
    return rows


def load_model(model_path, torch, *, attn_implementation):
    """返回模型和processor；CUDA设备和seed由入口设置。"""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer needs an explicit pad token for batch generation')
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation=attn_implementation).eval()
    return model, processor


def make_prompt(processor):
    """固定非思考prompt；与原脚本保持在模型加载计时之外构造。"""
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT_TEMPLATE}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt


def load_images(rows):
    """返回RGB图片；成功时调用者关闭，部分读图失败时在此关闭已打开的图片。"""
    from PIL import Image
    images = []
    try:
        for row in rows:
            with Image.open(row['image']) as source:
                images.append(source.convert('RGB'))
        return images
    except BaseException:
        for image in images:
            image.close()
        raise


def prepare_inputs(images, processor, prompt):
    """保留CPU BatchEncoding供入口检查预算、记录元数据并决定传输范围。"""
    return processor(text=[prompt] * len(images), images=images, return_tensors='pt',
                     padding=True, truncation=False)


def generate(model, processor, inputs, max_new_tokens, torch):
    """每次generate创建自己的cache；不传入上次生成的past_key_values。"""
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              num_beams=1, use_cache=True, return_dict_in_generate=False,
                              pad_token_id=processor.tokenizer.pad_token_id)


def read_diagnostic_config(path):
    """两个短诊断入口共用固定负载配置，不接受原扫描器的mode/DP参数。"""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    required = {'sample_manifest', 'model', 'out', 'gpu', 'batch_size',
                'warmup_batches', 'capture_batches', 'context_budget', 'max_new_tokens',
                'cpu_threads_per_worker', 'attn_implementation'}
    if set(config) != required:
        raise ValueError(f'Expected exactly diagnostic fields; missing={required-set(config)}, extra={set(config)-required}')
    for key in ('batch_size', 'warmup_batches', 'capture_batches', 'context_budget',
                'max_new_tokens', 'cpu_threads_per_worker'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if config['max_new_tokens'] >= config['context_budget']:
        raise ValueError('max_new_tokens must be smaller than context_budget')
    if not isinstance(config['gpu'], str) or not config['gpu'] or ',' in config['gpu']:
        raise ValueError('gpu must select exactly one device ID/UUID')
    if not isinstance(config['attn_implementation'], (str, dict)) or not config['attn_implementation']:
        raise ValueError('attn_implementation must be an explicit backend string or subconfig mapping')
    for key in ('sample_manifest', 'model', 'out'):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(key + ' must be a nonempty path')
        value = Path(config[key]).expanduser()
        config[key] = str((value if value.is_absolute() else path.parent / value).resolve())
    return config


def prepare_fixed_batch(rows, processor, prompt, config):
    """完整准备CPU输入并关闭图片；短诊断复用，原测量入口保留分阶段编排。"""
    images = load_images(rows)
    try:
        inputs = prepare_inputs(images, processor, prompt)
    finally:
        for image in images:
            image.close()
    width = inputs['input_ids'].shape[-1]
    if width + config['max_new_tokens'] > config['context_budget']:
        raise ValueError('Padded input + output budget exceeded')
    metadata = {'sample_ids': [r['sample_id'] for r in rows],
                'input_tokens': inputs['attention_mask'].sum(dim=-1).tolist(),
                'padded_input_tokens': width,
                'image_grid_thw': inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else None,
                'input_shapes': {k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}}
    return inputs, metadata
