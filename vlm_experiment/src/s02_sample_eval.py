"""用固定验证样本扫描batch、独立profile及单卡/多卡分片吞吐；不替代正式baseline。"""
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager, nullcontext
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

from experiment_profiler import MemoryProfiler, OperatorProfiler, TimingProfiler, memory_stats
from PIL import Image
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
OPERATOR_SETTINGS = {'record_shapes': True, 'profile_memory': True, 'with_stack': False}


def atomic_json(path, value):
    """先写同目录临时文件再替换，避免中断留下半份case或summary。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def summarize_inference(rows, cases):
    """仅统计执行覆盖和生成失败，不读取GT或计算精度。"""
    return {
        'expected_images': len(rows), 'completed_images': len(cases),
        'generated_images': sum(c['status'] == 'generated' for c in cases.values()),
        'errors': dict(Counter(c.get('error_type', 'unknown') for c in cases.values() if c['status'] != 'generated')),
        'hit_token_limit_images': sum(c.get('hit_token_limit', False) for c in cases.values()),
    }


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


def read_config(path):
    """只读取真实实验变量；按模式/样本来源检查必填项，不合并默认配置。"""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    obsolete = {'gpu', 'dp', 'profiler', 'seed'} & config.keys()
    if obsolete:
        raise ValueError('Obsolete config fields; migrate using SAMPLE_EVAL.md: ' + ', '.join(sorted(obsolete)))
    required = {'dataset', 'out', 'model', 'mode', 'sample_manifest', 'batch_sizes',
                'gpus', 'warmup_batches', 'context_budget', 'max_new_tokens', 'cpu_threads_per_worker'}
    missing = required - config.keys()
    if missing:
        raise ValueError('Missing config fields: ' + ', '.join(sorted(missing)))
    mode = config['mode']
    if mode not in ('benchmark', 'timing', 'memory', 'operator'):
        raise ValueError('mode must be benchmark, timing, memory or operator')
    if config['sample_manifest'] is not None and (not isinstance(config['sample_manifest'], str) or not config['sample_manifest']):
        raise ValueError('sample_manifest must be null or a nonempty path')
    if not config['sample_manifest']:
        required = {'sample_count', 'sample_seed'}
        missing = required - config.keys()
        if missing:
            raise ValueError('Random sampling requires: ' + ', '.join(sorted(missing)))
        if config['sample_count'] < 1:
            raise ValueError('sample_count must be positive')
    else:
        # 清单决定本轮样本；不将未参与运行的随机抽样参数写成生效配置。
        config.pop('sample_count', None)
        config.pop('sample_seed', None)
    if min(config[k] for k in ('warmup_batches', 'context_budget', 'max_new_tokens', 'cpu_threads_per_worker')) < 1:
        raise ValueError('Warmup, token budgets and CPU threads must be positive')
    if config['max_new_tokens'] >= config['context_budget']:
        raise ValueError('max_new_tokens must be smaller than context_budget')
    sizes = config['batch_sizes']
    if not sizes or min(sizes) < 1 or sizes != sorted(set(sizes)):
        raise ValueError('Increasing positive unique batch_sizes required')
    gpus = config['gpus']
    if (not isinstance(gpus, list) or not gpus or
            any(not isinstance(g, str) or not g or ',' in g for g in gpus) or len(set(gpus)) != len(gpus)):
        raise ValueError('gpus must contain unique nonempty GPU ID/UUID strings')
    if mode == 'benchmark':
        config.pop('capture_batches', None)
        if len(gpus) == 1:
            config.pop('dp_batch_size', None)
        if config['repeats'] < 1:
            raise ValueError('Positive repeats required')
        if len(gpus) > 1 and config['dp_batch_size'] not in sizes:
            raise ValueError('dp_batch_size must be in batch_sizes for the single-GPU reference')
    else:
        config.pop('repeats', None)
        config.pop('dp_batch_size', None)
        if len(gpus) != 1:
            raise ValueError('Timing, memory and operator diagnostics require exactly one GPU')
        if config['capture_batches'] < 1:
            raise ValueError('capture_batches must be positive')
    for key in ('dataset', 'out', 'model', 'sample_manifest'):
        if key == 'sample_manifest' and config[key] is None:
            continue
        target = Path(config[key]).expanduser()
        config[key] = str((target if target.is_absolute() else path.parent / target).resolve())
    return config


def select_samples(rows, count, seed):
    """按种子均匀随机无放回采样；排序仅使清单行重排不改变选样结果。"""
    candidates = sorted(rows, key=lambda row: row['sample_id'])
    return random.Random(seed).sample(candidates, min(count, len(candidates)))


def split_samples(rows, workers):
    """每个sample只进入一个rank；各rank维持自身样本顺序，允许不同尾批大小。"""
    if len(rows) < workers:
        raise ValueError('Sample count must be >= worker count; empty ranks cannot measure DP')
    return [rows[rank::workers] for rank in range(workers)]


def generated_tokens(tokens, eos_ids, pad_id):
    """保留第一个EOS，去掉batch中其他样本继续生成产生的尾padding。"""
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index + 1], True
    while tokens and tokens[-1] == pad_id:
        tokens.pop()
    return tokens, False


@contextmanager
def generation_profile(model, collector, operator=None):
    """Qwen3.5非思考greedy生成的模块边界；仅挂局部hook，不改forward或generate。

    视觉、语言主干和输出投影是相邻阶段，互不包含。语言阶段依据进入forward
    前的缓存长度和输入长度识别；模块路径不匹配时显式失败，不退回调用次序猜测。
    hook不保留输入/输出tensor；finally移除全部handle并关闭异常中的活动标记。
    """
    modules = dict(model.named_modules())
    paths = ('model.visual', 'model.language_model', 'lm_head')
    if any(path not in modules for path in paths):
        raise ValueError('Generation profiling requires Qwen3.5 visual/language_model/lm_head modules')
    handles, active = [], {}
    language_phase = None
    language_step = 0

    def before(module, args, kwargs):
        nonlocal language_phase, language_step
        path = next(path for path in paths if modules[path] is module)
        metadata = {'module': path}
        if path == 'model.language_model':
            tensor = kwargs.get('inputs_embeds')
            if tensor is None:
                tensor = kwargs.get('input_ids')
            if tensor is None:
                raise ValueError('Cannot identify language input length')
            length = tensor.shape[1]
            cache = kwargs.get('past_key_values')
            if cache is not None and not hasattr(cache, 'get_seq_length'):
                raise ValueError('Cannot identify language cache length')
            cached = int(cache.get_seq_length()) if cache is not None else 0
            if cached == 0:
                language_phase = 'language_prefill'
            elif length == 1:
                language_phase = 'language_decode'
            else:
                raise ValueError('Cached multi-token language call is outside greedy profiling contract')
            name = language_phase
            metadata.update(sequence_length=length, cached_tokens=cached, forward_index=language_step)
            language_step += 1
        elif path == 'model.visual':
            name = 'vision'
        else:
            if language_phase is None:
                raise ValueError('Output projection called before language forward')
            name = 'prefill_logits' if language_phase == 'language_prefill' else 'decode_logits'
            metadata['forward_index'] = language_step - 1
        context = ExitStack()
        try:
            if operator is not None:
                context.enter_context(operator.region('experiment/' + name))
            if collector is not None:
                context.enter_context(collector.measure(name, **metadata))
        except BaseException:
            context.close()
            raise
        active[module] = context

    def after(module, args, kwargs, output):
        context = active.pop(module, None)
        if context is not None:
            # 这三个目标模块成功时均返回非None；always_call让forward异常也关闭标记。
            failure = RuntimeError('Profiled module forward did not return output') if output is None else None
            context.__exit__(type(failure) if failure else None, failure, None)

    try:
        for path in paths:
            module = modules[path]
            handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for context in active.values():
            failure = RuntimeError('Generation profiling interrupted during module forward')
            context.__exit__(type(failure), failure, None)


def infer_batch(rows, model, processor, prompt, config, torch, profiling=False, operator=None):
    """一次真实batch generate；计时含读图、处理、传输、生成、输出解码，不含写盘。"""
    cases = {r['sample_id']: {'sample_id': r['sample_id'], 'image': r['image'],
                             'raw_output': None, 'status': 'failed'} for r in rows}
    stages, images, stage_memory = {}, [], {}
    inputs = outputs = None
    stage = 'image'
    torch.cuda.synchronize()
    started = time.perf_counter()
    # 三种测量互斥：普通计时不重置显存峰值，显存测量不启动算子追踪。
    measurement = config['mode'] if profiling else None
    pipeline_memory = MemoryProfiler(torch) if measurement == 'memory' else None

    @contextmanager
    def mark(name):
        if not profiling and operator is None:
            yield
            return
        try:
            with ExitStack() as contexts:
                if operator is not None:
                    contexts.enter_context(operator.region('pipeline/' + name))
                if pipeline_memory is not None:
                    contexts.enter_context(pipeline_memory.measure(name))
                yield
        finally:
            if pipeline_memory is not None and pipeline_memory.records:
                record = pipeline_memory.records[-1]
                if record['phase'] == name:
                    stage_memory[name] = {key: value for key, value in record.items()
                                          if key not in ('phase', 'metadata', 'status')}

    try:
        tick = time.perf_counter()
        with mark('read_images'):
            for row in rows:
                with Image.open(row['image']) as source:
                    images.append(source.convert('RGB'))
        stages['read_images_us'] = (time.perf_counter() - tick) * 1e6
        batch = {'requested_images': len(rows), 'generated_images': 0, 'stage_us': stages}
        if pipeline_memory is not None:
            batch['stage_memory'] = stage_memory

        stage = 'processor'
        tick = time.perf_counter()
        with mark('processor'):
            inputs = processor(text=[prompt] * len(images), images=images, return_tensors='pt',
                               padding=True, truncation=False)
            width = inputs['input_ids'].shape[-1]
            # 不过滤样本、不缩小实际batch；预算不足时整份配置失败。
            stage = 'context_budget'
            if width + config['max_new_tokens'] > config['context_budget']:
                raise ValueError('Padded input + output budget exceeded')
            stage = 'processor'
            lengths = inputs['attention_mask'].sum(dim=-1).tolist()
            for row, length in zip(rows, lengths):
                cases[row['sample_id']].update(input_tokens=length, padded_input_tokens=width)
        stages['processor_us'] = (time.perf_counter() - tick) * 1e6

        stage = 'transfer'
        tick = time.perf_counter()
        with mark('host_to_device'):
            inputs = inputs.to('cuda:0')
            torch.cuda.synchronize()
        stages['host_to_device_us'] = (time.perf_counter() - tick) * 1e6
        stage = 'generate'
        tick = time.perf_counter()
        with mark('generate'), torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=config['max_new_tokens'], do_sample=False,
                                     num_beams=1, use_cache=True, return_dict_in_generate=False,
                                     pad_token_id=processor.tokenizer.pad_token_id)
            torch.cuda.synchronize()
        stages['generate_us'] = (time.perf_counter() - tick) * 1e6
        if outputs.shape[0] != len(rows):
            raise ValueError('Generated batch size differs from input batch')
        batch['generated_images'] = len(rows)
        stage = 'decode'
        tick = time.perf_counter()
        with mark('decode_outputs'):
            # 所有输出按补齐后的输入宽度截取，不能用各sample的未padding长度。
            tails = outputs[:, width:].cpu().tolist()
            eos = model.generation_config.eos_token_id
            if eos is None:
                eos = processor.tokenizer.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            for row, tail in zip(rows, tails):
                case = cases[row['sample_id']]
                tokens, ended = generated_tokens(tail, eos_ids, processor.tokenizer.pad_token_id)
                case.update(output_tokens=len(tokens), hit_token_limit=not ended and len(tokens) >= config['max_new_tokens'],
                            generated_padded_steps=outputs.shape[-1] - width)
                raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                case['raw_output'] = raw
                case['status'] = 'generated'  # 仅确认生成/解码完成，不判断答案是否正确。
        stages['decode_us'] = (time.perf_counter() - tick) * 1e6
        batch['pipeline_us'] = (time.perf_counter() - started) * 1e6
        batch['input_tokens'] = sum(c.get('input_tokens', 0) for c in cases.values())
        batch['output_tokens'] = sum(c.get('output_tokens', 0) for c in cases.values())
        batch['padded_input_tokens'] = width
        return cases, batch
    except Exception as exc:
        # 配置级OOM/processor/generate故障由worker中止，不拆成小批伪装成成功。
        for row in rows:
            if cases[row['sample_id']]['status'] != 'generated':
                cases[row['sample_id']].update(error_type=stage, error='{}: {}'.format(type(exc).__name__, exc))
        exc.batch_cases = cases
        exc.batch_measurement = {'failed_stage': stage, 'stage_us': stages, 'stage_memory': stage_memory,
                                 'elapsed_until_failure_us': (time.perf_counter() - started) * 1e6}
        raise
    finally:
        inputs = outputs = None
        for image in images:
            image.close()


def runtime_context(torch, model, processor, gpu):
    import transformers
    properties = torch.cuda.get_device_properties(0)
    return {
        'gpu_id': gpu, 'gpu_name': properties.name, 'gpu_total_bytes': properties.total_memory,
        'python': sys.version, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'transformers': transformers.__version__, 'model_class': type(model).__name__,
        'processor_image_settings': processor.image_processor.to_dict(),
        'padding_side': processor.tokenizer.padding_side,
    }


def write_cases(path, cases):
    path.write_text(''.join(json.dumps(case, ensure_ascii=False, allow_nan=False) + '\n'
                            for case in cases.values()), encoding='utf-8')


def worker(config, rows, gpu, batch_size, rank, output, barrier, profiling):
    """一个进程只拥有一张卡、一份模型；进程退出释放显存，不污染后续batch配置。"""
    output = Path(output)
    result = {'rank': rank, 'gpu': gpu, 'batch_size': batch_size, 'status': 'starting', 'repeats': []}
    current_cases = {}
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Expected exactly one visible CUDA GPU per worker')
        torch.set_num_threads(config['cpu_threads_per_worker'])
        torch.manual_seed(GENERATION_SEED)
        torch.cuda.manual_seed_all(GENERATION_SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        tick = time.perf_counter()
        processor = AutoProcessor.from_pretrained(config['model'], local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        if processor.tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer needs an explicit pad token for batch generation')
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            config['model'], local_files_only=True, dtype=torch.bfloat16,
            device_map={'': 0}, attn_implementation='sdpa').eval()
        torch.cuda.synchronize()
        result['load_us'] = (time.perf_counter() - tick) * 1e6
        result['runtime'] = runtime_context(torch, model, processor, gpu)
        result['loaded_memory'] = memory_stats(torch)
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT_TEMPLATE}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
        # 首次调用单列，随后预热；预热失败也终止该配置，不能计入稳定吞吐。
        _, result['cold_batch'] = infer_batch(batches[0], model, processor, prompt, config, torch)
        for index in range(config['warmup_batches']):
            infer_batch(batches[index % len(batches)], model, processor, prompt, config, torch)
        if profiling:
            profiled_rows = rows[:batch_size * config['capture_batches']]
            measurements = []
            target = config['mode']
            result.update(measurement=target, profile_batches=measurements)
            generation_reports = []
            if target == 'timing':
                result['generation_timing'] = generation_reports
            # Operator窗口只在显式选择operator时创建；其他模式不调用torch.profiler。
            profile = OperatorProfiler(torch, OPERATOR_SETTINGS) if target == 'operator' else None
            with profile.capture() if profile is not None else nullcontext():
                for start in range(0, len(profiled_rows), batch_size):
                    collector = TimingProfiler(torch) if target == 'timing' else None
                    try:
                        hooks = generation_profile(model, collector, profile) if target != 'memory' else nullcontext()
                        with hooks:
                            cases, measurement = infer_batch(profiled_rows[start:start + batch_size], model, processor,
                                                            prompt, config, torch, profiling=True, operator=profile)
                    finally:
                        if collector is not None:
                            collector.resolve()
                            generation_reports.append({'batch_index': len(measurements), **collector.report()})
                    current_cases.update(cases)
                    measurements.append(measurement)
                    if profile is not None:
                        profile.step()
            if profile is not None:
                profile.export(output)
            write_cases(output / 'profile_outputs.jsonl', current_cases)
            if target == 'memory':
                # 生成内部峰值另跑，禁止嵌套pipeline显存区间；不会顺带采集算子。
                result['generation_memory'] = generation_reports
                for index, start in enumerate(range(0, len(profiled_rows), batch_size)):
                    collector = MemoryProfiler(torch)
                    try:
                        with generation_profile(model, collector):
                            memory_cases, _ = infer_batch(profiled_rows[start:start + batch_size], model, processor,
                                                          prompt, config, torch, profiling=False)
                    finally:
                        generation_reports.append({'batch_index': index, **collector.report()})
                    generation_reports[-1]['output_comparison'] = {
                        'images': len(memory_cases),
                        'different_raw_outputs': sum(memory_cases[sid]['raw_output'] != current_cases[sid]['raw_output']
                                                     for sid in memory_cases),
                        'pipeline_output_tokens': sum(current_cases[sid]['output_tokens'] for sid in memory_cases),
                        'generation_output_tokens': sum(v['output_tokens'] for v in memory_cases.values()),
                    }
            if target != 'operator':
                atomic_json(output / 'generation_profile.json', {
                    'measurement': target, 'batches': generation_reports,
                    'note': 'Selected measurement only. Memory synchronizes module boundaries and is not throughput. '
                            'Module regions do not cover all of generate; peaks must not be added.',
                })
            full_batches = [m for m in measurements if m['requested_images'] == batch_size]
            result['full_profile_batches'] = len(full_batches)
            result['stage_summary'] = {}
            # 两类结果分别汇总，不在无显存测量时伪造峰值或headroom。
            if target == 'timing':
                # timing固定解释generate内部，基础pipeline耗时由benchmark负责。
                calls = [call for report in generation_reports
                         if measurements[report['batch_index']]['requested_images'] == batch_size
                         for call in report['calls'] if call['status'] == 'complete']
                result['stage_summary'] = {
                    name: {'completed_calls': sum(c['phase'] == name for c in calls),
                           'mean_host_elapsed_us': statistics.mean(c['host_elapsed_us'] for c in calls if c['phase'] == name),
                           'mean_cuda_elapsed_us': statistics.mean(c['cuda_elapsed_us'] for c in calls if c['phase'] == name)}
                    for name in dict.fromkeys(c['phase'] for c in calls)}
            elif target == 'memory':
                fields = ('peak_allocated_bytes', 'peak_reserved_bytes',
                          'peak_allocated_increase_bytes', 'peak_reserved_increase_bytes')
                result['stage_summary'] = {
                    name: {'max_' + field: max(m['stage_memory'][name][field] for m in full_batches) for field in fields}
                    for name in full_batches[0]['stage_memory']}
                for field in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                    result['profile_' + field] = max(v[field] for m in measurements for v in m['stage_memory'].values())
                result['allocated_headroom_bytes'] = result['runtime']['gpu_total_bytes'] - result['profile_peak_allocated_bytes']
            result.update(status='profile_complete', inference=summarize_inference(profiled_rows, current_cases),
                          profiled_images=len(profiled_rows),
                          note='Timing, memory and operator are separate runs. Use benchmark for stable throughput. '
                               'Memory headroom excludes non-PyTorch allocations and other processes.')
        else:
            for repeat in range(config['repeats']):
                current_cases, measurements = {}, []
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                if barrier is not None:
                    barrier.wait(timeout=600)  # 所有模型预热后同时开始；每轮都对齐。
                started = time.perf_counter()
                for batch in batches:
                    cases, measurement = infer_batch(batch, model, processor, prompt, config, torch)
                    current_cases.update(cases)
                    measurements.append(measurement)
                torch.cuda.synchronize()
                finished = time.perf_counter()
                observed = {'repeat': repeat, 'started_monotonic_s': started, 'finished_monotonic_s': finished,
                            'elapsed_us': (finished - started) * 1e6, 'batches': measurements,
                            'memory': memory_stats(torch), 'inference': summarize_inference(rows, current_cases)}
                result['repeats'].append(observed)
                # 保存、日志、跨rank等待不在各轮稳定测量窗口中。
                write_cases(output / ('outputs_r{}.jsonl'.format(repeat)), current_cases)
                print('gpu={} B={} repeat={}/{} images={} seconds={:.3f}'.format(
                    gpu, batch_size, repeat + 1, config['repeats'], len(rows), observed['elapsed_us'] / 1e6), flush=True)
            result['status'] = 'complete'
    except Exception as exc:
        if profiling and 'profile' in locals() and profile is not None:
            try:
                profile.export(output)
            except Exception as export_error:
                result['operator_export_error'] = str(export_error)
        result.update(status='oom' if type(exc).__name__ in ('OutOfMemoryError', 'CUDAOutOfMemoryError') else 'failed',
                      error='{}: {}'.format(type(exc).__name__, exc), traceback=traceback.format_exc())
        current_cases.update(getattr(exc, 'batch_cases', {}))
        result['failed_batch'] = getattr(exc, 'batch_measurement', None)
        if 'torch' in locals() and torch.cuda.is_initialized():
            try:
                result['failure_memory'] = memory_stats(torch)
            except RuntimeError as memory_error:
                result['failure_memory_error'] = str(memory_error)
        write_cases(output / 'partial_outputs.jsonl', current_cases)
        atomic_json(output / 'result.json', result)
        if barrier is not None:
            barrier.abort()  # 先保存原始故障，再唤醒同组rank，避免取消竞争丢失OOM证据。
        return
    atomic_json(output / 'result.json', result)


def launch(config, rows, gpus, batch_size, output, profiling=False):
    """父进程不加载CUDA；spawn隔离模型生命周期，任一rank失败终止其余进程。"""
    output.mkdir()
    context = mp.get_context('spawn')
    shards = split_samples(rows, len(gpus))
    barrier = context.Barrier(len(gpus)) if len(gpus) > 1 else None
    processes, rank_dirs = [], []
    try:
        for rank, (gpu, shard) in enumerate(zip(gpus, shards)):
            rank_dir = output / ('rank{}'.format(rank))
            rank_dir.mkdir()
            rank_dirs.append(rank_dir)
            process = context.Process(target=worker, args=(config, shard, gpu, batch_size, rank, str(rank_dir), barrier, profiling))
            process.start()
            processes.append(process)
        pending = list(processes)
        while pending:
            for process in list(pending):
                process.join(timeout=0.1)
                if process.exitcode is not None:
                    pending.remove(process)
                    path = rank_dirs[processes.index(process)] / 'result.json'
                    success = (process.exitcode == 0 and path.exists() and
                               json.loads(path.read_text())['status'] in ('complete', 'profile_complete'))
                    if not success:
                        for other in pending:
                            other.terminate()
                        pending.clear()
                        break
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    results = []
    for rank, directory in enumerate(rank_dirs):
        path = directory / 'result.json'
        results.append(json.loads(path.read_text()) if path.exists() else
                       {'rank': rank, 'status': 'failed', 'error': 'Worker exited/cancelled; no completed result'})
    if profiling:
        atomic_json(output / 'result.json', results[0])
        return results[0]
    combined = {'batch_size_per_gpu': batch_size, 'gpus': gpus, 'status': 'complete', 'repeats': [],
                'rank_statuses': [r['status'] for r in results],
                'rank_errors': [r.get('error') for r in results]}
    if any(r['status'] != 'complete' for r in results):
        combined['status'] = 'oom' if any(r['status'] == 'oom' for r in results) else 'failed'
    else:
        for repeat in range(config['repeats']):
            cases, ranks = {}, [r['repeats'][repeat] for r in results]
            for directory in rank_dirs:
                for line in (directory / ('outputs_r{}.jsonl'.format(repeat))).read_text().splitlines():
                    case = json.loads(line)
                    if case['sample_id'] in cases:
                        raise ValueError('Duplicate sample across DP ranks: ' + case['sample_id'])
                    cases[case['sample_id']] = case
            if set(cases) != {r['sample_id'] for r in rows}:
                raise ValueError('DP result has missing or unexpected sample IDs')
            # 同一机器perf_counter共用时钟，覆盖最早开始至最晚结束及rank不平衡。
            wall_us = (max(r['finished_monotonic_s'] for r in ranks) - min(r['started_monotonic_s'] for r in ranks)) * 1e6
            batches = [b for r in ranks for b in r['batches']]
            generated = sum(b['generated_images'] for b in batches)
            generate_us = sum(b['stage_us'].get('generate_us', 0) for b in batches)
            combined['repeats'].append({
                'repeat': repeat, 'global_pipeline_us': wall_us,
                'attempted_images_per_s': len(rows) * 1e6 / wall_us,
                'generated_images_per_s': generated * 1e6 / wall_us,
                'generate_only_images_per_s_per_gpu_pooled': generated * 1e6 / generate_us if generate_us else None,
                'generate_only_output_tokens_per_s_per_gpu_pooled': (sum(b.get('output_tokens', 0) for b in batches) * 1e6 / generate_us if generate_us else None),
                'peak_allocated_bytes_max_rank': max(r['memory']['peak_allocated_bytes'] for r in ranks),
                'peak_reserved_bytes_max_rank': max(r['memory']['peak_reserved_bytes'] for r in ranks),
                'rank_peak_allocated_bytes': [r['memory']['peak_allocated_bytes'] for r in ranks],
                'gpu_seconds_global_window': wall_us / 1e6 * len(gpus),
                'input_tokens': sum(b.get('input_tokens', 0) for b in batches),
                'output_tokens': sum(b.get('output_tokens', 0) for b in batches),
                'rank_elapsed_us': [r['elapsed_us'] for r in ranks],
                'batch_pipeline_us': [b['pipeline_us'] for b in batches],
                'largest_requested_batch_images': max(b['requested_images'] for b in batches),
                'largest_generated_batch_images': max(b['generated_images'] for b in batches),
                'inference': summarize_inference(rows, cases),
                'sample_ids_verified': True,
            })
            write_cases(output / ('outputs_r{}.jsonl'.format(repeat)), cases)
        # benchmark始终报告基础计时；DP批统计跨rank池化，不等同全局批延迟。
        windows = [r['global_pipeline_us'] for r in combined['repeats']]
        observed_batches = [b for rank in results for repeat in rank['repeats'] for b in repeat['batches']]
        combined['mean_global_pipeline_us'] = statistics.mean(windows)
        combined['stdev_global_pipeline_us'] = statistics.stdev(windows) if len(windows) > 1 else 0.0
        combined['mean_batch_pipeline_us'] = statistics.mean(b['pipeline_us'] for b in observed_batches)
        combined['mean_stage_us_per_batch'] = {
            name: statistics.mean(b['stage_us'][name] for b in observed_batches)
            for name in observed_batches[0]['stage_us']}
        speeds = [r['generated_images_per_s'] for r in combined['repeats']]
        combined['mean_generated_images_per_s'] = statistics.mean(speeds)
        combined['stdev_generated_images_per_s'] = statistics.stdev(speeds) if len(speeds) > 1 else 0.0
    atomic_json(output / 'result.json', combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='JSON file containing all experiment parameters')
    args = parser.parse_args()
    config = read_config(args.config)
    meta, rows = load_dataset(config['dataset'], 'val')  
    if meta['purpose'] != 'formal_baseline':
        raise ValueError('GPU experiments require a formal_baseline manifest')
    if not Path(config['model']).is_dir():
        raise ValueError('Expected a local model directory; update config.model for this environment')
    if config['sample_manifest']:
        frozen = [json.loads(line) for line in Path(config['sample_manifest']).read_text().splitlines() if line.strip()]
        by_id = {row['sample_id']: row for row in rows}
        if not frozen or len({r['sample_id'] for r in frozen}) != len(frozen):
            raise ValueError('Frozen samples must be nonempty and unique')
        for row in frozen:
            current = by_id.get(row['sample_id'])
            if current is None or current['image'] != row['image']:
                raise ValueError('Frozen sample differs from current validation manifest')
        samples = frozen
    else:
        samples = select_samples(rows, config['sample_count'], config['sample_seed'])
    if config['mode'] != 'benchmark' and len(samples) < max(config['batch_sizes']):
        raise ValueError('Diagnostics require at least max(batch_sizes) samples; enlarge the sample selection or reduce batch_sizes')
    out = Path(config['out'])
    out.mkdir(parents=True, exist_ok=False)  # 不覆盖已存在结果；每轮使用新输出目录。
    effective = {**config, 'generation_seed': GENERATION_SEED,
                 'prompt_template': PROMPT_TEMPLATE, 'coordinate_scale': 'norm1000'}
    if config['mode'] == 'operator':
        effective['operator_settings'] = dict(OPERATOR_SETTINGS)
    atomic_json(out / 'config.json', effective)
    (out / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in samples), encoding='utf-8')
    sample_info = {'selected_images': len(samples), 'validation_images': len(rows),
                   'method': 'frozen_manifest' if config['sample_manifest'] else 'uniform_random_without_replacement'}
    if not config['sample_manifest']:
        sample_info['sample_seed'] = config['sample_seed']
    atomic_json(out / 'sample_info.json', sample_info)
    if config['mode'] != 'benchmark':
        # 相同batch档位/样本前缀与benchmark对齐；至少一整批才能证明该档真实容量。
        report = {'status': 'running', 'single_gpu': [], 'formal_baseline': False,
                  'measurement': config['mode'],
                  'selection_note': 'Inference profile only; training needs backward/optimizer measurements. '
                                    'Timing/memory scans precede optional operator tracing. Allocator headroom excludes other processes and non-PyTorch allocations.'}
        atomic_json(out / 'summary.json', report)
        try:
            for batch_size in config['batch_sizes']:
                result = launch(config, samples, [config['gpus'][0]], batch_size,
                                out / ('profile_b{}'.format(batch_size)), profiling=True)
                report['single_gpu'].append(result)
                atomic_json(out / 'summary.json', report)
                if result['status'] != 'profile_complete':
                    report.update(status='stopped', reason='OOM/runtime failure; larger batches not attempted')
                    break
            else:
                report['status'] = 'complete'
        except Exception as exc:
            report.update(status='failed', error='{}: {}'.format(type(exc).__name__, exc))
            raise
        finally:
            atomic_json(out / 'summary.json', report)
        if report['status'] != 'complete':
            raise RuntimeError('Profile scan failed; see summary.json and rank results')
        print('Batch profile comparison:', out / 'summary.json')
        return

    report = {'status': 'running', 'single_gpu': [], 'dp': None, 'formal_baseline': False}
    atomic_json(out / 'summary.json', report)
    try:
        for batch_size in config['batch_sizes']:
            directory = out / ('single_b{}'.format(batch_size))
            result = launch(config, samples, [config['gpus'][0]], batch_size, directory)
            report['single_gpu'].append(result)
            if result['status'] != 'complete':
                report.update(status='stopped', reason='OOM/runtime failure; larger batches not attempted')
                break
            reference = next((r for r in report['single_gpu'] if r['batch_size_per_gpu'] == 1), None)
            speed_b1 = reference['mean_generated_images_per_s'] if reference else None
            result['speedup_vs_b1'] = result['mean_generated_images_per_s'] / speed_b1 if speed_b1 else None
            atomic_json(out / 'summary.json', report)
        else:
            report['status'] = 'complete'
        if report['status'] == 'complete' and len(config['gpus']) > 1:
            batch_size = config['dp_batch_size']
            dp = launch(config, samples, config['gpus'], batch_size, out / 'dp')
            report['dp'] = dp
            if dp['status'] == 'complete':
                reference = next(r for r in report['single_gpu'] if r['batch_size_per_gpu'] == batch_size)
                speed = reference['mean_generated_images_per_s']
                dp['speedup_same_batch'] = dp['mean_generated_images_per_s'] / speed if speed else None
                dp['efficiency_same_batch'] = dp['speedup_same_batch'] / len(config['gpus']) if speed else None
            else:
                report['status'] = 'stopped'
        report['selection_note'] = 'Select using throughput variability, memory and trace; this experiment does not evaluate model accuracy.'
    except Exception as exc:
        report.update(status='failed', error='{}: {}'.format(type(exc).__name__, exc))
        raise
    finally:
        atomic_json(out / 'summary.json', report)
    if report['status'] != 'complete':
        raise RuntimeError('Experiment stopped: {}; see {}'.format(report['status'], out / 'summary.json'))
    print('Batch/DP comparison:', out / 'summary.json')


if __name__ == '__main__':
    main()
