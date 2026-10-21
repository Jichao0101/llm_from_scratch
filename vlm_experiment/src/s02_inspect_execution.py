"""核验S02实际attention路径；同步、包装和日志仅用于诊断，不报告吞吐。"""
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from functools import wraps
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch

from s02_workload import (GENERATION_SEED, PROMPT_TEMPLATE, read_diagnostic_config,
                          load_fixed_samples, load_model, make_prompt, prepare_fixed_batch, generate)


def _identity(value):
    return getattr(value, '__module__', type(value).__module__) + '.' + getattr(value, '__qualname__', type(value).__qualname__)


def _metadata(value):
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device),
                'stride': list(value.stride())}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return {'type': _identity(type(value))}


def runtime_report(model, processor, torch, requested, output):
    """配置与源码是路径线索；动态调用单独记录，不据安装包推断已使用后端。"""
    packages = {}
    for name in ('torch', 'transformers', 'accelerate', 'flash-attn', 'flash-linear-attention', 'causal-conv1d'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        driver = {'unavailable': str(exc)}
    props = torch.cuda.get_device_properties(0)
    modules, sources, unavailable = [], [], []
    seen = set()
    for name, module in model.named_modules():
        if not (name in ('', 'model.visual', 'model.language_model') or
                name.endswith(('.attn', '.self_attn', '.linear_attn'))):
            continue
        config = getattr(module, 'config', None)
        item = {'module': name, 'class': _identity(type(module)),
                'configured_backend': getattr(config, '_attn_implementation', None),
                'model_type': getattr(config, 'model_type', None),
                'training': module.training,
                'callable_bindings': {key: _identity(getattr(module, key)) for key in
                    ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule', 'recurrent_gated_delta_rule')
                    if callable(getattr(module, key, None))}}
        modules.append(item)
        implementation = inspect.getmodule(type(module))
        helpers = [getattr(implementation, 'is_flash_attention_requested', None)]
        registry = getattr(implementation, 'ALL_ATTENTION_FUNCTIONS', None)
        helpers.append(getattr(registry, 'get_interface', None))
        for function in helpers:
            if function is not None and function not in seen:
                seen.add(function)
                try:
                    sources.append('# ' + _identity(function) + '\n' + inspect.getsource(function))
                except (OSError, TypeError) as exc:
                    unavailable.append({'function': _identity(function), 'error': str(exc)})
        if type(module) not in seen:
            seen.add(type(module))
            try:
                sources.append('# ' + item['class'] + '\n' + inspect.getsource(type(module)))
            except (OSError, TypeError) as exc:
                unavailable.append({'class': item['class'], 'error': str(exc)})
    (output / 'implementation.py.txt').write_text('\n\n'.join(sources), encoding='utf-8')
    return {'python': sys.version, 'packages': packages, 'cuda_runtime': torch.version.cuda,
            'driver_inventory': driver, 'gpu_name': props.name, 'gpu_compute_capability': [props.major, props.minor],
            'gpu_total_bytes': props.total_memory, 'visible_device': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'requested_backend': requested,
            'model_revision': getattr(model.config, '_commit_hash', None),
            'model_revision_note': 'null means unknown; local directory name is not a revision',
            'model_config': model.config.to_dict(),
            'dtype': str(model.dtype), 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'compiled': hasattr(model, '_orig_mod'),
            'processor_image_settings': processor.image_processor.to_dict(),
            'padding_side': processor.tokenizer.padding_side,
            'generation_config': model.generation_config.to_dict(),
            'generation_overrides': {'do_sample': False, 'num_beams': 1, 'use_cache': True,
                                     'return_dict_in_generate': False},
            'modules': modules, 'source_unavailable': unavailable,
            'limits': ['Installed/configured backend is not proof of an executed CUDA implementation.',
                       'Alternative backend compatibility requires checking this installed source and a separate smoke run.',
                       'Python call counts and CUDA kernel counts must be associated through trace; they need not match.']}


@contextmanager
def attention_calls(model, torch, records):
    """只包装当前模型使用的分发入口及线性attention绑定；退出时恢复，包括异常路径。"""
    targets = {module: name for name, module in model.named_modules()
               if name.startswith(('model.visual.', 'model.language_model.'))
               and name.endswith(('.attn', '.self_attn', '.linear_attn'))}
    if not targets or not any(name.startswith('model.visual.') for name in targets.values()):
        raise ValueError('Expected Qwen3.5 vision/text attention module paths')
    active, sequence, handles = {}, [0], []

    def before(module, args, kwargs):
        sequence[0] += 1
        bound = inspect.signature(module.forward).bind(*args, **kwargs).arguments
        record = {'event': 'module', 'module': targets[module], 'invocation': sequence[0],
                  'class': _identity(type(module)),
                  'inputs': {k: _metadata(v) for k, v in bound.items()}, 'interface_calls': 0}
        cu = bound.get('cu_seqlens')
        if cu is not None:
            boundaries = cu.detach().cpu().tolist()
            record['cu_seqlens'] = boundaries
            record['sequence_lengths'] = [b-a for a, b in zip(boundaries, boundaries[1:])]
        scope = torch.profiler.record_function('inspect/' + targets[module] + '/' + str(sequence[0]))
        scope.__enter__()
        active[module] = (record, scope)
        records.append(record)

    def after(module, args, kwargs, result):
        item = active.pop(module, None)
        if item is not None:
            item[0]['returned'] = result is not None
            item[1].__exit__(None, None, None)

    def wrap(function, owner=None, binding=None):
        @wraps(function)
        def observed(*args, **kwargs):
            module = owner if owner is not None else (args[0] if args else kwargs.get('module'))
            if module not in active:
                return function(*args, **kwargs)
            parent = active[module][0]
            parent['interface_calls'] += 1
            values = args if owner is not None else args[1:]
            entry = {'event': 'call', 'module': targets[module], 'invocation': parent['invocation'],
                     'call_index': parent['interface_calls'], 'function': _identity(function), 'binding': binding,
                     'args': [_metadata(v) for v in values],
                     'kwargs': {k: _metadata(v) for k, v in kwargs.items()}, 'status': 'started'}
            for key in ('cu_seq_lens_q', 'cu_seq_lens_k'):
                if key in kwargs and kwargs[key] is not None:
                    entry[key] = kwargs[key].detach().cpu().tolist()
            records.append(entry)
            with torch.profiler.record_function('inspect/attention_call/' + str(parent['invocation']) + '/' + str(parent['interface_calls'])):
                try:
                    result = function(*args, **kwargs)
                    entry['status'] = 'complete'
                    return result
                except BaseException:
                    entry['status'] = 'failed'
                    raise
        return observed

    try:
        with ExitStack() as patches:
            registries = {}
            for module, name in targets.items():
                implementation = inspect.getmodule(type(module))
                registry = getattr(implementation, 'ALL_ATTENTION_FUNCTIONS', None)
                if not name.endswith('.linear_attn'):
                    if registry is None or not callable(getattr(registry, 'get_interface', None)):
                        raise ValueError('Unsupported attention dispatcher; inspect cloud implementation before adapting wrapper')
                    registries[id(registry)] = registry
                else:
                    count = 0
                    for key in ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule', 'recurrent_gated_delta_rule'):
                        function = getattr(module, key, None)
                        if callable(function):
                            patches.enter_context(patch.object(module, key, wrap(function, module, key)))
                            count += 1
                    if not count:
                        raise ValueError('Unsupported linear attention bindings')
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))
            for registry in registries.values():
                original = registry.get_interface
                def resolve(*args, _original=original, **kwargs):
                    return wrap(_original(*args, **kwargs))
                patches.enter_context(patch.object(registry, 'get_interface', resolve))
            yield
    finally:
        for handle in handles:
            handle.remove()
        for record, scope in active.values():
            record['returned'] = False
            scope.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().with_suffix('.json')),
                        help='JSON config (default: %(default)s)')
    parser.add_argument('--trace', action='store_true', help='Also capture short CPU/CUDA trace for association; adds overhead')
    args = parser.parse_args()
    config = read_diagnostic_config(args.config)
    rows = load_fixed_samples(config['dataset'], config['sample_manifest'])
    needed = config['batch_size'] * config['capture_batches']
    if len(rows) < needed:
        raise ValueError('Not enough frozen samples for complete capture batches')
    rows = rows[:needed]
    output = Path(config['out'])
    output.mkdir(parents=True, exist_ok=False)
    records = []
    result = {'status': 'starting', 'config': config, 'generation_seed': GENERATION_SEED,
              'prompt': PROMPT_TEMPLATE, 'batches': [], 'trace_enabled': args.trace}
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Expected exactly one visible CUDA GPU')
        torch.set_num_threads(config['cpu_threads_per_worker'])
        torch.manual_seed(GENERATION_SEED)
        torch.cuda.manual_seed_all(GENERATION_SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        model, processor = load_model(config['model'], torch,
                                              attn_implementation=config['attn_implementation'])
        prompt = make_prompt(processor)
        result['runtime'] = runtime_report(model, processor, torch, config['attn_implementation'], output)
        for index in range(config['capture_batches']):
            batch = rows[index * config['batch_size']:(index + 1) * config['batch_size']]
            inputs, metadata = prepare_fixed_batch(batch, processor, prompt, config)
            inputs = inputs.to('cuda:0')
            for _ in range(config['warmup_batches']):
                warm = generate(model, processor, inputs, config['max_new_tokens'], torch)
                del warm
            torch.cuda.synchronize()
            start = len(records)
            with ExitStack() as stack:
                profiler = None
                if args.trace:
                    profiler = stack.enter_context(torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True, profile_memory=False, with_stack=False))
                stack.enter_context(attention_calls(model, torch, records))
                outputs = generate(model, processor, inputs, config['max_new_tokens'], torch)
                torch.cuda.synchronize()
            if profiler is not None:
                profiler.export_chrome_trace(str(output / f'trace_batch{index}.json'))
            for entry in records[start:]:
                entry['batch_index'] = index
            metadata['generated_padded_steps'] = outputs.shape[-1] - metadata['padded_input_tokens']
            metadata['output_token_ids'] = outputs[:, metadata['padded_input_tokens']:].cpu().tolist()
            result['batches'].append(metadata)
            del inputs, outputs
        modules = [r for r in records if r['event'] == 'module']
        if not modules:
            raise ValueError('No target attention module executed inside diagnostic window')
        result['vision_invocations'] = [
            {'batch_index': r['batch_index'], 'module': r['module'], 'invocation': r['invocation'],
             'sequence_lengths': r.get('sequence_lengths'), 'attention_function_calls': r['interface_calls']}
            for r in modules if r['module'].startswith('model.visual.')]
        missing = [r['module'] for r in modules if r['interface_calls'] == 0]
        if missing:
            raise ValueError('Unobserved dispatch inside modules: ' + ', '.join(sorted(set(missing))))
        result['call_counts'] = dict(Counter(r['module'] for r in records if r['event'] == 'call'))
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        (output / 'attention_calls.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        (output / 'execution.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
