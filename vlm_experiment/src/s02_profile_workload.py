"""固定S02生成负载，供ncu/nsys启动；先basic核对调用清单，再full采集指定实例。"""
import argparse
from contextlib import contextmanager
import json
import importlib.metadata
import os
import shlex
import subprocess
import sys
from pathlib import Path
import traceback

from s02_workload import (GENERATION_SEED, PROMPT_TEMPLATE, read_diagnostic_config,
                          load_fixed_samples, load_model, make_prompt, prepare_fixed_batch, generate)


@contextmanager
def _generation_nvtx_regions(model, nvtx):
    """沿用sample_eval的Qwen3.5阶段判定；只标记提交范围，不同步GPU。

    hook仅在采集窗口安装。模块或cache接口不匹配时失败，不按调用次数猜测；
    不保存tensor，正常返回及异常退出均移除hook并关闭活动范围。
    """
    modules = dict(model.named_modules())
    paths = ('model.visual', 'model.language_model', 'lm_head')
    if any(path not in modules for path in paths):
        raise ValueError('Generation profiling requires Qwen3.5 visual/language_model/lm_head modules')
    handles, active = [], []
    language_phase = None

    def before(module, args, kwargs):
        nonlocal language_phase
        if module is modules['model.language_model']:
            tensor = kwargs.get('inputs_embeds')
            if tensor is None:
                tensor = kwargs.get('input_ids')
            if tensor is None:
                raise ValueError('Cannot identify language input length')
            cache = kwargs.get('past_key_values')
            if cache is not None and not hasattr(cache, 'get_seq_length'):
                raise ValueError('Cannot identify language cache length')
            cached = int(cache.get_seq_length()) if cache is not None else 0
            if cached == 0:
                language_phase = 'language_prefill'
            elif tensor.shape[1] == 1:
                language_phase = 'language_decode'
            else:
                raise ValueError('Cached multi-token language call is outside greedy profiling contract')
            name = language_phase
        elif module is modules['model.visual']:
            name = 'vision'
        else:
            if language_phase is None:
                raise ValueError('Output projection called before language forward')
            name = 'prefill_logits' if language_phase == 'language_prefill' else 'decode_logits'
        nvtx.range_push(name)
        active.append(module)

    def after(module, args, kwargs, output):
        if active and active[-1] is module:
            active.pop()
            nvtx.range_pop()

    try:
        for path in paths:
            module = modules[path]
            handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        while active:
            active.pop()
            nvtx.range_pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().with_suffix('.json')),
                        help='JSON config (default: %(default)s)')
    parser.add_argument('--ncu', choices=('basic', 'full'), help='由ncu启动同一负载；省略时直接运行')
    parser.add_argument('--phase', choices=('vision', 'language_prefill', 'language_decode',
                                          'prefill_logits', 'decode_logits'))
    parser.add_argument('--kernel', help='ncu demangled名称过滤表达式，支持regex:前缀')
    parser.add_argument('--launch-skip', type=int, help='显式填写当前过滤条件下跳过的匹配数，允许0')
    parser.add_argument('--launch-count', type=int, help='显式填写待采集的连续匹配实例数')
    parser.add_argument('--report', help='ncu报告路径/前缀，每次使用新名称')
    parser.add_argument('--print-command', action='store_true', help='仅校验并打印ncu命令，不运行负载')
    args = parser.parse_args()
    selection = (args.phase, args.kernel, args.launch_skip, args.launch_count)
    if args.ncu is None:
        if args.print_command or args.report is not None or any(x is not None for x in selection):
            parser.error('ncu options require --ncu basic/full')
    else:
        if not args.report or not args.report.strip():
            parser.error('--ncu requires --report')
        command = ['ncu', '--nvtx', '--kernel-name-base', 'demangled', '--set', args.ncu]
        if args.ncu == 'basic':
            if any(x is not None for x in selection):
                parser.error('basic inventories all measure kernels; selection options are not allowed')
            command += ['--nvtx-include', 'measure/']
        else:
            if any(x is None for x in selection) or not args.kernel.strip():
                parser.error('full requires --phase, --kernel, --launch-skip and --launch-count')
            if args.launch_skip < 0 or args.launch_count <= 0:
                parser.error('full requires launch-skip >= 0 and launch-count > 0')
            command += ['--nvtx-include', 'measure/' + args.phase,
                        '--kernel-name', args.kernel, '--launch-skip', str(args.launch_skip),
                        '--launch-count', str(args.launch_count)]
        command += ['-o', args.report, sys.executable, str(Path(__file__).resolve()),
                    '--config', str(Path(args.config).resolve())]
        # 子进程不再传--ncu，避免递归启动；命令数组不经shell解释。
        print(shlex.join(command), flush=True)
        if not args.print_command:
            raise SystemExit(subprocess.run(command, check=False).returncode)
        return
    config = read_diagnostic_config(args.config)
    rows = load_fixed_samples(config['sample_manifest'])
    needed = config['batch_size'] * config['capture_batches']
    if len(rows) < needed:
        raise ValueError('Not enough frozen samples for complete capture batches')
    rows = rows[:needed]
    output = Path(config['out'])
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'starting', 'config': config, 'generation_seed': GENERATION_SEED,
              'prompt': PROMPT_TEMPLATE, 'batches': [], 'measurement': 'profile workload; no throughput claim'}
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
        result['runtime'] = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                             'transformers': importlib.metadata.version('transformers'),
                             'gpu_name': torch.cuda.get_device_properties(0).name,
                             'model_config': model.config.to_dict(), 'dtype': str(model.dtype)}
        # 完整generate预热；不传递生成所得cache。每批CPU输入保持至采集结束。
        for index in range(config['capture_batches']):
            batch = rows[index * config['batch_size']:(index + 1) * config['batch_size']]
            inputs, metadata = prepare_fixed_batch(batch, processor, prompt, config)
            inputs = inputs.to('cuda:0')
            for _ in range(config['warmup_batches']):
                warm = generate(model, processor, inputs, config['max_new_tokens'], torch)
                del warm
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push('measure')
            try:
                with _generation_nvtx_regions(model, torch.cuda.nvtx):
                    outputs = generate(model, processor, inputs, config['max_new_tokens'], torch)
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            metadata['generated_padded_steps'] = outputs.shape[-1] - metadata['padded_input_tokens']
            metadata['output_token_ids'] = outputs[:, metadata['padded_input_tokens']:].cpu().tolist()
            result['batches'].append(metadata)
            del inputs, outputs
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        (output / 'workload.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
