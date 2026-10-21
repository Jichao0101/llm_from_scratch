"""固定S02生成负载，供ncu/nsys启动；NVTX measure仅覆盖预热后的generate。"""
import argparse
import json
import importlib.metadata
import os
from pathlib import Path
import traceback

from s02_workload import (GENERATION_SEED, PROMPT_TEMPLATE, read_diagnostic_config,
                          load_fixed_samples, load_model, make_prompt, prepare_fixed_batch, generate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().with_suffix('.json')),
                        help='JSON config (default: %(default)s)')
    config = read_diagnostic_config(parser.parse_args().config)
    rows = load_fixed_samples(config['dataset'], config['sample_manifest'])
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
