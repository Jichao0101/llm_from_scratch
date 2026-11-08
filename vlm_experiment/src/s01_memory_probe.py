import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import Qwen3_5ForConditionalGeneration

MODEL = "/workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-9B"
OUT = Path("/workspace/dms-eye-status-turbo/vlm_eye_experiment/runs/s01")
OUT = OUT / datetime.now().strftime("load_%Y%m%d_%H%M%S_%f")
OUT.mkdir(parents=True, exist_ok=False)

GiB = 1024 ** 3
# 提前初始化cuda context
torch.cuda.set_device(0)
torch.cuda.init()

def memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    return {
        "allocated_GiB": torch.cuda.memory_allocated(0) / GiB,
        "reserved_GiB": torch.cuda.memory_reserved(0) / GiB,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated(0) / GiB,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved(0) / GiB,
        "device_used_GiB": (total - free) / GiB,
        "device_free_GiB": free / GiB,
    }

torch.cuda.reset_peak_memory_stats(0)
before = memory()

# 记录分配历史，限制事件数量以控制快照大小
torch.cuda.memory._record_memory_history(max_entries=100000)

try:
    start = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    after = memory()

    # 汇总实际参数的dtype和设备，不复制tensor
    layout = defaultdict(lambda: {"numel": 0, "bytes": 0})
    for param in model.parameters():
        key = f"{param.device}/{param.dtype}"
        layout[key]["numel"] += param.numel()
        layout[key]["bytes"] += param.numel() * param.element_size()

    report = {
        "stage": "model_loaded_no_forward",
        "model": MODEL,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds_with_memory_history": load_seconds,
        "before": before,
        "after": after,
        "parameter_layout": dict(layout),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (OUT / "load_metrics.json").write_text(text + "\n", encoding="utf-8")
    print(text)
finally:
    try:
        torch.cuda.memory._dump_snapshot(str(OUT / "load_memory.pickle"))
        print("\n结果目录：", OUT)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)