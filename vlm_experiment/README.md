# VLM experiment code

本地代码保存目录：/home/jichao/test/vlm_experiment。
训练和真实模型验证在云端执行。本地保存脚本不代表已在 GPU 验证。

## 当前步骤：S03 LoRA 模块清单

将 src/s03_lora_module_probe.py 传到云端实验目录的 src/ 下。
在云端 vlm_eye_experiment 根目录运行：

```bash
python src/s03_lora_module_probe.py --gpu 0 --out runs/s03/lora_modules_001.json
```

默认模型为 /workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-9B。
脚本使用离线 BF16 加载，只打印实际 Linear 模块并保存完整清单；不注入 LoRA，不训练。
输出路径已存在时拒绝覆盖。应先释放 GPU 0 上之前的模型进程，避免重复加载。
根据清单确认语言主干完整路径后，再固定显式 LoRA 白名单。

首轮候选：r=16、alpha=32、dropout=0；底座、视觉及连接模块冻结。
下一步是 assistant JSON + 结束标记监督检查，然后一次完整更新及保存重载。
