# 固定样本batch与DP性能实验

入口为`src/s02_sample_eval.py`，配置为`sample_eval.json`；云端需同时传输`src/experiment_profiler.py`。所有可变实验参数显式读取JSON，不合并默认配置。相对路径基于JSON所在目录，`dataset`指包含dataset.json与val.jsonl的构建结果；不接收原始data-root，不重新校验源文件hash。

```bash
python src/s02_sample_eval.py --config sample_eval.json
```

本环节只比较推理效率，不读取GT评分、不要求场景覆盖，不替代正式baseline。每次使用新的`out`目录，保留配置、样本、执行结果和失败信息；已有run不覆盖。

## 四种模式与固定测量位置

| mode | 用途 | 固定测量范围 |
|---|---|---|
| benchmark | 比较完整固定样本的稳定性能，多轮重复 | 整轮、每批、读图/processor/H2D/generate/输出解码的基础耗时、吞吐和整轮显存峰值 |
| timing | 解释generate内部耗时，短窗口 | 视觉编码、语言prefill、语言decode及输出投影；CPU计时和CUDA事件 |
| memory | 解释显存增长，短窗口 | 先测五环节allocator峰值，再独立测生成内部模块峰值，避免嵌套重置 |
| operator | 发现瓶颈后进行算子归因，短窗口 | 五环节及生成内部的region标记、CPU/CUDA算子统计和trace |

先benchmark，再按需要分别运行timing、memory，仍需算子归因时才运行operator。三种诊断不会互相自动开启，timing/memory不启动torch.profiler。pipeline/generation是结果分组名称，不再是配置选项，测量位置固定在代码。

## 配置示例与条件参数

单卡benchmark示例：

```json
{
  "dataset": "extract_data",
  "out": "runs/sample_eval_benchmark_v1",
  "model": "/workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-9B",
  "mode": "benchmark",
  "sample_manifest": null,
  "sample_count": 128,
  "sample_seed": 20261008,
  "batch_sizes": [1, 2, 4, 8, 16, 32],
  "gpus": ["0"],
  "warmup_batches": 2,
  "repeats": 3,
  "context_budget": 5600,
  "max_new_tokens": 256,
  "cpu_threads_per_worker": 4
}
```

| 参数 | 生效条件 |
|---|---|
| dataset/model/out、mode、gpus、batch_sizes、warmup_batches、context_budget、max_new_tokens、cpu_threads_per_worker | 所有模式必填 |
| sample_manifest | 所有模式必填；null表示按种子随机采样，路径表示复用冻结清单 |
| sample_count、sample_seed | 仅随机抽样时必填；复用清单时移除并且不要求填写 |
| repeats | 仅benchmark必填，控制完整样本重复次数 |
| capture_batches | timing/memory/operator必填，控制每个batch档位的短窗口批数 |
| dp_batch_size | 仅多GPU benchmark必填，表示DP每卡batch，必须属于batch_sizes |

`batch_sizes`为递增、唯一的正整数，可以只测[16,32]；无B1时不计算speedup_vs_b1。每档重新加载模型，冷调用与预热不混入稳定窗口。超输入预算直接终止该配置，不截断、降分辨率或过滤样本。

复用样本时将`sample_manifest`设为既有run的samples.jsonl，可以完全省略sample_count/sample_seed；脚本只核对sample ID和图片绝对路径与当前验证manifest一致。随机抽样为均匀无放回、最多sample_count张，不按场景/状态分层。输出sample_info仅在实际随机抽样时记录抽样seed。

诊断运行将mode改为timing、memory或operator，增加`capture_batches`并设置新out，例如捕获3批；operator可先缩到1批并只测瓶颈batch。诊断模式不要求repeats，也不使用DP配置。

代码固定生成seed=42；operator固定record_shapes=true、profile_memory=true、with_stack=false。前者保存在输出config.json的generation_seed，后者仅operator运行保存在operator_settings；它们是已明确的实现常量，不是默认配置合并。

旧版的gpu、dp、seed、profiler字段会被拒绝，请迁移为gpus、dp_batch_size、四种mode及capture_batches；不保留旧profiler.measurement/detail开关。

## GPU与DP

`gpus`使用GPU ID或UUID字符串。单元素表示单卡；多个元素仅允许benchmark，先在第一张卡完成batch扫描，再在全部指定卡上以dp_batch_size比较独立副本分片推理。例如：

```json
"gpus": ["0", "1", "2", "3"],
"dp_batch_size": 16
```

timing、memory、operator明确要求单元素gpus，不静默忽略其他卡。每个worker只可见所选的一张卡，进程内设备为cuda:0；配置不证明该卡当前空闲，运行前检查云端资源。

DP每卡独立进程、一份完整模型，样本分片不重复；不使用训练DDP同步。各轮预热后对齐，按最早开始至最晚完成的全局窗口计算吞吐，不相加局部吞吐。与相同每卡batch的单卡结果计算加速比和效率；样本不足以填满实际batch时不声称容量已验证。任一worker失败中止同组其他worker，保存原始故障和部分输出，不静默回退单卡或小batch。

## 输出与计时口径

所有模式保存config.json、samples.jsonl、sample_info.json、summary.json。结果记录实际输入/输出token、失败与生成上限命中情况；不因模型内容错误停止效率扫描。

benchmark保留每轮global_pipeline_us、逐批pipeline_us及stage_us，另汇总mean_global_pipeline_us、stdev_global_pipeline_us、mean_batch_pipeline_us、mean_stage_us_per_batch和吞吐波动。基础计时始终启用，不需要timing模式；单位微秒，用perf_counter与必要CUDA同步。模型加载、预热、barrier等待、写盘和日志在稳定窗口之外。批均值包含尾批，DP批/环节均值跨rank池化，不代表全局批延迟；单批耗时除以图片数只能称摊销耗时。

诊断取同一清单的前`batch_size × capture_batches`张，保留尾批；至少需要max(batch_sizes)张。完整批用于stage_summary，实际生成数和总窗口峰值仍包括尾批。

- timing：generation_timing、generation_profile.json保存内部逐调用事件；stage_summary汇总完整批中各模块调用的平均CPU/CUDA耗时及次数。CPU提交时间与当前stream CUDA事件区间不同，事件区间可能包含调度空隙，不等于纯kernel计算时间。
- memory：stage_memory保存五环节allocated/reserved入口、结束、峰值及增量，stage_summary仅汇总完整批峰值；generation_memory及generation_profile.json来自第二次独立模块测量，保存两次输出差异数和token数。同步后的诊断时长不用于稳定吞吐。
- operator：profile_b*/rank0/trace.json与operators.json保存区间标记和CPU/CUDA算子调用数、total/self耗时与内存净变化；stage_summary为空，不额外运行Timing或Memory窗口。包含关系下的region/aten/kernel时间不可相加。

加载后allocator快照与通用执行墙钟字段仍会保留，不能据此认为开启了阶段显存或CUDA事件采集。阶段峰值/headroom仅在memory诊断输出；峰值增量不全部归为activation，reserved不等于活跃tensor，各阶段峰值不可相加。headroom仅是总容量减PyTorch峰值，不覆盖其他进程和非PyTorch分配。

Qwen生成hook依据语言forward输入长度和进入前的缓存长度识别prefill/decode；接口不匹配时显式失败。视觉、语言主干和输出投影区间不覆盖generate的全部控制、embedding和采样开销。

OOM记录该配置不可行并停止更大档位；坏图、预算超限和其他运行故障亦终止该轮，不缩小batch伪装成功。每轮比较冻结模型、prompt、processor预算和生成条件；输出长度变化与工具开销均需单独解释。随机小样本不证明最大输入容量或正式精度。

## 后续LoRA复用

通用模块仍提供三个独立工具：TimingProfiler.measure/resolve/report、MemoryProfiler.measure/report、OperatorProfiler.capture/region/step/export。生成hook及阶段语义由推理脚本负责，模块不加载模型、读取数据或执行训练/参数搜索。

训练循环可用measure标记forward、backward、optimizer step和zero_grad；Timing事件须显式resolve后report，Timing不读取allocator；Memory同步测峰值，禁止同设备嵌套，包括不同实例。Operator以capture管理短窗口，region可嵌套，区间结束后才step，窗口退出后export。metadata仅允许标量，不保留tensor延长生命周期。

真实训练需明确梯度累积、有效监督tokens、首次及稳定完整更新，计时、峰值及算子窗口分别运行；最优微调配置仍需独立质量对照。本仓库尚未因此实现LoRA训练循环或参数搜索。
