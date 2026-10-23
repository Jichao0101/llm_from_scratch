# Qwen3.5-9B 实验脚本

本目录用于保存代码，本地做结构验证，真实推理在云端执行。一起上传整个`src/`；JSON配置与对应脚本放在同一目录。

## 文件与启动约定

| 文件 | 用途 | 配置 |
|---|---|---|
| [s02_build_baseline_dataset.py](src/s02_build_baseline_dataset.py) | 构建数据清单 | 脚本内BUILD_SETTINGS与CLI |
| [s02_sample_eval.py](src/s02_sample_eval.py) | 样本吞吐、batch/DP与阶段/算子测量 | [同名JSON](src/s02_sample_eval.json) |
| [s02_inspect_execution.py](src/s02_inspect_execution.py) | 核验实际模型实现与attention路径 | [同名JSON](src/s02_inspect_execution.json) |
| [s02_profile_workload.py](src/s02_profile_workload.py) | 为Nsight提供固定生成负载 | [同名JSON](src/s02_profile_workload.json) |
| [s02_evaluate_baseline.py](src/s02_evaluate_baseline.py) | 完整验证集质量评估 | CLI |
| [s02_workload.py](src/s02_workload.py) | 共享输入、模型加载与生成语义 | 供前三个推理实验入口调用 |
| [experiment_profiler.py](src/experiment_profiler.py) | 通用计时、显存和算子采集 | 库模块 |

三个JSON入口默认读取脚本自身目录中的同名`.json`，不依赖启动时的工作目录。也可用`--config /path/to/config.json`覆盖；JSON内部的相对路径始终基于该JSON目录。默认的是配置文件位置，实验参数仍要求显式填写。

示例配置中的`../datasets/baseline_v1`和`../runs/...`相对于src目录。填写云端实际模型、数据、样本清单和GPU；每次使用新的out。更改benchmark的out后，同步修改两个独立诊断配置的sample_manifest。以下命令从项目根目录执行。

## s02_build_baseline_dataset.py：构建数据

读取整图及同名标注，按完整会话目录划分train/val/test，生成dataset.json、三个manifest和排除/错误报告。先核对BUILD_SETTINGS中的批次映射和真实目录名。

```bash
python src/s02_build_baseline_dataset.py \
  --data-root /workspace/dms-eye-status-turbo/extract_data \
  --out datasets/baseline_v1
```

`--observed-list`可传入已查看诊断图片的相对路径清单，防止其进入最终测试集。manifest保存图片绝对路径；迁移图片目录后在实际环境重建。构建需要Python 3.10+和Pillow。

## s02_sample_eval.py：效率测量

默认JSON为benchmark：随机选取128张验证图片并保存samples.jsonl，扫描B1/2/4/8/16/32。不读取GT评分，不替代正式baseline。

```bash
python src/s02_sample_eval.py
```

通过同名JSON的mode选择功能，每次修改out：

| mode | 测什么 | 条件字段 |
|---|---|---|
| benchmark | 完整清单的吞吐、五阶段基础耗时及整轮显存峰值 | repeats |
| timing | 读图、processor、H2D、generate、输出解码的CPU/CUDA事件区间 | capture_batches |
| memory | 五阶段allocated/reserved入口、结束、峰值与增量 | capture_batches |
| operator | 流程和vision/prefill/decode区间、算子统计与trace | capture_batches |

切换到timing/memory/operator时移除repeats，增加正整数capture_batches，例如3；sample_manifest设为benchmark保存的清单路径，移除sample_count/sample_seed，以复用同一批样本。batch_sizes控制扫描档位，warmup_batches控制预热；只有随机抽样才需要sample_count/sample_seed。

gpus为GPU ID/UUID字符串列表。多卡仅benchmark支持，增加dp_batch_size并要求该档位属于batch_sizes；每卡独立进程和完整模型，样本无重复分片，以全部卡完成的全局窗口比较吞吐。

输出config.json、samples.jsonl、sample_info.json、summary.json及逐档结果；operator另存profile_b*/rank0/trace.json与operators.json。计时单位为微秒，批耗时除以图片数是摊销耗时。profiler数据用于归因，普通benchmark用于比较收益；OOM、超预算或运行错误停止，不静默缩小batch。

## s02_inspect_execution.py：执行路径核验

记录云端实际版本、模型/模块配置与源码，观察vision序列边界、attention函数、Q/K/V形状和调用次数；text区分普通attention及线性attention绑定。默认B16、1个完整采集批次，先按需要改为B1并使用新out，再测B16。

```bash
python src/s02_inspect_execution.py
# 需要CPU到CUDA事件关联时，使用另一个新out采集短trace：
python src/s02_inspect_execution.py --trace
```

输出execution.json、implementation.py.txt、attention_calls.jsonl；启用trace时另存trace_batchN.json。按模块invocation核对序列分段和函数调用，再沿CPU算子/CUDA API关联到kernel。函数调用不保证与kernel一一对应，配置SDPA也不能单独证明底层kernel路径。

当前包装参照Transformers5.2.0，接口不匹配或目标模块内无调用时显式失败。替代后端的架构、dtype、head dimension、mask和依赖适用条件需按实际源码核验。诊断有包装、日志与同步开销，不报告吞吐。

## s02_profile_workload.py：Nsight负载

读取同一冻结清单，在每个待采集batch完整generate预热后，用NVTX `measure`标记生成及完成同步；内部通过hook标记`vision`、`language_prefill`、`language_decode`及独立的`prefill_logits`/`decode_logits`。语言阶段按进入forward前的cache长度和输入长度判定，不按调用序号猜测；阶段之间不新增同步。输入准备、预热和输出写盘位于范围外。默认配置与路径核验保持同一负载，使用独立out。

```bash
python src/s02_profile_workload.py
```

已有torch.profiler trace用于定位热点；本轮以分阶段筛选后的代表性kernel做full采集，分析限制原因。先查云端`ncu --version`、`--list-sets`、`--list-sections`及计数器权限。每次运行使用新的JSON out和报告名。

下面以vision为例；TARGET_KERNEL替换为trace核实的名称表达式。分析prefill时将范围改为`measure/language_prefill`，报告名相应改变：

```bash
ncu --nvtx --nvtx-include 'measure/vision' \
  --kernel-name-base demangled --kernel-name 'regex:TARGET_KERNEL' \
  --launch-count 1 --set full \
  -o ncu_vision_b16 python src/s02_profile_workload.py
```

- 先比较B1/B16：vision选同层同shape的attention；prefill选同层同投影的GEMM并核对矩阵规模。同名kernel不保证来自同层，必要时根据当前调用顺序设置`--launch-skip`，核对报告中的launch参数；不要套用历史序号。
- `--launch-count`限制匹配的kernel实例，不是batch数。首次可将full改为basic，只采一个实例检查过滤与报告；不对整个generate重复做全量basic定位，也不对整个阶段的全部调用做full。
- NVTX标记CPU提交范围。默认kernel replay下，阶段筛选仍生成逐kernel指标，不是阶段整体Roofline；不根据CPU标记跨度计算GPU阶段耗时。
- full内容依云端版本和硬件而定。检查匹配精度/计算路径的Roofline、计算管线、各存储层级吞吐，再结合scheduler、eligible warps、依赖等待及spill判断；缺项按实际支持的section/metric补采。不能仅凭低occupancy或低DRAM带宽确定原因。

输出workload.json，保留实际环境、样本、输入形状、图像网格与生成步数；ncu设置另随报告记录，特别是筛选、replay、缓存和时钟控制。采集会扰动执行，其耗时不作为普通吞吐；优化收益另用benchmark验证。逐图attention的调用分支仍由执行路径核验脚本确认，硬件指标不能证明后端不支持batch。

## s02_evaluate_baseline.py：正式质量评估

逐图处理完整验证清单，报告解析、定位、对应侧四类状态和端到端指标；失败保留分母。先做可选CPU检查，再运行真实推理：

```bash
python src/s02_evaluate_baseline.py --dataset datasets/baseline_v1 \
  --out runs/baseline_preflight_v1 --preflight-only
python src/s02_evaluate_baseline.py --dataset datasets/baseline_v1 \
  --model /workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-9B \
  --gpu 0 --out runs/baseline_val_v1 --context-budget 5600 --max-new-tokens 256
```

`--resume`用相同参数恢复，成功与失败case均保留；`--save-overlays`保存GT/预测叠框，恢复时保持该选项一致。最终候选才用`--split test --final-test`。输出config.json、cases/、summary.json、by_batch.json；四类覆盖不全时macro-F1为null，evaluation_complete只代表清单处理完整。

## 共享模块与固定合同

s02_workload.py封装固定prompt、清单校验、离线模型加载、图片/processor处理和greedy生成，不拥有测量策略。experiment_profiler.py提供TimingProfiler、MemoryProfiler和OperatorProfiler，可供后续训练循环复用，当前未实现LoRA训练。

- 正式批次：67/68/70/71/75训练、77验证、79测试；完整会话不能拆分，同人不同会话允许跨集合。hard或缺失/非法标注整图排除，构建时检查跨集合相同图片。
- GT按标注ID固定映射：id=2为image_left_eye，id=1为image_right_eye，不按框中心重新排序。schema v4保留像素GT与整数Norm1000 xyxy目标；预测还原一次为浮点像素评分，不补框、不交换左右。
- BF16、离线加载、左padding、非思考greedy、num_beams=1、use_cache=True。示例总预算5600、输出上限256，超限不截断；图像预算沿用processor。三项推理实验seed=42，正式评估CLI默认seed=20261003。
- 每次generate不传递上次输出cache。样本效率固定请求SDPA；两个独立诊断的attn_implementation显式配置。独立诊断只接受完整采集批次，样本数须达到batch_size×capture_batches。

## 当前状态与依据

同步日期2026-10-21。已有batch/operator/trace分析完成；B16仍是DP候选。下一步核验实际后端路径并采集代表性vision attention、prefill GEMM硬件指标，DP可独立推进。正式全量baseline及新增诊断入口仍待云端运行，本地结构/模拟验证不代表GPU性能或模型质量通过。

知识库实验方案定义方法，实验记录及附录保存历史结果；本说明不复制历史分析。用户后续明确约定优先于旧文档中的GT排序、非法标注处理或schema表述。原文位置见AGENTS.md，历史结果不要求适配当前接口。

参考：[Qwen3.5实现](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。
