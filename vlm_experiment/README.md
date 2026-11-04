# Qwen3.5-9B 实验脚本

本目录用于保存代码，本地做结构验证，真实推理在云端执行。一起上传整个`src/`；JSON配置与对应脚本放在同一目录。

## 文件与启动约定

| 文件 | 用途 | 配置 |
|---|---|---|
| [s02_build_baseline_dataset.py](src/s02_build_baseline_dataset.py) | 构建数据清单 | 脚本内BUILD_SETTINGS与CLI |
| [s02_export_samples.py](src/s02_export_samples.py) | 从完整验证集导出固定清单和原图 | [同名JSON](src/s02_export_samples.json) |
| [s02_sample_eval.py](src/s02_sample_eval.py) | 样本吞吐、batch/DP与阶段/算子测量 | [同名JSON](src/s02_sample_eval.json) |
| [s02_inspect_execution.py](src/s02_inspect_execution.py) | 核验实际模型实现与attention路径 | [同名JSON](src/s02_inspect_execution.json) |
| [s02_profile_workload.py](src/s02_profile_workload.py) | 为Nsight提供固定生成负载 | [同名JSON](src/s02_profile_workload.json) |
| [s02_evaluate_baseline.py](src/s02_evaluate_baseline.py) | 完整验证集质量评估 | CLI |
| [s02_workload.py](src/s02_workload.py) | 共享输入、模型加载与生成语义 | 供前三个推理实验入口调用 |
| [experiment_profiler.py](src/experiment_profiler.py) | 通用计时、显存和算子采集 | 库模块 |

四个JSON入口默认读取脚本自身目录中的同名`.json`，不依赖启动时的工作目录。也可用`--config /path/to/config.json`覆盖；JSON内部的相对路径始终基于该JSON目录。默认的是配置文件位置，实验参数仍要求显式填写。

示例配置路径相对于src目录。导出脚本读取完整dataset；三个性能入口均读取`../samples/perf_32/samples.jsonl`，与benchmark输出目录独立。填写当前机器实际的4B模型目录和GPU，每轮使用新的out。以下命令从项目根目录执行。

## s02_build_baseline_dataset.py：构建数据

读取整图及同名标注，按完整会话目录划分train/val/test，生成dataset.json、三个manifest和排除/错误报告。先核对BUILD_SETTINGS中的批次映射和真实目录名。

```bash
python src/s02_build_baseline_dataset.py \
  --data-root /workspace/dms-eye-status-turbo/extract_data \
  --out datasets/baseline_v1
```

`--observed-list`可传入已查看诊断图片的相对路径清单，防止其进入最终测试集。manifest保存图片绝对路径；迁移图片目录后在实际环境重建。构建需要Python 3.10+和Pillow。

## s02_export_samples.py：导出固定性能样本

在完整dataset可访问的机器上，填写同名JSON中的dataset和out，默认从val均匀随机无放回抽取32张，seed为20261008。按sample_id排序后抽样，保留抽样顺序；样本不足32张时明确报错。修改数量需显式调整sample_count。

```bash
python src/s02_export_samples.py
```

导出目录如下，迁移时整体复制：

```text
samples/perf_32/
├── samples.jsonl
├── sample_info.json
└── images/
    ├── 0001.jpg
    └── ...
```

原图按字节复制，不重新编码；清单保留sample_id、split及其他原字段，只将image改成`images/0001.jpg`这样的相对路径。sample_info.json记录抽样来源、seed和数量，其中source_dataset仅用于追溯，运行时不要求该目录存在。输出目录已存在时拒绝覆盖；导出失败时清理本次临时目录。

三个性能入口统一用sample_manifest指定导出的清单，图片路径始终相对清单目录解析，与启动目录无关。入口检查val归属、唯一ID、唯一图像路径和文件存在；完整验证集成员身份在导出时确认。旧运行结果里的绝对路径清单不再作为新输入。正式评估仍使用完整dataset。

## s02_sample_eval.py：效率测量

默认JSON为benchmark：读取导出的32张固定验证样本，扫描B1/2/4/8/16/32。不读取GT评分，不替代正式baseline。

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

切换到timing/memory/operator时移除repeats，增加正整数capture_batches，例如3；sample_manifest继续指向同一导出清单。batch_sizes控制扫描档位，warmup_batches控制预热；dataset、sample_count和sample_seed只属于导出配置，性能入口不再抽样。

gpus为GPU ID/UUID字符串列表。多卡仅benchmark支持，增加dp_batch_size并要求该档位属于batch_sizes；每卡独立进程和完整模型，样本无重复分片，以全部卡完成的全局窗口比较吞吐。

输出config.json、resolved_samples.json、sample_info.json、summary.json及逐档结果；resolved_samples.json保存本轮解析后的绝对路径，仅作为运行快照；operator另存profile_b*/rank0/trace.json与operators.json。计时单位为微秒，批耗时除以图片数是摊销耗时。profiler数据用于归因，普通benchmark用于比较收益；OOM、超预算或运行错误停止，不静默缩小batch。

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

已有torch.profiler trace用于定位热点；本轮先用整体basic建立当前调用清单，再对分阶段筛选后的代表性kernel做full采集。先查云端`ncu --version`、`--list-sets`、`--list-sections`及计数器权限。每次运行使用新的JSON out和报告名。

以下为9B历史trace目标清单（依据实验附录1.10.4～1.10.5），供方法参考；4B实际名称、shape及实例需重新核验。full首轮只采vision和prefill；名称与启动参数用于核对当前运行，不是跨版本保证。

| 阶段与优先级 | 已识别的kernel名称或特征 | 代表性分组核验条件 |
|---|---|---|
| vision：首轮 | `pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<96,128,64,4,...>>`（模板名称省略后续参数） | B1/B16的Q/K/V均为`[1,18144,16,72]`；grid=`[142,1,16]`，block=`[128,1,1]`；关联`aten::_flash_attention_forward` |
| language_prefill：首轮 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8` | 关联`aten::mm`；权重`[4096,12288]`，B1/B16输入分别`[4781,4096]`/`[76496,4096]`；grid分别`[304,6,1]`/`[4784,6,1]`，block=`[256,1,1]` |
| language_decode：暂缓，必要时补测 | B1：`internal::gemvx::kernel`；B16：`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8`；B32：`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8` | 同类权重`[4096,12288]`；输入分别`[1,4096]`、`[16,4096]`、`[32,4096]`；同层同投影、相应decode步另行匹配 |
| prefill_logits / decode_logits：暂不独立深采 | 尚未指定目标kernel | 保留阶段标记，不为每个标记强行安排采集；出现新的热点证据后再选择 |

**第一轮：整个measure＋basic。** 不设置kernel筛选、skip或count；保持JSON的capture_batches=1、完整生成及既定预热。B1/B16分别运行并保存对应报告；不要通过缩短生成改变负载。

```bash
python src/s02_profile_workload.py --ncu basic --report ncu_basic_b16
```

脚本会打印并启动等价的`ncu --nvtx --nvtx-include 'measure/' --kernel-name-base demangled --set basic ...`命令。可追加`--print-command`只预览。报告覆盖当前measure内各kernel，用于核对名称、顺序、NVTX阶段与grid/block；basic不是无开销trace，其耗时也不作为benchmark。

**确定实例。** 在basic报告中找到表内目标分组，结合既有trace/执行路径核验确认层、投影与shape；basic本身不保证包含模型层名及张量shape。保留报告位置、阶段、名称表达式、grid/block、模块/shape依据与选择数量。不能把报告的全局ID直接当作launch-skip：按第二轮实际NVTX和名称过滤条件核对匹配序列，skip表示目标之前需跳过多少匹配实例。若当前工具计数口径或阶段归属仍有歧义，先用完全相同筛选条件跑basic小范围核验，再把set改为full。B1/B16分别确认，不复制序号。

**第二轮：子阶段＋明确的匹配区间＋full。** 下列环境变量须先填为已核实的非负skip和正整数count；没有隐含的0或1。每次运行改JSON的out并使用新的report。

```bash
python src/s02_profile_workload.py --ncu full --phase vision \
  --kernel 'regex:pytorch_flash::flash_fwd_kernel' \
  --launch-skip "${VISION_SKIP:?填写已核实的匹配skip}" \
  --launch-count "${VISION_COUNT:?填写已核实的匹配count}" \
  --report ncu_vision_b16

python src/s02_profile_workload.py --ncu full --phase language_prefill \
  --kernel 'regex:cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8' \
  --launch-skip "${PREFILL_SKIP:?填写已核实的匹配skip}" \
  --launch-count "${PREFILL_COUNT:?填写已核实的匹配count}" \
  --report ncu_prefill_b16
```

- full缺少任何选择参数即失败；只有确定首个匹配就是目标时才显式填skip=0。count=1适用于一个已确认实例；多个不连续目标应分次采集，不能把count误作任意实例集合。
- 包装入口只负责启动ncu，子进程仍运行原固定负载；JSON接口不变，不自动缩短输出或修改样本。无`--ncu`时仍可被外部ncu/nsys直接启动。精细核验需要自定义ncu参数时，使用预览命令并自行调整，不向脚本增加任意参数转发。
- full后核对报告的实际名称、阶段、启动配置和采集数量是否满足所选范围；无匹配、少采、执行失败或实例对应不明时，不作预期目标的硬件结论。
- NVTX标记CPU提交范围。默认kernel replay下，阶段筛选仍生成逐kernel指标，不是阶段整体Roofline；不根据CPU标记跨度计算GPU阶段耗时。
- full内容依云端版本和硬件而定。检查匹配精度/计算路径的Roofline、计算管线、各存储层级吞吐，再结合scheduler、eligible warps、依赖等待及spill判断；缺项按实际支持的section/metric补采。不能仅凭低occupancy或低DRAM带宽确定原因。

输出workload.json，保留实际环境、样本、输入形状、图像网格与生成步数；ncu设置另随报告记录，特别是筛选、replay、缓存和时钟控制。采集会扰动执行，其耗时不作为普通吞吐；优化收益另用benchmark验证。逐图attention的调用分支仍由执行路径核验脚本确认，硬件指标不能证明后端不支持batch。

## s02_evaluate_baseline.py：正式质量评估

逐图处理完整验证清单，报告解析、定位、对应侧四类状态和端到端指标；失败保留分母。先做可选CPU检查，再运行真实推理：

```bash
python src/s02_evaluate_baseline.py --dataset datasets/baseline_v1 \
  --out runs/baseline_preflight_v1 --preflight-only
python src/s02_evaluate_baseline.py --dataset datasets/baseline_v1 \
  --model /workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-4B \
  --gpu 0 --out runs/baseline_val_v1 --context-budget 5600 --max-new-tokens 256
```

`--resume`用相同参数恢复，成功与失败case均保留；`--save-overlays`保存GT/预测叠框，恢复时保持该选项一致。最终候选才用`--split test --final-test`。输出config.json、cases/、summary.json、by_batch.json；四类覆盖不全时macro-F1为null，evaluation_complete只代表清单处理完整。

## 共享模块与固定合同

s02_workload.py封装固定prompt、可迁移样本清单校验与路径解析、离线模型加载、图片/processor处理和greedy生成，不拥有测量策略。experiment_profiler.py提供TimingProfiler、MemoryProfiler和OperatorProfiler，可供后续训练循环复用，当前未实现LoRA训练。

- 正式批次：67/68/70/71/75训练、77验证、79测试；完整会话不能拆分，同人不同会话允许跨集合。hard或缺失/非法标注整图排除，构建时检查跨集合相同图片。
- GT按标注ID固定映射：id=2为image_left_eye，id=1为image_right_eye，不按框中心重新排序。schema v4保留像素GT与整数Norm1000 xyxy目标；预测还原一次为浮点像素评分，不补框、不交换左右。
- BF16、离线加载、左padding、非思考greedy、num_beams=1、use_cache=True。示例总预算5600、输出上限256，超限不截断；图像预算沿用processor。三项推理实验seed=42，正式评估CLI默认seed=20261003。
- 每次generate不传递上次输出cache。样本效率固定请求SDPA；两个独立诊断的attn_implementation显式配置。独立诊断只接受完整采集批次，样本数须达到batch_size×capture_batches。

## 当前状态与依据

同步日期2026-11-04。当前模型统一为4B，本地4090用于NCU学习与小规模优化对照，云端用于优化复验和训练。固定性能样本由独立导出脚本生成，默认32张。上述kernel表和已有batch/operator/trace结果来自9B历史运行；4B需重新确认热点、shape及匹配位置。历史B16是9B的DP候选。下一步核验实际后端路径并采集代表性vision attention、prefill GEMM硬件指标，DP可独立推进。正式全量baseline及新增诊断入口仍待云端运行，本地结构/模拟验证不代表GPU性能或模型质量通过。

知识库实验方案定义方法，实验记录及附录保存历史结果；本说明不复制历史分析。用户后续明确约定优先于旧文档中的GT排序、非法标注处理或schema表述。原文位置见AGENTS.md，历史结果不要求适配当前接口。

参考：[Qwen3.5实现](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。
