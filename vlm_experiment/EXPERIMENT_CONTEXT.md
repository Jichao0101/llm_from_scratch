# VLM 实验上下文快照

同步日期：2026-10-08。用途：上下文压缩后的任务恢复，以及云端无法访问本地知识库时的有限上下文。不是云端实时状态报告，也不是完整实验记录副本。

## 来源与当前步骤

知识库完整路径见 [AGENTS.md](AGENTS.md)。设计依据为方案1.3、1.5、1.5.3；执行依据为记录1.2、1.3、1.4.5；实现核对为本仓库三个 `src/s02_*.py` 脚本及README。下列用户后续约定来自本次持续会话，并已体现在构建/评估代码中。

| 执行编号 | 对应设计 | 同步时状态 |
|---|---|---|
| S02-A | E1开发集抽样诊断 | 已有40图报告；属于诊断，不能充当正式baseline |
| S02-B | 方案1.5.3推理效率实验 | 数据集已由用户人工确认构建完成，输入前置通过；batch/profiler/DP实验待执行 |
| S02-C | E1正式baseline | 等待效率实验确定执行配置，再做验证集全量评估 |
| S03 | E2 targeted SFT/LoRA | 候选；正式baseline后决定质量训练 |

当前下一步是在 `src/s02_sample_eval.py` 实现样本batch实验。同步时该文件为空；`s02_evaluate_baseline.py`仍为单卡逐图实现，不能把计划中的batch/DP能力当成已经实现。正式数据集完成不等于baseline推理完成。

## 已确定的数据与输出合同

- 正式批次：67/68/70/71/75训练、77验证、79测试。本地样例只用于验证代码；目录名称不保证为FA格式，以构建脚本中的实际映射为准。
- 完整会话目录是不可拆分的场景group。同一人员不同会话允许跨集合；不同会话按采集标准视为不同场景。相同图片内容跨集合在构建时阻止，不在评估时重复hash扫描。
- 标注ID固定映射：`id=2`为驾驶员右眼，即`image_left_eye`；`id=1`为驾驶员左眼，即`image_right_eye`。GT不按框中心重新排序。
- `eye_open/eye_closed/eye_narrow/eye_occluded`映射为`open/closed/narrow/occluded`。hard、缺失及非法标注/框整图进入exclude；hard不映射为occluded。权限、图片读取、目录/划分错误仍需显式处理，不能都当成坏标签跳过。
- manifest schema v4：图片与标注路径为绝对路径，`gt_pixel`保留像素GT，`target_norm1000`保存整数xyxy监督目标。清单必须在实际运行环境构建；迁移数据路径后重建。
- 模型输出Norm1000整数xyxy；预测还原一次为浮点像素坐标评分。prompt不携带图像尺寸、待测文件名或GT，不补框、不交换预测左右。
- 正式评估使用完整验证清单，失败保留分母；测试集只用于最终候选验收。40图与split关系和四类覆盖仍需在正式比较中说明。

已知文档差异：知识库方案1.3.1及记录1.3.1仍存在按框中心排序的旧表述，部分段落仍要求缺失/未知标注报错，历史实现段落仍提schema v3。上述内容落后于用户后续明确约定及当前代码；恢复任务时按本节所列最新约定继续，不将代码回退。本快照仅指出差异，未将知识库原文视为已修正。

## 当前代码默认配置（不是全数据验证结论）

来源：`src/s02_evaluate_baseline.py`的`main()`与`run()`。运行时以实际CLI覆盖值及输出`config.json`为准。

| 项目 | 当前默认值/行为 |
|---|---|
| 模型 | Qwen3.5-9B，`Qwen3_5ForConditionalGeneration` |
| 云端模型目录 | `/workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-9B`；实际可用性运行前确认 |
| 加载 | 本地离线，BF16，SDPA，单卡完整模型，eval/inference mode |
| processor | `AutoProcessor`读取模型目录保存配置；不猜测min/max pixels，不自动改变图像预算 |
| 解码 | 非思考，greedy，`do_sample=False`，`num_beams=1`，`use_cache=True` |
| 总context预算 | 5600，生成上限256；输入加生成预算超限记失败，不截断 |
| seed / GPU / split | 20261003 / GPU 0 / val；GPU编号是默认选择，不证明当前空闲 |
| 精度脚本 | 无hash校验、耗时统计；支持resume；没有batch/DP实现 |

构建配置在 `src/s02_build_baseline_dataset.py`的`BUILD_SETTINGS`；评估不接收`--data-root`。数据实际输出目录由`--dataset`指定，不能把本地`extract_data/`默认当成云端清单位置。

## 云端环境与性能的历史报告

来源：实验记录1.3.2、1.4.2～1.4.4。以下为历史配置/用户报告，未在此次文档更新中复测：

- 4张RTX PRO 5000 72GB Blackwell，每卡73415MiB（约71.7GiB），驱动580.126.20。GPU 1～3曾被占用，当前占用未知。
- Python3.11，PyTorch2.10.0+cu128，CUDA runtime12.8；Transformers5.2.0、Accelerate1.13.0、PEFT0.18.1、Pillow11.3.0。
- causal-conv1d1.6.1、flash-linear-attention0.4.2；安装不代表对应kernel已执行。vLLM0.17.1已有导入/CLI报告；当前正确性评估使用Transformers。
- 加载allocated约17.53GiB，旧单图峰值约18.55GiB；旧输入4732 tokens、输出95 tokens，首次generate约43.30秒，后续3次均值约2.77秒。
- 这些是旧样本、旧配置、generate范围的测量；不能当作当前端到端吞吐、最大batch或训练容量。每卡总容量约72GB不等于单图占用70GB。

## 已选定的下一步实验方法

方案1.5.3的恢复摘要，具体方法仍以可访问的方案原文为准：

1. 从已构建验证manifest固定约128张性能样本，覆盖尺寸/场景/类别和较大输入；测试集不参与选参。
2. 单卡扫描B=1/2/4/8，有收益和余量再测16。各请求仍一图一指令，左padding、attention mask、按padding后输入宽度截取输出；保留样本对应和尾批。
3. 固定Norm1000、BF16、processor预算和生成规则；对比B=1精度与失败分母。OOM记录配置不可行，不静默拆小批计入原成绩；不每批empty_cache。
4. 普通预热后测量与短窗口CPU/CUDA profiler分开，保存吞吐、allocated/reserved峰值与trace；不对全量数据启用profiler。
5. 单卡结果后决定是否测试四卡：每卡独立进程和完整模型，分片相同总样本，核对无重复/遗漏与精度。不使用训练DDP同步，不预设4倍加速；无收益或资源不可用可选择单卡。

完成这一实验后，才将实测选定配置用于正式baseline。同步实际结果时更新记录及本快照；不要把计划改写为已验证能力。
