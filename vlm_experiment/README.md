# Qwen3.5 眼睛定位与状态 baseline

本地代码目录：`/home/jichao/test/vlm_experiment`。将本目录的 `src/` 传到云端实验目录即可；图片保留原处，manifest 保存图片与标注的绝对路径，必须在实际运行评估的环境中构建；数据搬迁后重新构建清单。

## 代码边界与manifest合同

- `src/build_baseline_dataset.py`独立负责扫描、过滤、分组、像素GT读取与norm1000目标生成。
- `src/evaluate_baseline.py`独立负责读取冻结manifest、构造prompt、推理、解析和评分；不导入构建脚本，也不读取原始标注的dataList结构。
- 两者无相互导入或共享业务模块，仅通过版本化manifest对接。JSON写入逻辑留在各自脚本中。

manifest schema v4中，`gt_pixel`保存原图浮点像素GT，`target_norm1000`保存按实际W/H转换的整数xyxy目标：`round(x/W*1000)`、`round(y/H*1000)`，采用ties-to-even。原始JSON不改写。
模型prompt要求norm1000输出；解析先检查0..1000，再还原浮点像素坐标与`gt_pixel`评分，避免先量化GT再还原造成额外评分误差。
`image`和`annotation`为绝对路径，`dataset.json.path_format`为`absolute`。评估仅接收`--dataset`，不再接收`--data-root`；旧版相对路径清单需要重新构建。
`dataset.json.coordinate_contract`显式标记这两种坐标；评测拒绝不匹配的版本/坐标合同。

## 固定数据合同

- 批次：67/68/70/71/75 → train，77 → val，79 → test。
- 完整会话目录是场景group，不能拆到不同集合。同一人员ID的不同会话允许跨集合；人员重叠只报告，不报错。
- group 使用批次目录下的完整相对会话路径，避免同一目录重复交付到不同批次后漏检。不按 `NorDrv` 等类别名合并独立会话。
- 任一眼为 `hard`，整张图排除，并写入 `exclusions.json`。hard表示可能无法定位眼睛，不映射为occluded。
- 四类映射：eye_open/eye_closed/eye_narrow/eye_occluded → open/closed/narrow/occluded。缺失JSON/眼标签、非法JSON结构、未知标签、非双眼、非法ID、非法/越界/量化后退化框均整图排除，记录图片、标注路径及具体原因；不补框或修正GT。
- 按实际图片尺寸读取像素GT，标注`id=2`（驾驶员右眼）映射为`image_left_eye`，`id=1`（驾驶员左眼）映射为`image_right_eye`，不按框中心排序。模型输出norm1000整数xyxy，验证0..1000后转换一次到像素评分；不裁剪、不旋转、不修正预测左右。
- 跨集合完全相同的图片内容仍报错。文件hash不等于近重复检测；原始视频有不同编码或不同拷贝路径时，需保证采集组身份约定成立。

## 1. 构建云端正式数据集

在云端实验根目录、已有Python训练环境中执行：

```bash
python src/build_baseline_dataset.py \
  --data-root /workspace/dms-eye-status-turbo/extract_data \
  --out datasets/baseline_v1
```

运行前直接修改`src/build_baseline_dataset.py`顶部`BUILD_SETTINGS`中的`batch_dirs`：键是批次号字符串，值是该批次相对`--data-root`的实际目录路径，例如`"67": "delivery_67"`。目录名无需包含FA、批次号或日期；`splits`仍决定各批次属于train/val/test。本地划分沿用你在代码中的设置；云端正式实验前按上述正式批次划分填写实际目录名。

要求代码中选择的批次目录均存在且不相互重叠，完整递归收集，不随机采样。图片无需复制。批次目录以下层级沿用原始结构，图片及同名JSON直接位于会话目录；人员统计仍使用会话名的数字前缀。
生成 `train.jsonl`、`val.jsonl`、`test.jsonl`、`dataset.json`、`exclusions.json`、`build_errors.json`。
`dataset.json` 包含各split图片/眼睛四类/人员/会话/场景数量、人员交叠及manifest hash。
`exclusions.json`记录不可用样本，`dataset.json.excluded_by_reason`按原因汇总；这些排除不导致构建失败。权限/图片读取错误、孤立JSON缺图、目录/划分错误、跨集合重复和排除后空集合仍记为构建错误，不发布可用manifest。输出目录存在会拒绝覆盖。

既有40图已用于诊断，正式比较前需要明确其与各集合的关系：可提供 `--observed-list observed_images.txt`，文件每行是相对数据根目录的图片路径。传入的列表应覆盖已知诊断样例；测试集出现这类样本会报错。若在其他流程完成核对，可在脚本 `BUILD_SETTINGS` 的 `diagnostic_overlap_review` 填写实际依据。未核对时仍能构建和推理，但报告保持 `diagnostic_overlap_reviewed=false`，不自动声明正式baseline就绪。

## 2. 可选：CPU数据读取检查

```bash
python src/evaluate_baseline.py \
  --dataset datasets/baseline_v1 \
  --out runs/baseline_preflight_v1 --preflight-only
```

读取全部验证图片并检查路径、尺寸和解码。只检查数据，不加载模型、不产生模型质量指标。

## 3. 运行完整验证集 baseline

```bash
python src/evaluate_baseline.py \
  --dataset datasets/baseline_v1 \
  --model /workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-9B \
  --gpu 0 --out runs/baseline_val_v1 \
  --context-budget 5600 --max-new-tokens 256
```

默认评估val，读取全部manifest，无limit参数。采用离线BF16、SDPA、单卡、非思考模板及确定性greedy生成，processor沿用模型目录配置。超预算样本记录失败，不自动截断或改变图像预算；需要改预算时建立新的run。
评估脚本只负责基模精度：不计算或校验任何hash，不记录耗时和运行时间戳，也不执行为计时添加的CUDA同步。已完成case恢复时不重新读取源文件。首次运行请确保GPU 0有足够显存且没有重复加载模型。

中断后用完全相同命令加 `--resume`。模型/数据目录与推理参数必须一致；已保存的成功和失败case均保留，不挑选性覆盖失败。图片、标注和模型权重内容视为固定，不再通过hash检测其变化。本次运行记录格式已调整，旧版本run不能直接续跑，请使用新输出目录。
需要叠框图时首次运行加 `--save-overlays`，生成全部case的GT/预测对照图；恢复时保持此选项相同。

输出：

- `config.json`：模型/数据目录、prompt及推理参数，用于解释结果与断点续跑。
- `cases/<sample_id>.json`：原始输出、像素预测框、token数及失败原因。
- `summary.json`：固定清单分母、成功率、定位及对应侧四类指标。
- `by_batch.json`：按批次汇总。

定位分别报告一对一匹配召回和不交换左右的同侧召回；端到端要求原始左右、定位及状态同时正确。状态混淆矩阵直接按对应侧统计，不依赖IoU匹配，失败进入invalid列。四类覆盖不全时四类macro-F1为null，不能把仅open的高分当作四类能力。
`evaluation_complete`只表示清单处理完整，不代表质量通过；`baseline_ready_for_comparison`还要求val四类覆盖与既有诊断样例关系已核对。中断时summary记录incomplete。若启动/恢复失败，以命令错误为准，不把旧报告当作本次成功。

最终候选才使用 `--split test --final-test`。当前主指标代表跨采集场景表现，不能直接声明未见人员泛化。

## 验证边界

2026-10-03本地数据验证：4189图，hard排除66图，保留4123图；契约测试和保留图片的实际读取/内容身份验证通过。按用户要求，临时清单、结果、测试配置、测试脚本与LoRA探测脚本已清理，仅保留正式baseline代码。本地验证不代表云端模型推理通过。

构建仅需Python 3.10+和Pillow；真实推理使用云端已有PyTorch/Transformers/Accelerate环境。
模型接口参考：[Transformers Qwen3.5官方文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)。真实模型端到端运行尚待云端验证。
