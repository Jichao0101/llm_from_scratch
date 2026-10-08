# 固定样本 batch / DP 实验

入口为 `src/s02_sample_eval.py`，实验配置为 `sample_eval.json`。`--config`必须显式指定，所有参数直接读取JSON；保留必填项与当前模式的关键参数检查，未启用的DP/profiler参数不校验，脚本不内置默认参数，不合并默认配置，也不从CLI覆盖mode。独立读取构建好的验证manifest；不重复检查GT坐标或量化合同，内部维护固定Norm1000 prompt与输出解码；不需要原始data-root，也不校验图片、模型或清单hash。

## 云端运行

只需传输`src/s02_sample_eval.py`和JSON配置文件即可运行本环节，无下一阶段脚本依赖。在配置中填写实际`dataset`、`model`和新的`out`目录；相对路径基于配置文件所在目录，不基于当前工作目录。`dataset`指包含`dataset.json`和`val.jsonl`的构建结果目录。

```bash
python src/s02_sample_eval.py --config sample_eval.json
```

示例JSON填写`mode=benchmark`，单卡GPU 0，固定最多128张验证样本，扫描B=1/2/4/8/16/32，每档首次调用、2批预热后完整样本重复3轮。每档重新加载模型，加载和冷调用单列，稳定吞吐不含加载。本环节只测模型推理效率，不读取GT评分，也不要求生成内容正确；模型精度留到正式baseline评估。

修改`batch_sizes`控制扫描档位，必须递增且从1开始。按`sample_seed`随机无放回抽样，不按状态、场景、会话或尺寸分层；各配置使用同一清单。实际输入token和padding宽度随输出保存。

仅生成样本与运行配置、不加载模型时，在JSON中设置`"mode": "prepare"`及新的`out`目录：

```bash
python src/s02_sample_eval.py --config sample_eval.json
```

每次运行都要求新的`out`目录，不覆盖旧结果。prepare、benchmark和profile使用不同输出目录。示例JSON的`sample_manifest`为null，按固定种子选样；需要严格复用某次清单时，将其设为该次输出的`samples.jsonl`。脚本仅核对sample ID与图片路径和当前val manifest一致，不比较GT或其他元数据，不重新扫描源文件hash。

## DP开关

先查看单卡吞吐与显存余量，再在配置中设置：

```json
"dp": {
  "enabled": true,
  "gpus": ["0", "1", "2", "3"],
  "batch_size": 8
}
```

GPU标识为CUDA可见设备ID或UUID字符串；开启时第一张卡须与`gpu`一致，`dp.batch_size`必须在单卡扫描列表中。batch size为**每卡**大小，例如4卡×8为每轮最多32张；不足整批正常处理。报告保存实际最大批大小；样本数少于配置档位时，不能据此宣称该档位的完整容量已经验证。先按配置完成单卡扫描，再运行多卡，不自动根据旧显存数据选最大batch。若此前扫描有OOM、输入错误，先处理问题或缩小`batch_sizes`，不会继续DP。

DP每卡独立进程、一份完整模型，按sample分片；无训练梯度同步。各轮预热后对齐开始，按最早开始到最晚完成的全局窗口计算吞吐；不将各卡局部吞吐简单相加。合并时检查重复/遗漏sample ID。任一worker故障中止同组其他worker，保留日志和部分输出，不静默切回单卡或小batch。GPU选择是配置，不代表当前空闲；实际运行前查看云端资源。

## 独立profiler

在JSON中设置`"mode": "profile"`，将`out`设为新的profile目录，`profiler.batch_size`改为已可运行的档位；示例JSON填写记录3批，使用`gpu`指定的单卡。`dp`配置不参与profile模式。

```bash
python src/s02_sample_eval.py --config sample_eval.json
```

输出`profile/rank0/trace.json`和`operators.json`，包含读图、processor、传输、generate、输出解码的阶段标记以及CPU/CUDA算子耗时和内存。可用Perfetto等trace查看器分析实际prefill/decode、视觉编码和kernel路径；不根据扩展包安装情况推断kernel已执行。profiler含工具开销，仅用于归因，不与benchmark稳定吞吐直接比较。

## 结果与停止条件

- 根目录：`config.json`、`samples.jsonl`、`sample_info.json`、`summary.json`；不复制源码或查询Git状态。
- 每档：`single_b*/result.json`、逐轮`outputs_r*.jsonl`。`rank0/result.json`包含核心运行环境、processor配置、冷调用、分阶段耗时、输入/输出token、逐批延迟和显存。
- DP：`dp/result.json`及各rank结果，根`summary.json`另报告对相同每卡batch单卡结果的吞吐加速比和并行效率。
- Profile：短窗口trace、原始算子聚合及profile样本输出，不宣称全样本稳定测量。

稳定窗口含读图/processor/H2D/generate/输出解码，GPU计时显式同步；写盘、日志、模型加载、预热、等待barrier及profiler不计入。耗时统一为微秒；`generated_images_per_s`统计实际完成生成的图片，`attempted_images_per_s`包含尝试处理的图片，图片读取或预算超限直接终止当前配置，不筛除样本、不缩小batch，也不发布该配置的完整吞吐成绩。`generate_only_*_per_gpu_pooled`仅为生成阶段每GPU汇总参考，不能当作多卡端到端吞吐。batch耗时是批延迟，不是单请求延迟。

OOM保留该档失败并停止更大档位；图片读取、预算或运行失败也阻止后续比较。生成内容不正确不会中止扫描。结合吞吐波动、显存余量和trace选择执行配置；记录实际输出token数和生成上限命中情况，避免把输出长度变化直接解释成加速。随机小样本结果仅代表本轮工作负载，不声明场景覆盖或最大输入容量。

接口核对：[Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler)。具体运行使用云端固定依赖版本，不自动升级。
