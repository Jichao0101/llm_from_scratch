# 眼睛标注可视化

依赖 Python 3.8+、Pillow（`python -m pip install Pillow`）。读取原始标注的 `dataList[].coordinates` 和 `properties.eye_status`。

编辑独立的 `config.json`，在 `paths` 中填写要处理的目录、图片或标注JSON。可混合多个条目，例如：

```json
{
  "source_root": "/path/to",
  "paths": [
    "/path/to/session_directory",
    "/path/to/another_session/frame.jpg",
    "/path/to/third_session/frame.json"
  ]
}
```

- 目录：处理目录内全部图片，不递归子目录。
- 图片：读取同目录同名JSON，只处理该图片。
- 标注JSON：定位同目录同名图片，只处理该图片。
- 相对路径以配置文件所在目录为基准；重复图片只处理一次。
- `source_root`指定保留目录结构的基准，所有输入图片必须位于其下；它也支持相对于配置文件的路径。
- 输入路径无需写入Python代码或逐个传命令行。JSON中的路径必须用双引号包围。

```bash
python visualize_eyes.py
# 也可指定其他配置文件
python visualize_eyes.py --config /path/to/config.json
```

输出固定为脚本所在仓库的 `temp/<图片相对source_root的路径>.png`。例如 `source_root=/data`，图片 `/data/batch71/session/frame.jpg` 输出到 `visualizer/temp/batch71/session/frame.jpg.png`。无论从哪个工作目录启动，输出位置均不变。再次运行覆盖对应可视化图片；原始数据目录只读，不创建temp或修改原图/JSON；不处理输出目录中的生成图片。

- `id=2`：驾驶员右眼 / 图像左眼；`id=1`：驾驶员左眼 / 图像右眼。
- 图像左眼（id=2）固定绿色，图像右眼（id=1）固定青色，与睁闭眼状态无关。
- 眼框上方直接显示原始状态标签，正常图片没有左上角文字或横幅。
- 缺少JSON/单眼标注、非法ID/状态/眼框及越界框在图像底部用橙色文字提示，并在终端报告。
- hard不被排除，有有效框就画出；缺失或非法框显示提示，不猜测位置。
- 图片保持原始尺寸和存储方向，不自动进行EXIF旋转或Norm1000转换。
