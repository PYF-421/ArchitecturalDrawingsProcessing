## Design_Description_Text_Extraction 模块说明

本目录提供“设计说明文字提取”流程示例，整体步骤为：

1. 读取 DXF 图纸并按图框自动切分为多张 PNG（`fontstest.py`）。
2. 使用多模态大模型对每个图框切分图片做文字识别，按列顺序整理结果（`llm_call_tool.py`）。
3. 将所有识别结果的 txt 文件合并为一个总的说明文件（`Text_Concatenation.py`）。

配合本目录的 `demo.py`，可以快速跑通从 DXF 到最终合并文本的完整 Demo。

---

## 目录与核心脚本

- `fontstest.py`
  - 功能：DXF 图框识别与渲染，将 DXF 中的图框自动切成若干张 PNG。
  - 主要接口：
    - `split_dxf_to_png(dxf_path=None, output_dir=None)`
      - 不传参数时，使用文件顶部的默认 `interest.dxf` 和 `fontstest-2` 目录。
      - 传入参数时，会覆盖默认路径：
        - `dxf_path`：输入 DXF 文件绝对或相对路径。
        - `output_dir`：输出 PNG 图像根目录。
  - 输出结构示例：
    - `fontstest-2/frame_1/*.png`
    - `fontstest-2/frame_2/*.png`

- `llm_call_tool.py`
  - 功能：调用千问多模态大模型，对指定目录内的切分 PNG 进行文字识别。
  - 主要接口：
    - `image_to_txt(image_dir: str, output_txt_path: str) -> Optional[str]`
      - `image_dir`：某一个 `frame_x` 目录，例如 `fontstest-2/frame_8`。
      - `output_txt_path`：识别结果保存的 txt 文件路径。
  - 用法特点：
    - 会按文件名中的列、行信息排序（如 `frame_8_1_1.png`），严格按列顺序传给大模型。
    - 输出文本中会带上“第几列第几行”的描述，方便后续对照图纸。

- `Text_Concatenation.py`
  - 功能：将一个文件夹内的多个 txt 文件，按文件名排序合并成一个大 txt。
  - 主要接口：
    - `merge_txt_in_folder(input_dir: str, output_path: str) -> str`
      - 通用合并函数，可用于任何 txt 文件夹。
    - `merge_fontstest_txt() -> str`
      - 针对本 Demo 的快捷封装：
        - 输入目录固定为：`Design_Description_Text_Extraction/fontstest-2/txt`
        - 输出文件为：`merged_all_frames.txt`

- `base64_util.py`
  - 功能：将 PNG 图片转为 Base64 字符串，供 `llm_call_tool.py` 调用时使用。
  - 主要类：
    - `ImageToBase64Converter`

- `__init__.py`
  - 将本目录封装为可导入的包，对外暴露统一接口：
    - `split_dxf_to_png`
    - `image_to_txt`
    - `ImageToBase64Converter`
    - `merge_fontstest_txt`

---

## 一条龙 Demo：从 DXF 到合并文本

根目录下的 `demo.py` 展示了一个完整的调用示例。逻辑如下：

```python
from Design_Description_Text_Extraction import split_dxf_to_png, image_to_txt, merge_txt_in_folder
import os
from concurrent.futures import ThreadPoolExecutor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 切分DXF文件为PNG图片
dxf_path = os.path.join(SCRIPT_DIR, "Design_Description_Text_Extraction", "interest.dxf")   
output_dir = os.path.join(SCRIPT_DIR, "Design_Description_Text_Extraction", "fontstest-2")
split_dxf_to_png(dxf_path, output_dir)


# 2. 对切分后的图片进行文字识别
txt_root = os.path.join(output_dir, "txt")
os.makedirs(txt_root, exist_ok=True)

frames = [
    name for name in os.listdir(output_dir)
    if name.startswith("frame_") and os.path.isdir(os.path.join(output_dir, name))
]

with ThreadPoolExecutor(max_workers=10) as executor:    # （并行调用）通过max_workers设置最多多少个线程并行处理
    future_to_frame = {}
    for frame_name in frames:
        frame_dir = os.path.join(output_dir, frame_name)
        output_txt = os.path.join(txt_root, f"{frame_name}_recognition_result.txt")
        future = executor.submit(image_to_txt, frame_dir, output_txt)
        future_to_frame[future] = frame_name

    for future, frame_name in future_to_frame.items():
        try:
            future.result()
            print(f"{frame_name} done")
        except Exception as e:
            print(f"{frame_name} error: {e}")


# 3. 合并所有文本文件
output_txt = os.path.join(txt_root, "merged_all_frames.txt")
merge_txt_in_folder(txt_root, output_txt)
```

执行步骤建议：

1. 确认 `interest.dxf` 放在 `Design_Description_Text_Extraction` 目录下。
2. 在项目根目录运行：
   - `python demo.py`
3. 查看输出：
   - 切分图片：`Design_Description_Text_Extraction/fontstest-2/frame_x/*.png`
   - 单个图框识别结果：`Design_Description_Text_Extraction/fontstest-2/txt/frame_x_recognition_result.txt`
   - 合并后的总说明：`Design_Description_Text_Extraction/fontstest-2/txt/merged_all_frames.txt`

---

## 注意事项

- API Key 等敏感信息在 `llm_call_tool.py` 中配置，实际使用时请改为自己的安全配置方式。
- 如需调整并行度，可以在 `debug.py` 中修改 `ThreadPoolExecutor(max_workers=...)` 的参数。
- 如果 DXF 文件名或输出目录有变化，优先通过调用 `split_dxf_to_png(dxf_path, output_dir)` 的方式传入，不必修改内部脚本常量。
