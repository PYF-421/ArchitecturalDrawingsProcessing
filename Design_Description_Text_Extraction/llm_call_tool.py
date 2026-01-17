import logging
from typing import List, Optional, Dict, Any
import os
from openai import OpenAI
from openai._exceptions import OpenAIError, APIError, APIConnectionError, AuthenticationError

from .base64_util import ImageToBase64Converter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding="utf-8"
)
logger = logging.getLogger("LlmCaller")


API_KEY = "sk-0d85fa6eb17947d79efdf4b0708a7faf"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-max"


class LlmCaller:

    def __init__(self, api_key: str, url: str, model_name: str):
        """
        :param api_key: API Key
        :param url: OpenAI兼容API地址（阿里云百炼：https://dashscope.aliyuncs.com/compatible-mode/v1）
        :param model_name: 模型名称（如qwen-max/qwen-vl-max/qwen-turbo）
        """
        # 基础参数校验
        for param_name, param_value in {"api_key": api_key, "url": url, "model_name": model_name}.items():
            if not param_value or not param_value.strip():
                raise ValueError(f"{param_name} 不能为空")

        self.api_key = api_key.strip()
        self.base_url = url.strip()
        self.model_name = model_name.strip()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        logger.info(f"模型：{self.model_name} | API地址：{self.base_url}")

    def _build_request_content(self, question: str, images_base64: List[str] = None) -> List[Dict[str, Any]]:
        """构建请求内容（文本+图片）"""
        content = []

        # 处理Base64图片（支持多张）
        if isinstance(images_base64, list) and images_base64:
            for idx, img_b64 in enumerate(images_base64):
                if img_b64 and img_b64.strip():
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64.strip()}",
                            "detail": "high"
                        }
                    })
                    logger.debug(f"成功添加第{idx + 1}张图片")
                else:
                    logger.warning(f"跳过第{idx + 1}张空的Base64图片编码")

        # 处理文本问题
        if question and question.strip():
            content.append({
                "type": "text",
                "text": question.strip()
            })
        else:
            raise ValueError("提问文本不能为空")

        return content

    def call(self, question: str, images_base64: List[str] = None, temperature: float = 0.6) -> Optional[str]:
        """
        :param question: 用户提问文本
        :param images_base64: Base64编码的图片列表（PNG/JPG格式），可选
        :param temperature: 生成温度（0-2），值越高输出越随机
        :return: 模型回答文本，失败返回None
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": self._build_request_content(question, images_base64)
                }
            ]

            logger.info(f"调用千问API | 图片数：{len(images_base64) if images_base64 else 0} | 问题：{question[:30]}...")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=24576
            )

            # 4. 提取并返回结果
            response = completion.choices[0].message.content.strip()
            logger.info(f"API调用成功 | 回复长度：{len(response)}字符")
            return response

        except Exception as e:
            logger.error(f"异常：{str(e)}", exc_info=True)
            return None


def run_image_folder_ocr(image_dir: str, output_txt_path: str) -> Optional[str]:
    """
    识别一个文件夹内按列切分好的建筑图纸截图, 并将识别结果写入指定的txt文件。

    参数:
        image_dir: 图片所在文件夹路径, 内部文件名需为 frame_N_col_row.png 形式。
        output_txt_path: 大模型识别后的完整文本输出路径, 如 xxx/yyy/result.txt。

    返回:
        大模型返回的完整文本内容, 如果识别失败则返回None。
    """
    caller = LlmCaller(
        api_key=API_KEY,
        url=API_URL,
        model_name=MODEL_NAME
    )

    def parse_filename(filename):
        """解析文件名，提取列、行信息"""
        # 格式: frame_N_col_row.png
        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 4:
            try:
                frame_idx = int(parts[1])
                col_idx = int(parts[2])
                row_idx = int(parts[3])
                return (frame_idx, col_idx, row_idx)
            except ValueError:
                return None
        return None

    # 获取所有切分图片（排除full.png和debug.png）
    all_files = [
        f for f in os.listdir(image_dir)
        if f.endswith('.png') and not f.endswith('_full.png') and not f.endswith('_debug.png')
    ]

    # 按列分组，每列内按行排序
    columns_dict = {}
    for f in all_files:
        parsed = parse_filename(f)
        if parsed:
            _, col, row = parsed
            if col not in columns_dict:
                columns_dict[col] = []
            columns_dict[col].append((row, f))  # 存储(行号, 文件名)元组

    # 每列内按行号排序（确保按整数排序）
    for col in columns_dict:
        columns_dict[col].sort(key=lambda x: int(x[0]))

    print(f"找到 {len(all_files)} 张切分图片，分布在 {len(columns_dict)} 列")
    for col in sorted(columns_dict.keys(), key=int):
        print(f"  列{col}: {len(columns_dict[col])} 张图片")

    # 按列顺序（1→2→3），每列内按行顺序（1→2→3），将所有图片的base64按顺序放入列表
    image_base64_list = []
    image_info_list = []  # 记录每张图片的信息，用于输出
    
    # 确保列号按整数排序
    for col in sorted(columns_dict.keys(), key=int):
        # 确保行号按整数排序
        sorted_rows = sorted(columns_dict[col], key=lambda x: int(x[0]))
        for row_idx, img_file in sorted_rows:
            img_path = os.path.join(image_dir, img_file)
            try:
                image_base64 = ImageToBase64Converter.image_to_base64(img_path, with_data_uri=False)
                image_base64_list.append(image_base64)
                image_info_list.append(f"列{col}行{row_idx}")
                print(f"✓ 已加载: {img_file} (列{col}, 行{row_idx})")
            except Exception as e:
                print(f"✗ 加载失败 {img_file}: {e}")
    
    # 一次性将所有图片传给大模型（按列顺序）
    if image_base64_list:
        print(f"\n正在调用大模型识别所有图片（共{len(image_base64_list)}张，按列顺序）...")
        
        # 使用已构建的image_info_list来确保顺序映射一致
        image_order_desc = []
        for idx, info in enumerate(image_info_list, 1):
            image_order_desc.append(f"第{idx}张图片 = {info}")
        
        # 构建详细的提示词
        question = (
            "我上传了多张建筑图纸的切分图片，这些图片已经按照列和行的顺序排列好了。"
            "请严格按照图片的输入顺序进行识别，不要重新排序。\n\n"
            "图片顺序映射关系如下（图片按输入顺序排列）：\n" + "\n".join(image_order_desc) + "\n\n"
            "重要要求：\n"
            "1. 必须严格按照图片的输入顺序（第1张、第2张、第3张...）依次识别，不要改变顺序\n"
            "2. 每张图片必须明确标注对应的列号和行号（格式：第X列第Y行）\n"
            "3. 严格按照上述映射关系输出，不要自己判断图片位置或重新排序\n"
            "4. 输出格式：先说明图片序号和位置（如：第1张图片，第1列第1行），然后提取该图片中的所有文字内容\n"
            "5. 如果图片数量不是3列3行，请按照实际图片数量和上述映射关系进行识别\n\n"
            "请开始识别："
        )

        '''
        "6. 当图片中所有文字已经完成提取之后，请分析所提取的文字，找到以下内容所对应的答案，将答案放在提取到的文字的最后篇幅中进行输出\n"
                "a. 确定塑钢窗、铝合金窗等选型\n"
                "b. 明确门窗系列，玻璃类型、层数、厚度，LOW-e 玻璃使用范围\n"
                "c. 确定铝型材材质、厚度要求（粉末喷涂或氟碳涂层）\n"
                "d. 确定是否标明存在木质防火门门套线\n\n"
            "7. 如果存在问题找不到答案的情况，请列出是哪个问题没有找到答案。\n"
        '''
        
        img_response = caller.call(
            question=question,
            images_base64=image_base64_list,
            temperature=0.3
        )
        print(f"\n识图结果：\n{img_response}")
        
        if img_response:
            try:
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(img_response)
                print(f"\n✓ 结果已保存到: {output_txt_path}")
            except Exception as e:
                print(f"\n✗ 保存文件失败: {e}")
    else:
        print("没有找到有效的图片文件")

    return img_response if image_base64_list else None


def image_to_txt(image_dir: str, output_txt_path: str) -> Optional[str]:
    """
    对外推荐使用的简易接口。

    只需传入:
        1) 图片所在文件夹路径 image_dir
        2) 结果保存的txt路径 output_txt_path

    即可完成:
        图片加载 -> 排序 -> 组装提示词 -> 调用大模型 -> 文本写入到txt。
    """
    return run_image_folder_ocr(image_dir=image_dir, output_txt_path=output_txt_path)


        
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "fontstest-2", "frame_8")
    output_txt = os.path.join(image_dir, "frame_8_recognition_result.txt")
    image_to_txt(image_dir, output_txt)
