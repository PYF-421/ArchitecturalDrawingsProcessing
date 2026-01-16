import logging
from typing import List, Optional, Dict, Any
import os
from openai import OpenAI
from openai._exceptions import OpenAIError, APIError, APIConnectionError, AuthenticationError

from base64_util import ImageToBase64Converter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding="utf-8"
)
logger = logging.getLogger("LlmCaller")


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


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 配置参数（替换为你的实际参数）
    API_KEY = "sk-0d85fa6eb17947d79efdf4b0708a7faf"
    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen-vl-max"  # 推荐使用qwen-max/qwen-vl-max

    # 初始化全能力调用器
    caller = LlmCaller(
        api_key=API_KEY,
        url=API_URL,
        model_name=MODEL_NAME
    )

    # # 示例1：纯文本提问
    # print("===== 纯文本提问 =====")
    # text_response = caller.call(
    #     question="什么是计算机视觉？用通俗的话解释",
    #     temperature=0.5
    # )
    # print(f"回答：{text_response}\n")

    print("===== 图片识图（按列顺序） =====")
    # 这里修改为目录文件名
    image_dir = r"C:\Users\Admin\Desktop\ArchitecturalDrawingsProcessing\ArchitecturalDrawingsProcessing\fontstest-2\frame_2"

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
        
        img_response = caller.call(
            question=question,
            images_base64=image_base64_list,
            temperature=0.3  # 降低温度，提高一致性
        )
        print(f"\n识图结果：\n{img_response}")
        
        # 保存大模型输出结果到txt文件
        if img_response:
            frame_name = os.path.basename(image_dir)  # 例如: frame_2
            output_txt_path = os.path.join(image_dir, f"{frame_name}_recognition_result.txt")
            
            try:
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(img_response)
                print(f"\n✓ 结果已保存到: {output_txt_path}")
            except Exception as e:
                print(f"\n✗ 保存文件失败: {e}")
    else:
        print("没有找到有效的图片文件")
