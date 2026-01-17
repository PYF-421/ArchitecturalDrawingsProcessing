import base64
import os
from typing import Optional, Union


class ImageToBase64Converter:
    """图片转Base64编码工具类"""

    @staticmethod
    def get_image_mime_type(image_path: str) -> Optional[str]:
        """
        获取图片文件的MIME类型
        :param image_path: 图片文件路径
        :return: MIME类型字符串（如image/jpeg），未知类型返回None
        """
        # 映射常见图片后缀到MIME类型
        mime_mapping = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml'
        }
        # 获取文件后缀并转换为小写
        ext = os.path.splitext(image_path)[1].lower()
        return mime_mapping.get(ext)

    @classmethod
    def image_to_base64(
            cls,
            image_source: Union[str, bytes],
            with_data_uri: bool = True
    ) -> str:
        """
        将图片转换为Base64编码字符串
        :param image_source: 图片路径（字符串）或图片二进制数据（bytes）
        :param with_data_uri: 是否返回带Data URI前缀的Base64字符串（默认True）
        :return: Base64编码字符串
        """
        try:
            # 处理输入：如果是路径则读取文件，否则直接使用二进制数据
            if isinstance(image_source, str):
                # 检查文件是否存在
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"图片文件不存在: {image_source}")
                # 读取图片二进制数据
                with open(image_source, 'rb') as f:
                    image_data = f.read()
            elif isinstance(image_source, bytes):
                image_data = image_source
            else:
                raise TypeError("image_source 必须是文件路径字符串或二进制bytes数据")

            # 编码为Base64字符串
            base64_encoded = base64.b64encode(image_data).decode('utf-8')

            # 如果需要Data URI前缀，拼接MIME类型
            if with_data_uri:
                if isinstance(image_source, str):
                    mime_type = cls.get_image_mime_type(image_source) or 'application/octet-stream'
                else:
                    # 如果是二进制数据，默认使用通用二进制类型
                    mime_type = 'application/octet-stream'
                base64_encoded = f"data:{mime_type};base64,{base64_encoded}"

            return base64_encoded

        except Exception as e:
            raise RuntimeError(f"图片转Base64失败: {str(e)}") from e

    @staticmethod
    def base64_to_image(base64_str: str, save_path: str) -> None:
        """
        将Base64编码字符串转回图片文件并保存
        :param base64_str: Base64编码字符串（可带Data URI前缀）
        :param save_path: 保存图片的路径
        """
        try:
            # 移除Data URI前缀（如果存在）
            if ';base64,' in base64_str:
                base64_str = base64_str.split(';base64,')[1]

            # 解码Base64并保存文件
            image_data = base64.b64decode(base64_str)
            with open(save_path, 'wb') as f:
                f.write(image_data)

        except Exception as e:
            raise RuntimeError(f"Base64转图片失败: {str(e)}") from e


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 测试图片路径（替换为你的图片路径）
    test_image_path = "test.jpg"
    # 测试保存路径
    output_image_path = "output_from_base64.jpg"

    try:
        # 1. 图片转Base64（带Data URI前缀）
        base64_with_uri = ImageToBase64Converter.image_to_base64(test_image_path)
        print("带Data URI的Base64编码（前50个字符）:", base64_with_uri[:50])

        # 2. 图片转纯Base64字符串（无前缀）
        base64_pure = ImageToBase64Converter.image_to_base64(test_image_path, with_data_uri=False)
        print("纯Base64编码（前50个字符）:", base64_pure[:50])

        # 3. Base64转回图片
        ImageToBase64Converter.base64_to_image(base64_pure, output_image_path)
        print(f"Base64已转回图片并保存到: {output_image_path}")

    except Exception as e:
        print(f"执行出错: {e}")