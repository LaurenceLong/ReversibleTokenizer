import functools
import logging
import re
from typing import List, Pattern

from transformers import AutoTokenizer, PreTrainedTokenizerBase

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReversibleTokenizer:
    """为tokenizer添加文本反转功能的装饰器类，支持流式输出"""

    # 用户可见的指令标签
    USER_TAG_START = "<|do_r2l_start|>"
    USER_TAG_END = "<|do_r2l_end|>"

    # 内部用于编码/解码的标记
    MARKER_START = "<|r2l_marker_start|>"
    MARKER_END = "<|r2l_marker_end|>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """初始化可反转tokenizer装饰器"""
        self.tokenizer = tokenizer
        self.original_encode = tokenizer.encode
        self.original_decode = tokenizer.decode

        # 状态追踪 - 用于流式处理
        self.incomplete_markers = {}  # 跟踪不完整的标记 - id -> 部分文本的映射

        # 编译正则表达式
        self.encode_pattern = self._compile_pattern(self.USER_TAG_START, self.USER_TAG_END)
        self.decode_pattern = self._compile_pattern(self.MARKER_START, self.MARKER_END)

        # 部分标记匹配模式
        self.start_marker_pattern = re.compile(re.escape(self.MARKER_START) + r'(.*?)$', re.DOTALL)
        self.end_marker_pattern = re.compile(r'^(.*?)' + re.escape(self.MARKER_END), re.DOTALL)

        # 应用补丁
        self._apply_patch()

    @staticmethod
    def _compile_pattern(start_tag: str, end_tag: str) -> Pattern:
        """编译用于匹配标签的正则表达式"""
        return re.compile(
            re.escape(start_tag) + r'(.*?)' + re.escape(end_tag),
            re.DOTALL
        )

    def _apply_patch(self) -> None:
        """应用编码/解码补丁到tokenizer"""
        # 添加内部标记作为特殊标记
        special_tokens_dict = {'additional_special_tokens': [self.MARKER_START, self.MARKER_END]}
        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"添加了 {num_added} 个内部标记: {special_tokens_dict['additional_special_tokens']}")

        # 获取标记的token ID
        self.marker_start_id = self.tokenizer.convert_tokens_to_ids(self.MARKER_START)
        self.marker_end_id = self.tokenizer.convert_tokens_to_ids(self.MARKER_END)

        # 存储其他特殊标记，用于decode时的清理步骤
        self._store_other_special_tokens()

        # 替换编码和解码方法
        self.tokenizer.encode = functools.partial(self.reversible_encode, self.tokenizer)
        self.tokenizer.decode = functools.partial(self.reversible_decode, self.tokenizer)

        # 添加流式解码方法
        self.tokenizer.stream_decode = functools.partial(self.stream_decode, self.tokenizer)

        logger.info(f"成功为tokenizer应用了反转补丁: {self.tokenizer.__class__.__name__}")

    def _store_other_special_tokens(self) -> None:
        """存储不包括我们的标记在内的其他特殊标记"""
        all_special_ids = self.tokenizer.all_special_ids
        marker_ids = {self.marker_start_id, self.marker_end_id}
        other_special_ids = set(all_special_ids) - marker_ids
        self.tokenizer._other_special_tokens = set(
            self.tokenizer.convert_ids_to_tokens(list(other_special_ids))
        )
        logger.info(f"识别出需要在skip时处理的其他特殊标记: {self.tokenizer._other_special_tokens}")

    def reversible_encode(self, tokenizer: PreTrainedTokenizerBase, text: str, **kwargs) -> List[int]:
        """
        编码方法：反转<|do_r2l_start|>标签内的文本，并在调用原始编码前替换为内部标记
        """

        def reverse_and_mark(match: re.Match) -> str:
            """反转内容并用内部标记包装"""
            original_text = match.group(1)
            reversed_text = original_text[::-1]
            return f"{self.MARKER_START}{reversed_text}{self.MARKER_END}"

        # 用包含反转文本的内部标记替换用户标签
        processed_text = self.encode_pattern.sub(reverse_and_mark, text)

        # 在处理后的文本上调用原始tokenizer的encode
        return self.original_encode(processed_text, **kwargs)

    def reversible_decode(self, tokenizer: PreTrainedTokenizerBase,
                          token_ids: List[int], **kwargs) -> str:
        """
        解码方法：使用原始方法解码，然后找到内部标记，将其内容反转回原始内容，
        并移除标记。然后处理用户对非标记特殊标记的skip_special_tokens首选项。
        """
        # 存储用户对跳过特殊标记的偏好
        user_skip_special_tokens = kwargs.pop('skip_special_tokens', False)

        # 步骤1：保留标记进行解码
        decode_kwargs = kwargs.copy()
        decode_kwargs['skip_special_tokens'] = False
        decoded_text_with_markers = self.original_decode(token_ids, **decode_kwargs)

        # 步骤2和3：找到标记，反转内容，移除标记
        def restore_original_and_remove_markers(match: re.Match) -> str:
            """将内容反转回来并仅返回原始内容"""
            reversed_text = match.group(1).strip()
            return reversed_text[::-1]

        restored_text = self.decode_pattern.sub(restore_original_and_remove_markers, decoded_text_with_markers)

        # 步骤4：处理用户对其他标记的skip_special_tokens偏好
        if user_skip_special_tokens and hasattr(self.tokenizer, "_other_special_tokens"):
            # 按长度降序排序以防止部分替换
            sorted_tokens = sorted(list(self.tokenizer._other_special_tokens), key=len, reverse=True)
            for token in sorted_tokens:
                restored_text = restored_text.replace(token, "")

        return restored_text

    def stream_decode(self, tokenizer: PreTrainedTokenizerBase,
                      token_ids: List[int], stream_id: str = "default", **kwargs) -> str:
        """
        流式解码方法：专为流式输出设计，可以处理跨批次的标记

        Args:
            tokenizer: 使用的tokenizer
            token_ids: 要解码的token IDs
            stream_id: 流识别符，用于追踪多个并行流
            **kwargs: 传递给原始解码器的参数

        Returns:
            str: 当前批次解码后的文本
        """
        # 初始化流状态（如果不存在）
        if stream_id not in self.incomplete_markers:
            self.incomplete_markers[stream_id] = {
                "pending_start": None,  # 等待结束标记的开始标记内容
                "buffer": [],  # 收集待处理的token
                "output_buffer": "",  # 缓存的输出文本
                "processed_text": ""  # 已处理的文本
            }

        stream_state = self.incomplete_markers[stream_id]

        # 将新token追加到buffer
        stream_state["buffer"].extend(token_ids)

        # 使用完整的buffer解码
        decode_kwargs = kwargs.copy()
        decode_kwargs['skip_special_tokens'] = False
        current_full_text = self.original_decode(stream_state["buffer"], **decode_kwargs)

        # 检查是否有完整的标记对
        if self.MARKER_START in current_full_text and self.MARKER_END in current_full_text:
            # 处理完整的标记对
            processed_text = self._process_complete_markers(current_full_text)
            # 更新状态并准备输出
            new_text = processed_text[len(stream_state["processed_text"]):]
            stream_state["processed_text"] = processed_text
            return new_text

        # 如果没有完整的标记，但发现了开始标记
        elif self.MARKER_START in current_full_text and self.MARKER_END not in current_full_text:
            # 只返回标记之前的文本
            parts = current_full_text.split(self.MARKER_START, 1)
            if len(parts) > 1:
                start_text = parts[0]
                new_text = start_text[len(stream_state["processed_text"]):]
                stream_state["processed_text"] = start_text
                return new_text
            return ""

        # 如果既没有开始标记也没有结束标记，则处理所有文本
        else:
            # 返回新的文本部分
            new_text = current_full_text[len(stream_state["processed_text"]):]
            stream_state["processed_text"] = current_full_text
            return new_text

    def _process_complete_markers(self, text: str) -> str:
        """处理完整的标记对"""

        def restore_original_and_remove_markers(match: re.Match) -> str:
            """将内容反转回来并仅返回原始内容"""
            reversed_text = match.group(1).strip()
            return reversed_text[::-1]

        return self.decode_pattern.sub(restore_original_and_remove_markers, text)

    def reset_stream(self, stream_id: str = "default") -> None:
        """重置特定流的状态"""
        if stream_id in self.incomplete_markers:
            del self.incomplete_markers[stream_id]


# 便捷函数
def patch_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """为tokenizer应用反转文本的补丁，并返回修改后的tokenizer"""
    reversible = ReversibleTokenizer(tokenizer)
    # 添加重置方法
    tokenizer.reset_stream = reversible.reset_stream
    return tokenizer

