import functools
import os.path
import re
from typing import List, Optional, Union

from transformers import PreTrainedTokenizerBase, BatchEncoding


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
        self.original_call_one = tokenizer._call_one

        # 状态追踪 - 用于流式处理
        self.incomplete_markers = {}

        # 编译正则表达式
        self.encode_pattern = re.compile(
            re.escape(self.USER_TAG_START) + r'(.*?)' + re.escape(self.USER_TAG_END),
            re.DOTALL
        )
        self.decode_pattern = re.compile(
            re.escape(self.MARKER_START) + r'(.*?)' + re.escape(self.MARKER_END),
            re.DOTALL
        )

        # 应用补丁
        self._apply_patch()

    def _apply_patch(self) -> None:
        """应用编码/解码补丁到tokenizer"""
        # 添加内部标记作为特殊标记
        special_tokens_dict = {'additional_special_tokens': [self.MARKER_START, self.MARKER_END]}
        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"添加了 {num_added} 个内部标记: {special_tokens_dict['additional_special_tokens']}")

        # 替换原始方法
        self.tokenizer.encode = functools.partial(self.reversible_encode, self.tokenizer)
        self.tokenizer.decode = functools.partial(self.reversible_decode, self.tokenizer)
        self.tokenizer._call_one = functools.partial(self.reversible_call_one, self.tokenizer)

        # 添加流式解码方法
        self.tokenizer.stream_decode = functools.partial(self.stream_decode, self.tokenizer)

        print(f"成功为tokenizer应用了反转补丁: {self.tokenizer.__class__.__name__}")

    def reversible_call_one(self, tokenizer: PreTrainedTokenizerBase,
                            text: Union[str, List[str], List[List[str]]],
                            text_pair: Optional[Union[str, List[str], List[List[str]]]] = None,
                            **kwargs) -> BatchEncoding:
        """
        重写_call_one方法，在调用原始方法前处理文本反转
        """
        # 处理文本反转，仅当输入是字符串时
        if isinstance(text, str):
            processed_text = self._process_text_for_reversing(text)
            # 调用原始_call_one方法
            return self.original_call_one(processed_text, text_pair, **kwargs)
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            # 处理字符串列表
            processed_text = [self._process_text_for_reversing(t) for t in text]
            return self.original_call_one(processed_text, text_pair, **kwargs)
        elif isinstance(text, list) and all(isinstance(t, list) and all(isinstance(s, str) for s in t) for t in text):
            # 处理字符串列表的列表
            processed_text = [[self._process_text_for_reversing(s) for s in t] for t in text]
            return self.original_call_one(processed_text, text_pair, **kwargs)

        # 处理text_pair，如果存在且是字符串
        if text_pair is not None:
            if isinstance(text_pair, str):
                processed_text_pair = self._process_text_for_reversing(text_pair)
                return self.original_call_one(text, processed_text_pair, **kwargs)
            elif isinstance(text_pair, list) and all(isinstance(t, str) for t in text_pair):
                processed_text_pair = [self._process_text_for_reversing(t) for t in text_pair]
                return self.original_call_one(text, processed_text_pair, **kwargs)
            elif isinstance(text_pair, list) and all(
                    isinstance(t, list) and all(isinstance(s, str) for s in t) for t in text_pair):
                processed_text_pair = [[self._process_text_for_reversing(s) for s in t] for t in text_pair]
                return self.original_call_one(text, processed_text_pair, **kwargs)

        # 如果输入不是字符串或字符串列表，直接调用原始方法
        return self.original_call_one(text, text_pair, **kwargs)

    def _process_text_for_reversing(self, text: str) -> str:
        """处理文本中的反转标记，保持空格处理的一致性"""

        def reverse_and_mark(match):
            original_text = match.group(1)
            # 保留原始文本的空格，仅反转字符
            reversed_text = original_text[::-1]
            # 不添加额外空格
            return f"{self.MARKER_START}{reversed_text}{self.MARKER_END}"

        return self.encode_pattern.sub(reverse_and_mark, text)

    def reversible_encode(self, tokenizer: PreTrainedTokenizerBase, text: str, **kwargs) -> List[int]:
        """
        编码方法：反转<|do_r2l_start|>标签内的文本，并在调用原始编码前替换为内部标记
        """
        processed_text = self._process_text_for_reversing(text)

        # 在处理后的文本上调用原始tokenizer的encode
        return self.original_encode(processed_text, **kwargs)

    def reversible_decode(self, tokenizer: PreTrainedTokenizerBase,
                          token_ids: List[int], **kwargs) -> str:
        """
        解码方法：使用原始方法解码，然后找到内部标记，将其内容反转回原始内容
        """
        # 存储用户对跳过特殊标记的偏好
        user_skip_special_tokens = kwargs.pop('skip_special_tokens', False)
        # 存储用户对清理标记化空格的偏好，默认为False以保持空格
        clean_up_tokenization_spaces = kwargs.pop('clean_up_tokenization_spaces', False)

        # 步骤1：保留标记进行解码
        decode_kwargs = kwargs.copy()
        decode_kwargs['skip_special_tokens'] = False
        decode_kwargs['clean_up_tokenization_spaces'] = clean_up_tokenization_spaces
        decoded_text_with_markers = self.original_decode(token_ids, **decode_kwargs)

        # 步骤2：找到标记，反转内容，移除标记
        # 这个正则表达式处理需要更精确，以确保不引入额外空格
        def restore_original_and_remove_markers(match):
            # 获取标记内的文本并反转回来，移除任何前导和尾随空格
            reversed_text = match.group(1).strip()
            return reversed_text[::-1]

        restored_text = self.decode_pattern.sub(restore_original_and_remove_markers, decoded_text_with_markers)

        # 步骤3：如果用户要求，跳过特殊标记
        if user_skip_special_tokens:
            # 移除所有特殊标记，但保持空格处理一致
            for token in tokenizer.all_special_tokens:
                restored_text = restored_text.replace(token, "")

        return restored_text

    def stream_decode(self, tokenizer: PreTrainedTokenizerBase,
                      token_ids: List[int], stream_id: str = "default", **kwargs) -> str:
        """
        流式解码方法：专为流式输出设计，可以处理跨批次的标记
        """
        # 获取清理标记化空格的设置
        clean_up_tokenization_spaces = kwargs.pop('clean_up_tokenization_spaces', False)

        # 初始化流状态
        if stream_id not in self.incomplete_markers:
            self.incomplete_markers[stream_id] = {
                "buffer": [],
                "processed_text": "",
                "in_marker": False
            }

        state = self.incomplete_markers[stream_id]

        # 将新token添加到缓冲区
        state["buffer"].extend(token_ids)

        # 使用完整buffer解码
        decode_kwargs = kwargs.copy()
        decode_kwargs['skip_special_tokens'] = False
        decode_kwargs['clean_up_tokenization_spaces'] = clean_up_tokenization_spaces
        current_full_text = self.original_decode(state["buffer"], **decode_kwargs)

        # 处理反转标记
        processed_text = self._process_stream_text(current_full_text, state)

        # 计算新增的文本部分
        new_text = processed_text[len(state["processed_text"]):]
        state["processed_text"] = processed_text

        return new_text

    def _process_stream_text(self, text: str, state: dict) -> str:
        """处理流文本中的反转标记，状态跟踪确保标记处理正确"""
        # 完整标记对处理
        if self.MARKER_START in text and self.MARKER_END in text:
            # 处理所有完整的标记对
            processed = self.decode_pattern.sub(
                lambda m: m.group(1).strip()[::-1],
                text
            )
            state["in_marker"] = False
            return processed

        # 处理开始了但未结束的标记
        elif self.MARKER_START in text and not self.MARKER_END in text:
            # 记录我们正在处理一个标记
            state["in_marker"] = True
            # 只返回标记开始前的文本
            before_marker = text.split(self.MARKER_START, 1)[0]
            return before_marker

        # 当之前有未完成的标记，但这一批次没有结束标记时，不返回新文本
        elif state["in_marker"] and not self.MARKER_END in text:
            # 仍在标记内，不返回新内容
            return state["processed_text"]

        # 没有任何标记或标记已处理完，直接返回
        return text

    def reset_stream(self, stream_id: str = "default") -> None:
        """重置特定流的状态"""
        if stream_id in self.incomplete_markers:
            del self.incomplete_markers[stream_id]


# 便捷函数
def patch_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """为tokenizer应用反转文本的补丁，并返回修改后的tokenizer"""
    # 设置clean_up_tokenization_spaces为False
    tokenizer.clean_up_tokenization_spaces = False
    reversible = ReversibleTokenizer(tokenizer)
    # 添加重置方法
    tokenizer.reset_stream = reversible.reset_stream
    return tokenizer
