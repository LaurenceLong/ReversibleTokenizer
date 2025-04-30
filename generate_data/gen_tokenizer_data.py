"""
生成用于训练可反转tokenizer的数据脚本。
该脚本为现有tokenizer词汇表中的token生成对应的正向和反向表示，
以帮助模型学习token对应的R2L表示。
"""

import argparse
import json
import os
import random
# 导入项目的ReversibleTokenizer类
import sys
from typing import List, Dict, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reversible_tokenizer import ReversibleTokenizer


def save_examples(examples: List[dict], output_file: str) -> None:
    """保存生成的示例到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    print(f"已保存 {len(examples)} 个训练示例到 {output_file}")


def generate_token_examples(
        tokenizer: PreTrainedTokenizerBase,
        max_examples: int = 100000,
        min_token_freq: int = 5,
        exclude_special_tokens: bool = True,
        seed: int = 42
) -> Tuple[List[Dict], Dict]:
    """
    为tokenizer的词汇表生成训练示例。

    Args:
        tokenizer: 要生成数据的tokenizer
        max_examples: 要生成的最大示例数
        min_token_freq: 每个token至少生成的示例数
        exclude_special_tokens: 是否排除特殊token
        seed: 随机种子

    Returns:
        Tuple[List[str], Dict]: 生成的示例列表和统计信息
    """
    random.seed(seed)

    # 获取词汇表（排除特殊token）
    vocab = list(tokenizer.get_vocab().items())
    vocab_size = len(vocab)

    # 过滤特殊token
    special_token_ids = set(tokenizer.all_special_ids) if exclude_special_tokens else set()
    filtered_vocab = [(token, idx) for token, idx in vocab
                      if idx not in special_token_ids
                      and len(token) > 1]  # 排除单字符token以获得更有意义的示例

    print(f"词汇表大小: {vocab_size}, 过滤后: {len(filtered_vocab)}")

    examples = []
    stats = {
        "total_tokens_processed": len(filtered_vocab),
        "examples_generated": 0,
        "tokens_covered": 0,
        "token_frequency": {}
    }

    # 为每个token创建至少min_token_freq个示例
    token_examples_count = {}

    # 计算需要多少轮才能达到min_token_freq
    rounds_needed = (min_token_freq + 2) // 3  # 每个token平均可以生成3种模式

    raw_to_r2l_instruction_list = [
        "Represent text in a right to left way",
        "Convert this text to read from right to left",
        "Transform the following text to read backwards",
        "Reverse the character order in this text",
        "Display this text in reverse character order",
        "Change the direction of text to be read from right to left",
        "Flip the character order in this text",
        "Rearrange these characters to read from right to left",
        "Format this text to be read in reverse order",
        "Mirror the character sequence in this text",
        "Present this text in a way that reads from the end to the beginning",
        "Invert the direction of character placement in this text",
        "Turn this text around to read from right to left",
        "Process this text to reverse its character ordering",
        "Modify this text to have a right-to-left reading direction",
        "Apply right-to-left character ordering to this text",
        "Restructure this text to read backwards by character",
        "Arrange the characters in this text in reverse sequence",
        "Encode this text in a right-to-left format",
        "Change the reading direction of this text to right-to-left"
    ]

    r2l_to_raw_instruction_list = [
        "Convert right to left way text to normal way",
        "Transform reversed text back to normal reading order",
        "Change text from right-to-left to standard left-to-right format",
        "Restore reversed text to its original character order",
        "Return this backwards text to normal reading direction",
        "Revert right-to-left text to conventional reading order",
        "Convert reversed character sequence back to normal",
        "Fix the direction of this text to read from left to right",
        "Normalize the character order in this reversed text",
        "Correct the reading direction of this text",
        "Restore the natural reading order of this reversed text",
        "Make this right-to-left text readable in standard format",
        "Repair the character sequence of this reversed text",
        "Adjust this backwards text to read normally",
        "Reorganize this reversed text to conventional reading order",
        "Decode this right-to-left text to normal format",
        "Return this text to proper reading direction",
        "Process reversed text to display in regular order",
        "Translate this backwards text to standard character ordering",
        "Fix this reversed text so it reads naturally"
    ]

    with tqdm(total=min(max_examples, len(filtered_vocab) * rounds_needed)) as pbar:
        for round_idx in range(rounds_needed):
            if stats["examples_generated"] >= max_examples:
                break

            # 每轮随机打乱词汇表
            random.shuffle(filtered_vocab)

            for token, token_id in filtered_vocab:
                if stats["examples_generated"] >= max_examples:
                    break

                # 如果这个token已经有足够的示例，跳过
                if token in token_examples_count and token_examples_count[token] >= min_token_freq:
                    continue

                # 获取token的原始文本表示
                text = tokenizer.decode([token_id])

                # 增加计数
                if token not in token_examples_count:
                    token_examples_count[token] = 0
                    stats["tokens_covered"] += 1

                # 随机选择生成模式
                mode = random.choice([1, 2])

                # alpaca instruction
                if mode == 1:
                    # 模式1: text -> <|do_r2l_start|>text<|do_r2l_end|>
                    s_instruction = random.choice(raw_to_r2l_instruction_list)
                    s_input = text
                    s_output = f"{ReversibleTokenizer.USER_TAG_START}{text}{ReversibleTokenizer.USER_TAG_END}"
                    example = dict(instruction=s_instruction, input=s_input, output=s_output, )
                    examples.append(example)
                elif mode == 2:
                    # 模式2: <|do_r2l_start|>text<|do_r2l_end|> -> text
                    s_instruction = random.choice(r2l_to_raw_instruction_list)
                    s_input = f"{ReversibleTokenizer.USER_TAG_START}{text}{ReversibleTokenizer.USER_TAG_END}"
                    s_output = text
                    example = dict(instruction=s_instruction, input=s_input, output=s_output, )
                    examples.append(example)

                # 更新统计信息
                token_examples_count[token] = token_examples_count.get(token, 0) + 1
                if token not in stats["token_frequency"]:
                    stats["token_frequency"][token] = 0
                stats["token_frequency"][token] += 1

                stats["examples_generated"] += 1
                pbar.update(1)

    return examples, stats


def main():
    parser = argparse.ArgumentParser(description='生成用于训练可反转tokenizer的数据')
    parser.add_argument('--model_name', type=str, default="unsloth/Qwen2.5-7B",
                        help='用于生成数据的基础模型名称')
    parser.add_argument('--output_file', type=str, default="tokenizer_r2l_training_data.txt",
                        help='输出文件路径')
    parser.add_argument('--max_examples', type=int, default=100000,
                        help='要生成的最大示例数')
    parser.add_argument('--min_token_freq', type=int, default=5,
                        help='每个token至少生成的示例数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--stats_file', type=str, default="tokenizer_r2l_stats.json",
                        help='统计信息输出文件')

    args = parser.parse_args()

    # 加载tokenizer
    print(f"加载tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 生成示例
    print(f"开始生成训练示例，最大数量: {args.max_examples}")
    examples, stats = generate_token_examples(
        tokenizer=tokenizer,
        max_examples=args.max_examples,
        min_token_freq=args.min_token_freq,
        seed=args.seed
    )

    # 保存示例
    save_examples(examples, args.output_file)

    # 保存统计信息
    with open(args.stats_file, 'w', encoding='utf-8') as f:
        # 只保留前100个token的频率信息以保持文件大小合理
        top_tokens = dict(sorted(stats["token_frequency"].items(), key=lambda x: x[1], reverse=True)[:100])
        stats["token_frequency"] = top_tokens
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"统计信息已保存到 {args.stats_file}")
    print(f"总共生成了 {stats['examples_generated']} 个示例，覆盖了 {stats['tokens_covered']} 个token")


if __name__ == "__main__":
    main()
