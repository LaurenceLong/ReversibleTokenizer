from transformers import AutoTokenizer

from reversible_tokenizer import patch_tokenizer

# 示例用法
if __name__ == "__main__":
    model_name = "google-bert/bert-base-uncased"
    print(f"加载tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"已加载tokenizer类型: {type(tokenizer)}")

    # 应用补丁
    tokenizer = patch_tokenizer(tokenizer)

    # 测试标准解码
    original_string = "Order number is <|do_r2l_start|>1003456<|do_r2l_end|>, please process it quickly."
    print("\n原始字符串:")
    print(original_string)

    token_ids = tokenizer.encode(original_string, add_special_tokens=False)
    print("\n编码后token_ids:")
    print(token_ids)

    token_ids_1 = tokenizer(text=original_string)
    print("\n编码后token_ids_1:")
    print(token_ids_1)

    token_list = []
    for tok_id in token_ids:
        token_list.append(tokenizer.decode([tok_id]))
    print("\n解码字符串列表:")
    print(token_list)

    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    print("\n解码后的字符串:")
    print(decoded)

    # 测试流式解码
    print("\n流式解码测试:")
    # 将token分成多批
    batch_size = 4
    batches = [token_ids[i:i + batch_size] for i in range(0, len(token_ids), batch_size)]

    print("分批token数量:", [len(batch) for batch in batches])

    # 模拟流式输出
    stream_output = ""
    for i, batch in enumerate(batches):
        part = tokenizer.stream_decode(batch, stream_id="test")
        stream_output += part
        print(f"批次 {i + 1} 输出: '{part}'")

    print("\n完整流式输出:")
    print(stream_output)

    # 验证流式输出与标准解码一致
    print("\n流式输出与标准解码一致:", stream_output == decoded)

    # 重置流
    tokenizer.reset_stream("test")
    print("\n已重置流状态")
