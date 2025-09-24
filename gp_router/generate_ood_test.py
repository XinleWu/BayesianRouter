import json

Dir = "/data/cs.aau.dk/zh45qz/router_data/RM_Bench/"

# 输入输出文件路径
input_path = Dir + "test_samples_init.json"  # 格式改改；
output_path = Dir + "ood_RMB_subset_test.json"

# 读取 JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

target_rm_indices = [1, 3, 4, 7]  # 只考虑这些 rm
filtered_data = []  # 仅保存非空 bt_pairs 的样本

for sample in data:
    labels = sample["int_labels"]
    bt_pairs = []

    # 只在 target_rm_indices 内部做比较
    for i in target_rm_indices:
        if labels[i] != 1:
            continue
        for j in target_rm_indices:
            if i != j and labels[j] == 0:
                bt_pairs.append([i, j])

    sample["bt_pairs"] = bt_pairs

    # 如果 bt_pairs 不为空，就加入结果集
    if bt_pairs:
        filtered_data.append(sample)

# 统计信息
total_samples = len(data)
non_empty_count = len(filtered_data)
ratio = non_empty_count / total_samples if total_samples > 0 else 0

print(f"总样本数: {total_samples}")
print(f"bt_pairs 非空的样本数: {non_empty_count}")
print(f"比例: {ratio:.2%}")

# 保存仅包含非空 bt_pairs 的样本
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"处理完成，结果已保存到 {output_path}")
