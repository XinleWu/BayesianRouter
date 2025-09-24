# from huggingface_hub import HfApi, create_repo, upload_folder, login
#
# login(token="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")
#
# # 1. 设置您的模型路径和想要创建的仓库名
# model_path = "/data/cs.aau.dk/zh45qz/router_data/helpsteer3/oot"  # 替换为您的文件夹实际路径，例如 "./my_awesome_model"
# repo_id = "wuxinle/output8-3500"  # 例如 "tony/llm-checkpoint-4328"
#
# # 2. (可选) 在HF上创建仓库。如果仓库已存在，这步会跳过。
# create_repo(repo_id, exist_ok=True)
#
# # 3. 初始化API客户端
# api = HfApi()
#
# # 4. 上传整个文件夹！这是最简单的方法。
# api.upload_folder(
#     folder_path=model_path,
#     repo_id=repo_id,
#     repo_type="model"
# )
#
# print(f"模型已成功上传至: https://huggingface.co/{repo_id}")



# RM0: 0.6361
# RM1: 0.6452
# RM3: 0.6482
#
# RM2: 0.7240; 0.7453
# online: 0.7248; 0.7548
# both: 0.7248; 0.7596
# offline: 0.7233; 0.7460
#
#
#
# RM1: 0.7233  0.7532
# RM2: 0.7172  0.7453
# RM3: 0.7202  0.7512
# offline: 0.7165 0.7548
# online: 0.7195 0.7595
# both: 0.7202 0.7640
#
#
# gsm8k:
# 8B: 0.7066
# RM0: 0.7218
# RM1: 0.7195, loss=0.5709
# RM2: 0.7180
# RM3: 0.7036, loss=0.5841
# offline: 0.7089, loss=0.5773
# online: 0.7058, loss=0.5689
# both: 0.7142, loss=0.5670
#
#
#
# 3B judge  8B judge
# RM0: 0.5627
# RM1: 0.5702
# RM2: 0.5702
# RM3: 0.5727
# offline: 0.5702
# online: 0.5627
# both: 0.5652
#
#
#
#
#
# constrained:
# rm0: 22.39
# rm1: 12.35
# rm2: 14.12
# rm3: 14.66
# online:
# offline: 13.41, 13.64, 13.68, 12.39
# both:
#
#
#
#
#
#
#
# rm0: 23.46
# rm1: 17.31
# rm2: 24.09
# rm3: 14.64
# offline: 18.04
# both: 19.70











import matplotlib.pyplot as plt
import numpy as np

# Methods
methods = ["Fastest RM", "Slowest RM", "Majority Voting", "LASER", "BayesianRouter"]

# Replace with your real data
training_times_4RMs = [10, 25, 30, 15, 18]
training_times_8RMs = [12, 40, 50, 18, 22]

# Number of methods
n_methods = len(methods)

# Bar width and positions
bar_width = 0.15
x_positions = np.array([0, 1])  # 0 for 4RMs, 1 for 8RMs

fig, ax = plt.subplots(figsize=(3.5, 2.5))  # small figure for half column

# For each method, plot bars slightly shifted
for i, method in enumerate(methods):
    offsets = (i - (n_methods - 1)/2) * bar_width
    ax.bar(x_positions + offsets,
           [training_times_4RMs[i], training_times_8RMs[i]],
           width=bar_width,
           label=method)

# Labels
ax.set_xticks(x_positions)
ax.set_xticklabels(["4 RMs", "8 RMs"])
ax.set_ylabel("Training Time (hours)")
ax.set_title("Training Efficiency")

# Legend in top-right corner
# ax.legend(fontsize=8, loc="upper right")
# ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 1.15))
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()


# Optional grid
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()




# def find_integer_x(limit, decimals, tolerance=1e-6):
#     """
#     在小于 limit 的整数中寻找 x，使得 x*decimals[i] 接近整数。
#
#     :param limit: 上限整数
#     :param decimals: 小数列表
#     :param tolerance: 容差
#     :return: 满足条件的 x 值列表
#     """
#     results = []
#     for x in range(1, limit):  # 从1到limit-1
#         ok = True
#         for d in decimals:
#             product = x * d
#             if abs(product - round(product)) > tolerance:
#                 ok = False
#                 break
#         if ok:
#             results.append(x)
#     return results
#
#
# if __name__ == "__main__":
#     limit = 2017
#     decimals = [
#         0.8972721996517702,
#         0.77539175856065,
#         0.8142774230992456,
#         0.7951247823563552,
#         0.8113755078351712,
#         0.7864190365641324,
#         # 0.9031,
#     ]
#
#     xs = find_integer_x(limit, decimals, tolerance=1e-6)
#     if xs:
#         print("找到的满足条件的整数 x：", xs)
#         for x in xs:
#             print(f"\n验证 x = {x}:")
#             for d in decimals:
#                 product = x * d
#                 print(f"{x} * {d} = {product:.12f} ≈ {round(product)}")
#     else:
#         print("没有找到满足条件的整数。")

# import json
# import re
#
#
# def parse_choice_from_output(output_text):
#     """
#     尝试从模型输出中提取字母选项 A/B/C/D（或 a-d）。
#     支持这些格式：
#       - “The answer is B.”
#       - “答案是 B”
#       - “答案：B”
#       - “选项 C”
#       - “選 C”
#       - “第三个选项” / “第三个”
#       - “C.”
#       - “c)”
#       - etc.
#     如果无法确定，返回 None。
#     """
#     # 优先 letter A-D 的直接 match
#     m = re.search(r'\b([A-Da-d])\b', output_text)
#     if m:
#         return m.group(1).upper()
#
#     # 中文 "选项 X"
#     m = re.search(r'选项\s*([A-Da-d])', output_text)
#     if m:
#         return m.group(1).upper()
#     m = re.search(r'答案\s*[:：]?\s*([A-Da-d])', output_text)
#     if m:
#         return m.group(1).upper()
#
#     # 中文数字 -> 第三 / 第二 etc.
#     # 假设 mapping 一->A, 二->B, 三->C, 四->D
#     ch_map = {'一': 'A', '二': 'B', '三': 'C', '四': 'D', '1': 'A', '2': 'B', '3': 'C', '4': 'D'}
#     m = re.search(r'第([一二三四1234])', output_text)
#     if m:
#         cn = m.group(1)
#         return ch_map.get(cn)
#     m = re.search(r'([一二三四1234])\s*个选项', output_text)
#     if m:
#         cn = m.group(1)
#         return ch_map.get(cn)
#
#     # “选 A”
#     m = re.search(r'选\s*([A-Da-d])', output_text)
#     if m:
#         return m.group(1).upper()
#
#     # 括号形式 (C), C), [C] 等
#     m = re.search(r'[\(\[]\s*([A-Da-d])\s*[\)\]]', output_text)
#     if m:
#         return m.group(1).upper()
#
#     # 如果都失败
#     return None
#
#
# def evaluate(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     total = 0
#     correct = 0
#     no_parse = 0
#     for sample in data:
#         total += 1
#         llm_out = sample.get("llm_output", "")
#         gold = sample.get("correct_answer", "").strip().upper()
#         pred = parse_choice_from_output(llm_out)
#         if pred is None:
#             no_parse += 1
#             # 可视情况把这类样本 log 出来
#         else:
#             if pred == gold:
#                 correct += 1
#     accuracy = correct / total if total > 0 else 0.0
#     print(f"Total samples: {total}")
#     print(f"Correct: {correct}")
#     print(f"Parse failed (no decision): {no_parse}")
#     print(f"Accuracy (including parse fails as wrong): {accuracy:.4f}")
#     print(f"Accuracy (excluding parse fails): {correct / (total - no_parse) if total != no_parse else 0:.4f}")
#
#
# if __name__ == "__main__":
#     evaluate("log/mmlu_RM3.json")

