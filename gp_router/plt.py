import random

import matplotlib.pyplot as plt
import numpy as np

# # 数据
# categories = ['Creative', 'Analysis', 'Coding', 'Fact', 'Math']
# RM0 = [19, 23, 17, 20, 19]
# RM1 = [22, 12, 25, 27, 25]
# RM2 = [17, 11, 13, 7, 9]
# RM3 = [25, 32, 35, 30, 31]
#
# # 设置柱状图参数
# x = np.arange(len(categories))  # x轴位置
# width = 0.2  # 每个柱子的宽度
#
# # 绘图
# fig, ax = plt.subplots(figsize=(8, 5))
#
# bars1 = ax.bar(x - 1.5*width, RM0, width, label='RM0', color='orange')
# bars2 = ax.bar(x - 0.5*width, RM1, width, label='RM1', color='brown')
# bars3 = ax.bar(x + 0.5*width, RM2, width, label='RM2', color='pink')
# bars4 = ax.bar(x + 1.5*width, RM3, width, label='RM3', color='blue')
#
# # 添加标签和图例
# ax.set_ylabel('Utilization rate (%)')
# ax.set_xticks(x)
# ax.set_xticklabels(categories)
# ax.legend()
#
# plt.tight_layout()
# plt.show()


import json
import torch
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.2-3B"
BATCH_SIZE = 32  # 可以根据显存调

# -------- 加载数据 --------
before_data = []
after_data = []
with open("output/iter3_outputs_before_onlyOnline.json", "r", encoding="utf-8") as f:
    # before_data = json.load(f)
    for line in f:
        before_data.append(json.loads(line))
    # lines = f.readlines()
    # before_data = json.loads(lines[-2])
    # print(len(before_data))
with open("output_full/onlyOnline2.json", "r", encoding="utf-8") as f:
    # after_data = json.load(f)
    for line in f:
        after_data.append(json.loads(line))
    # lines = f.readlines()
    # after_data = json.loads(lines[-2])

before_data_shuffled, after_data_shuffled = shuffle(before_data, after_data, random_state=42)

for i in range(180, 190):
    print(before_data[i])
    print(after_data[i])
    print('='*10)
print(len(before_data))
print(len(after_data))
# after_data = after_data[:800]



# batch size减小好像是好事儿？16好像就差不多了；
# baseline_alpha设为0.001试试？不行
# _arm_min_count增大到100试试？不行
# lambda增大到100试试；和50差不多？





assert len(before_data) == len(after_data), "两个文件长度不一致！"

# -------- 加载模型 --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# -------- 计算分数 --------
def score_batch(samples):
    texts = [s["instruction"] + "\n" + s["output"] for s in samples]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1).cpu().tolist()
    return scores

def add_scores(data):
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]
        batch_scores = score_batch(batch)
        for sample, score in zip(batch, batch_scores):
            sample["score"] = score

add_scores(before_data)
add_scores(after_data)

# -------- 统计 win rate --------
equal_count = sum(a["score"] == b["score"] for a, b in zip(after_data, before_data))
win_count = sum(a["score"] > b["score"] for a, b in zip(after_data, before_data))
print(equal_count)
win_rate = win_count / (len(before_data)-equal_count)
print(win_count)
print(equal_count)
print(f"After win rate: {win_rate:.4f}")




# # -------- 可选：保存带 score 的文件 --------
# with open("before_scored.json", "w", encoding="utf-8") as f:
#     json.dump(before_data, f, ensure_ascii=False, indent=2)
# with open("after_scored.json", "w", encoding="utf-8") as f:
#     json.dump(after_data, f, ensure_ascii=False, indent=2)









# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Patch
#
# # 模型和任务
# models = ['Llama-3-8B', 'Mistral-7B']
# tasks = ['LASeR vs. Best RM', 'LASeR vs. Seq. RM Selection', 'LASeR vs. Random RM Selection']
#
# # 数据：laser_wins 和 baseline_wins 分别为每个模型在每个任务下的胜率
# laser_wins = np.array([
#     [56.34, 71.45, 78.33],  # Llama-3-8B
#     [58.72, 63.72, 70.61]  # Mistral-7B
# ])
# baseline_wins = 100 - laser_wins
#
# # 图形参数
# bar_height = 0.35
# bar_spacing = 0.5
# group_width = 2 * bar_height + bar_spacing
# num_tasks = len(tasks)
#
# fig, ax = plt.subplots(figsize=(4, 2.8))
#
# # y 位置计算（3组任务，每组2个模型条）
# for task_idx in range(num_tasks):
#     for model_idx in range(len(models)):
#         y = task_idx * group_width + model_idx * bar_height
#         lw = laser_wins[model_idx, task_idx]
#         bw = baseline_wins[model_idx, task_idx]
#
#         # 绿色条（LASeR Wins）
#         ax.barh(y, lw, height=bar_height, color='lightgreen', edgecolor='black')
#         # 红色条（Baseline Wins）
#         ax.barh(y, bw, height=bar_height, left=lw, color='lightcoral', edgecolor='black')
#
#         # 百分比文本
#         ax.text(lw / 2, y, f'{lw:.2f}%', va='center', ha='center', weight='bold')
#         ax.text(lw + bw / 2, y, f'{bw:.2f}%', va='center', ha='center')
#
# # 设置Y轴
# yticks = []
# ytick_labels = []
# for task_idx in range(num_tasks):
#     for model_idx in range(len(models)):
#         yticks.append(task_idx * group_width + model_idx * bar_height)
#         ytick_labels.append(models[model_idx])
# ax.set_yticks(yticks)
# ax.set_yticklabels(ytick_labels, fontsize=10)
#
# # 设置X轴
# ax.set_xlim(0, 100)
# ax.set_xticks([])
#
# # # 设置任务标签（每组下方居中）
# # for task_idx, task_label in enumerate(tasks):
# #     y_center = task_idx * group_width + bar_height
# #     ax.text(50, y_center + bar_height - 1.0, task_label, ha='center', va='bottom', fontsize=10)
#
# # 图例
# legend_elements = [
#     Patch(facecolor='lightgreen', edgecolor='black', label='LASeR Wins'),
#     Patch(facecolor='lightcoral', edgecolor='black', label='Baseline Wins')
# ]
# ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)
#
# # 美化
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.tick_params(left=False, bottom=False)
# ax.set_xlabel('')
# ax.set_yticks(yticks)
# ax.set_yticklabels(ytick_labels)
#
# plt.tight_layout()
# plt.show()





