import argparse
import os
import json
import math
from copy import deepcopy
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict, Counter
from transformers import Trainer, TrainingArguments, Qwen2Model, AutoConfig
from torch.utils.data import Dataset
import torch
from offline_model import RewardDiffPredictor, get_tokenizer

Dir = "/data/cs.aau.dk/zh45qz/router_data/all_tulu/"
Dir2 = "/data/cs.aau.dk/zh45qz/router_data/rewardbench_behaviour/"


# def compute_directional_metrics(eval_preds):
#     preds, labels = eval_preds
#     # preds[0]取回归头，preds[1]取分类头
#     direction_logits = preds[1] if isinstance(preds, tuple) and len(preds) > 1 else preds
#     labels = labels[1] if isinstance(labels, tuple) and len(labels) > 1 else labels
#
#     metrics = {}
#     # 对每个 RM 单独计算指标
#     k = direction_logits.shape[1]
#     for i in range(k):
#         logits_i = direction_logits[:, i]
#         labels_i = labels[:, i]
#         # pred_bin = (logits_i > 0).astype(int)  # 似乎等价于pred_bin = (torch.sigmoid(logits) > 0.5).int()
#         pred_bin = (torch.sigmoid(torch.from_numpy(logits_i)) > 0.5).int()
#         label_bin = labels_i.astype(int)
#         acc = accuracy_score(label_bin, pred_bin)
#         prec = precision_score(label_bin, pred_bin, average='macro')
#         rec = recall_score(label_bin, pred_bin, average='macro')
#         f1 = f1_score(label_bin, pred_bin, average='macro')
#         metrics[f"directional_accuracy_rm{i}"] = acc
#         metrics[f"directional_precision_rm{i}"] = prec
#         metrics[f"directional_recall_rm{i}"] = rec
#         metrics[f"directional_f1_rm{i}"] = f1
#
#     return metrics


def compute_combined_metrics(eval_preds):
    preds, labels = eval_preds
    # preds[0]取回归头，preds[1]取分类头
    direction_logits = preds[1] if isinstance(preds, tuple) and len(preds) > 1 else preds
    labels = labels[1] if isinstance(labels, tuple) and len(labels) > 1 else labels

    metrics = {}
    num_rms = direction_logits.shape[1]
    # === Part 1: Directional Metrics (per RM) ===
    for i in range(num_rms):
        logits_i = direction_logits[:, i]
        labels_i = labels[:, i].astype(int)
        pred_bin = (torch.sigmoid(torch.from_numpy(logits_i)) > 0.5).int()

        acc = accuracy_score(labels_i, pred_bin)
        prec = precision_score(labels_i, pred_bin, average='macro')
        rec = recall_score(labels_i, pred_bin, average='macro')
        f1 = f1_score(labels_i, pred_bin, average='macro')
        metrics[f"directional_accuracy_rm{i}"] = acc
        metrics[f"directional_precision_rm{i}"] = prec
        metrics[f"directional_recall_rm{i}"] = rec
        metrics[f"directional_f1_rm{i}"] = f1

    # === Part 2: Router vs. Baselines ===
    router_pred = np.argmax(direction_logits, axis=-1)  # shape: (N,)

    def compute_router_accuracy(preds, label_matrix):
        correct = []
        for i, p in enumerate(preds):
            row = label_matrix[i]
            lp = row[p]
            other_idxs = [j for j in range(len(row)) if j != p]
            lo = max(row[j] for j in other_idxs)

            if lp == 1 or (lp == 0 and lo == 0):
                correct.append(1)
            else:
                correct.append(0)
        return accuracy_score([1] * len(correct), correct)

    acc_router = compute_router_accuracy(router_pred, labels)
    acc_rm0 = compute_router_accuracy(np.zeros_like(router_pred), labels)
    acc_rm1 = compute_router_accuracy(np.ones_like(router_pred), labels)
    acc_rm2 = compute_router_accuracy(np.full_like(router_pred, 2), labels)
    acc_random = compute_router_accuracy(np.random.randint(0, 3, size=router_pred.shape), labels)

    print(f"[Eval] router_acc={acc_router:.4f}, rm0={acc_rm0:.4f}, rm1={acc_rm1:.4f}, rm2={acc_rm2:.4f}, random={acc_random:.4f}")

    metrics.update({
        'router_accuracy': acc_router,
        'baseline_rm0': acc_rm0,
        'baseline_rm1': acc_rm1,
        'baseline_rm2': acc_rm2,
        'baseline_random': acc_random,
    })

    return metrics


# class PreferenceDataset(Dataset):
#     def __init__(self, data_path, tokenizer, max_length=512):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.data = self.load_data(data_path)
#
#     def load_data(self, data_path):
#         with open(data_path) as f:
#             raw_data = json.load(f)
#
#         augmented_data = []
#         for item in raw_data:
#             # 原始顺序 (q, a, b)
#             augmented_data.append({
#                 "question": item["question"],
#                 "chosen_answer": item["chosen_answer"],
#                 "rejected_answer": item["rejected_answer"],
#                 "int_labels": item["int_labels"],
#                 "score_diffs": item["score_diffs"]
#             })
#             # 对偶顺序 (q, b, a)，标签保持不变
#             augmented_data.append({
#                 "question": item["question"],
#                 "chosen_answer": item["rejected_answer"],  # 对调
#                 "rejected_answer": item["chosen_answer"],
#                 "int_labels": item["int_labels"],
#                 "score_diffs": [-v for v in item["score_diffs"]]
#             })
#
#         return augmented_data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#
#         text = f"Question: {item['question']} Answer_A: {item['chosen_answer']} Answer_B: {item['rejected_answer']}"
#         encoded = self.tokenizer(
#             text,
#             truncation=True,
#             max_length=self.max_length,
#             padding="max_length",
#             return_tensors="pt"
#         )
#
#         # raw_score_diffs = item["score_diffs"]  # e.g. [sd1, sd2, sd3]
#         # # tanh 归一化
#         # # norm_score_diff = math.tanh(raw_score_diff)
#         # norm_score_diffs = raw_score_diffs
#         # cls_labels = item["int_labels"]
#
#         labels_int = item['int_labels']
#         labels_score = item["score_diffs"]
#         label0 = labels_int[3]
#         label1 = labels_int[7]
#         score0 = labels_score[3]
#         score1 = labels_score[7]
#         labels = torch.tensor([label0, label1], dtype=torch.float32)
#         score_diffs = torch.tensor([score0, score1], dtype=torch.float32)
#
#         return {
#             "input_ids": encoded["input_ids"].squeeze(0),
#             "attention_mask": encoded["attention_mask"].squeeze(0),
#             "score_diff_labels": score_diffs,
#             "cls_labels": labels,
#         }


class PreferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, target_rm_indices=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_rm_indices = target_rm_indices or [0, 1, 2]  # 默认前三个 RM
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        with open(data_path) as f:
            raw_data = json.load(f)

        augmented_data = []
        for item in raw_data:
            # 正向样本
            augmented_data.append({
                "question": item["question"],
                "chosen_answer": item["chosen_answer"],
                "rejected_answer": item["rejected_answer"],
                "int_labels": item["int_labels"],
                "score_diffs": item["score_diffs"]
            })
            # 对偶样本
            augmented_data.append({
                "question": item["question"],
                "chosen_answer": item["rejected_answer"],
                "rejected_answer": item["chosen_answer"],
                "int_labels": item["int_labels"],
                "score_diffs": [-v for v in item["score_diffs"]]
            })

        return augmented_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = f"Question: {item['question']} Answer_A: {item['chosen_answer']} Answer_B: {item['rejected_answer']}"
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # 使用索引列表获取对应位置的标签
        cls_labels = [item["int_labels"][i] for i in self.target_rm_indices]
        score_labels = [item["score_diffs"][i] for i in self.target_rm_indices]
        cls_labels = torch.tensor(cls_labels, dtype=torch.float32)
        score_labels = torch.tensor(score_labels, dtype=torch.float32)

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "score_diff_labels": score_labels,
            "cls_labels": cls_labels,
        }


def train(args):
    tokenizer = get_tokenizer(args.tokenizer_name)
    config = AutoConfig.from_pretrained(args.model_name)
    model = RewardDiffPredictor(config=config, num_reward_models=args.num_reward_models)
    model.load_qwen_backbone(model_path=config._name_or_path)  # 加载qwen的预训练参数

    total_params = sum(p.numel() for p in model.qwen.parameters())
    print(f"Qwen 参数总数（元素个数）: {total_params:,}")

    model.resize_token_embeddings(len(tokenizer))
    print("Tokenizer vocab size:", len(tokenizer))
    print("Model embed tokens size:", model.get_input_embeddings().num_embeddings)

    train_dataset = PreferenceDataset(args.train_data_path, tokenizer, target_rm_indices=[0, 1, 2])
    val_dataset = PreferenceDataset(args.val_data_path, tokenizer, target_rm_indices=[0, 1, 2])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        adam_epsilon=args.adam_epsilon,
        save_steps=10000,
        save_strategy="steps",
        save_total_limit=1,
        logging_steps=10000,
        eval_strategy="steps",
        eval_steps=10000,
        load_best_model_at_end=True,  # 是否在训练结束时，用最佳 checkpoint 覆盖当前模型，设为false可以保存最后一个
        metric_for_best_model="eval_loss",  # 根据加权
        greater_is_better=False,
        # fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_combined_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)  # 保存最佳模型到输出目录
    tokenizer.save_pretrained(args.output_dir)  # 保存 tokenizer 配置（必须）


if __name__ == "__main__":
    # 不搞长度限制，让数据变多，看看效果；
    # 要不要把mean pooling改成cls token输出？
    # "可显式标记答案对（如添加 [A]/[B] 标识符），帮助模型定位关键部分。" 意思是用加入特殊token标记两个answer的起始点？比如用[SEP]区分；
    # query embedding也要做归一化吗？
    # 我为啥不考虑PPO？
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")  # 换成3B试试？
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train_data_path", type=str, default=Dir + "train_balance.json")  # chosen不一定在前的数据集
    parser.add_argument("--val_data_path", type=str, default=Dir + "test_balance.json")
    parser.add_argument("--output_dir", type=str, default=Dir + "out_final_15B")  # 要修改输出文件；
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=4.0e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1.0e-8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_reward_models", type=int, default=3)
    args = parser.parse_args()

    # train(args)


    # with open(os.path.join(Dir, "all_samples.json"), "r", encoding="utf-8") as f:
    #     full_samples = json.load(f)
    # gemma_without_multi_round = []
    # for example in full_samples:
    #     if example['tag'] == 'good sample':
    #         gemma_without_multi_round.append(example)
    #
    # gemma_length_512 = []
    # for sample in gemma_without_multi_round:
    #     if len(sample['question'].split()) + len(sample['chosen_answer'].split()) + len(sample['rejected_answer'].split()) < 350:
    #         gemma_length_512.append(sample)
    # print(len(gemma_length_512))
    #
    # gemma_pos = []
    # gemma_neg = []
    # for example in gemma_length_512:
    #     if -10<example["score_diffs"][0]<-0.2:
    #         gemma_neg.append(example)
    #     elif 0.2<example["score_diffs"][0]<10:
    #         gemma_pos.append(example)
    # print(len(gemma_neg))
    # print(len(gemma_pos))
    # gemma_neg.extend(gemma_pos)
    # print(len(gemma_neg))
    #
    # pos1 = []
    # neg1 = []
    # for example in gemma_neg:
    #     if -10<example["score_diffs"][1]<-0.2:
    #         neg1.append(example)
    #     elif 0.2<example["score_diffs"][1]<10:
    #         pos1.append(example)
    # print(len(pos1))
    # print(len(neg1))
    # neg1.extend(pos1)
    # print(len(neg1))
    #
    # pos2 = []
    # neg2 = []
    # for example in neg1:
    #     if -10 < example["score_diffs"][2] < -0.2:
    #         neg2.append(example)
    #     elif 0.2 < example["score_diffs"][2] < 10:
    #         pos2.append(example)
    # print(len(pos2))
    # print(len(neg2))
    # neg2.extend(pos2)
    # print(len(neg2))
    #
    # random.shuffle(neg2)
    # train_data = neg2[:int(len(neg2) * 0.9)]
    # test_data = neg2[int(len(neg2) * 0.9):]
    # print(len(train_data))
    # print(len(test_data))
    # with open(Dir + "train_0.2.json", "w") as f:
    #     json.dump(train_data, f, indent=2)
    # with open(Dir + "test_0.2.json", "w") as f:
    #     json.dump(test_data, f, indent=2)



    with open(os.path.join(Dir, "filtered_samples.json"), "r", encoding="utf-8") as f:
        full_samples = json.load(f)
    samples_without_multi_round = []
    for example in full_samples:
        if example['tag'] == 'good sample':
            samples_without_multi_round.append(example)
    print(len(samples_without_multi_round))

    # 使用元组(question, chosen_answer, rejected_answer)作为唯一标识
    unique_samples_dict = {}
    for sample in samples_without_multi_round:
        key = (sample['question'], sample['chosen_answer'], sample['rejected_answer'])
        if key not in unique_samples_dict:
            unique_samples_dict[key] = sample
    # 获取去重后的样本列表
    unique_samples = list(unique_samples_dict.values())
    print(len(unique_samples))

    unique_samples_8B = []  # 看起来8B的结果也就那样，没有排行榜那么惊艳；
    for sample in unique_samples:
        if sample['int_labels'][-3] == 1 or sample['int_labels'][-2] == 1 or sample['int_labels'][-1] == 1:
            unique_samples_8B.append(sample)
    print(len(unique_samples_8B))

    unique_query_dict = {}
    for sample in unique_samples_8B:
        key = sample['question']
        if key not in unique_query_dict:
            unique_query_dict[key] = sample
    unique_queries = list(unique_query_dict.values())
    print(len(unique_queries))  # 重复query只保留一个sample


    def balance_per_rm(data, rm_indices, seed=42):
        random.seed(seed)
        rm_to_indices = defaultdict(set)

        for rm_idx in rm_indices:
            pos_indices = [i for i, item in enumerate(data) if item["int_labels"][rm_idx] == 1]
            neg_indices = [i for i, item in enumerate(data) if item["int_labels"][rm_idx] == 0]
            print(len(pos_indices), len(neg_indices))
            n = min(len(pos_indices), len(neg_indices))

            pos_selected = random.sample(pos_indices, n)
            neg_selected = random.sample(neg_indices, n)

            selected = pos_selected + neg_selected
            rm_to_indices[rm_idx] = set(selected)

        return rm_to_indices


    def get_label_mask_for_item(item, item_idx, rm_indices, rm_to_indices):
        new_item = deepcopy(item)
        new_int_labels = []
        new_score_diffs = []

        for i in range(len(item["int_labels"])):
            if i in rm_indices and item_idx in rm_to_indices[i]:
                new_int_labels.append(item["int_labels"][i])
                new_score_diffs.append(item["score_diffs"][i])
            else:
                new_int_labels.append(-1000)
                new_score_diffs.append(-1000.0)

        new_item["int_labels"] = new_int_labels
        new_item["score_diffs"] = new_score_diffs
        return new_item

    rm_indices = [2, 3, 4]
    rm_to_indices = balance_per_rm(unique_queries, rm_indices)
    # 合并所有 RM 的子集
    all_selected_indices = set()
    for s in rm_to_indices.values():
        all_selected_indices.update(s)
    all_selected_indices = sorted(list(all_selected_indices))

    balanced_data = []
    for idx in all_selected_indices:
        masked_item = get_label_mask_for_item(unique_queries[idx], idx, rm_indices, rm_to_indices)
        balanced_data.append(masked_item)
    print(f"Saved balanced + masked dataset with {len(balanced_data)} samples")
    for rm_idx in rm_indices:
        print(f"  RM{rm_idx}: {len(rm_to_indices[rm_idx])} samples (balanced)")
    random.shuffle(balanced_data)
    train_data = balanced_data[:int(len(balanced_data) * 0.9)]
    test_data = balanced_data[int(len(balanced_data) * 0.9):]
    print(len(train_data))
    print(len(test_data))

    id_to_sample = {}
    for sample in unique_samples:
        key = sample['id']
        id_to_sample[key] = sample
    new_test_data = []
    for sample in test_data:  # 给测试集恢复全标签
        id = sample['id']
        new_test_data.append(id_to_sample[id])
    print(len(new_test_data))

    with open(Dir + "train_balance.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open(Dir + "test_balance.json", "w") as f:
        json.dump(new_test_data, f, indent=2)








    # float_labels = [s["score_diff"] for s in gemma_length_512]
    # bins = [float('-inf'), -20, -10, -5, -1, -0.2, 0, 0.2, 1, 5, 10, 20, float('inf')]
    # bin_names = ["-20–", "-20~-10", "-10~-5", "-5~-1", "-1~-0.2", "-0.2~0", "0~0.2", "0.2~1", "1~5", "5~10", "10~20", "20+"]
    # length_counter = Counter()
    #
    # for label in float_labels:
    #     for i, b in enumerate(bins[:-1]):
    #         if b <= label < bins[i+1]:
    #             length_counter[bin_names[i]] += 1
    #             break
    # for bin_name in bin_names:
    #     print(f"{bin_name}: {length_counter[bin_name]}")
