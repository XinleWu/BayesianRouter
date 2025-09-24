# 2任务学习，2路输入
import argparse
import os
import json
import re
import math
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict, Counter
from transformers import Trainer, TrainingArguments, Qwen2Model, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from offline_model_2encoder_100M2 import RewardDiffPredictor, get_tokenizer

Dir = "/data/cs.aau.dk/zh45qz/router_data/all_tulu/"
Dir2 = "/data/cs.aau.dk/zh45qz/router_data/rewardbenchV2/"
Dir3 = "/data/cs.aau.dk/zh45qz/router_data/helpsteer3/"
os.environ["WANDB_DISABLED"] = "true"


class MultiOutputTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 标准的 Trainer 会尝试自动处理 labels 和 logits，但你的是 tuple，所以我们自定义
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if has_labels:
                labels = inputs.pop("labels")
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
            else:
                labels = None
                outputs = model(**inputs)
                loss = None

        logits = outputs.logits  # 这里你返回的是 tuple (score_diffs, cls_logits, bt_scores)
        return (loss, logits, labels)


# def compute_combined_metrics(eval_preds):
#     print(">>>> compute_combined_metrics 被调用了！")
#     score_preds, cls_preds, bt_preds = eval_preds.predictions
#     score_labels, cls_labels, bt_array = eval_preds.label_ids  # 你在 data_collator 中设置了 labels 为 tuple
#
#     metrics = {}
#
#     # —— Part 1: BT metrics ——
#     if bt_preds is not None and bt_array is not None:
#         B, num_rms = bt_preds.shape
#         # total_pairs = 0
#         # correct_pairs = 0
#         # for i in range(B):
#         #     for w, l in bt_array[i]:
#         #         if w < 0 or l < 0:
#         #             break
#         #         total_pairs += 1
#         #         if bt_preds[i, w] > bt_preds[i, l]:
#         #             correct_pairs += 1
#         # metrics['bt_accuracy'] = correct_pairs / total_pairs if total_pairs > 0 else 0.0
#
#         router_pred = np.argmax(bt_preds, axis=1)
#         # router_pred = np.argmax(cls_preds, axis=1)
#         # 统计每个 RM 被选中的次数
#         pick_counts = Counter(router_pred.tolist())
#         for rm_idx in range(bt_preds.shape[1]):
#             metrics[f'router_selected_rm{rm_idx}'] = pick_counts.get(rm_idx, 0)
#
#         # def router_acc(preds, label_matrix):
#         #     correct = 0
#         #     for idx, p in enumerate(preds):
#         #         row = label_matrix[idx]
#         #         lp = row[p]
#         #         other = [row[j] for j in range(num_rms) if j != p]
#         #         lo = max(other) if other else 0
#         #         if lp == 1 or (lp == 0 and lo == 0):
#         #             correct += 1
#         #     return correct / len(preds)
#
#         def router_acc(preds, label_matrix):
#             correct = 0
#             count = 0  # 只有行中有至少一个1才计数
#             for idx, p in enumerate(preds):
#                 row = label_matrix[idx]
#                 if sum(row) == 0:
#                     # 如果这一行没有任何1，跳过不计数
#                     continue
#                 count += 1
#                 if row[p] == 1:
#                     correct += 1
#             return correct / count
#
#         metrics['router_accuracy'] = router_acc(router_pred, cls_labels)
#         for k in range(num_rms):
#             metrics[f'baseline_rm{k}'] = router_acc(
#                 np.full(B, k, dtype=int), cls_labels
#             )
#         rnd = np.random.randint(0, num_rms, size=B)
#         metrics['baseline_random'] = router_acc(rnd, cls_labels)
#
#     # # —— Part 2: Classification metrics ——
#     # if cls_preds is not None:
#     #     num_cls = cls_preds.shape[1]
#     #     for i in range(num_cls):
#     #         logits = cls_preds[:, i]
#     #         pred_bin = (torch.sigmoid(torch.from_numpy(logits)) > 0.5).int().numpy()
#     #         gold = cls_labels[:, i].astype(int)
#     #         metrics[f'cls_acc_rm{i}'] = accuracy_score(gold, pred_bin)
#     #         metrics[f'cls_prec_rm{i}'] = precision_score(gold, pred_bin, average='macro')
#     #         metrics[f'cls_recall_rm{i}'] = recall_score(gold, pred_bin, average='macro')
#     #         metrics[f'cls_f1_rm{i}'] = f1_score(gold, pred_bin, average='macro')
#
#     return metrics


def compute_combined_metrics(eval_preds):
    print(">>>> compute_combined_metrics 被调用了！")
    score_preds, cls_preds, bt_preds = eval_preds.predictions
    score_labels, cls_labels, bt_array = eval_preds.label_ids

    metrics = {}

    # —— Part 1: BT metrics ——
    if bt_preds is not None and bt_array is not None:
        B, num_rms = bt_preds.shape

        router_pred = np.argmax(bt_preds, axis=1)
        # 统计每个 RM 被选中的次数
        pick_counts = Counter(router_pred.tolist())
        for rm_idx in range(bt_preds.shape[1]):
            metrics[f'router_selected_rm{rm_idx}'] = pick_counts.get(rm_idx, 0)

        def router_acc(preds, label_matrix):
            correct = 0
            count = 0
            for idx, p in enumerate(preds):
                row = label_matrix[idx]
                if sum(row) == 0:
                    continue
                count += 1
                if row[p] == 1:
                    correct += 1
            return correct / count if count > 0 else 0

        metrics['router_accuracy'] = router_acc(router_pred, cls_labels)
        for k in range(num_rms):
            metrics[f'baseline_rm{k}'] = router_acc(
                np.full(B, k, dtype=int), cls_labels
            )
        rnd = np.random.randint(0, num_rms, size=B)
        metrics['baseline_random'] = router_acc(rnd, cls_labels)

        # 新增：多数投票baseline
        def majority_voting_acc(label_matrix):
            correct = 0
            count = 0
            for idx in range(label_matrix.shape[0]):
                row = label_matrix[idx]
                if sum(row) == 0:
                    continue
                count += 1

                num_ones = np.sum(row)
                num_zeros = len(row) - num_ones

                if num_ones > num_zeros:
                    # 多数为1，预测正确
                    correct += 1
                elif num_zeros > num_ones:
                    # 多数为0，预测错误（不增加correct）
                    pass
                else:
                    # 平局，随机决定（50%概率正确）
                    if np.random.rand() > 0.5:
                        correct += 1
            return correct / count if count > 0 else 0

        metrics['baseline_majority_voting'] = majority_voting_acc(cls_labels)

    return metrics


def data_collator(features):
    batch = {}

    # 1) 堆输入和三个任务的 labels
    tensor_keys = [k for k in features[0].keys()]  # 包含 score_diff_labels, cls_labels, bt_pairs
    for key in tensor_keys:
        if key == "bt_pairs":
            continue
        batch[key] = torch.stack([f[key] for f in features])

    # 2) bt_pairs -> bt_array
    raw_bt = [f["bt_pairs"] for f in features]
    B = len(raw_bt)
    M = max(len(p) for p in raw_bt)
    bt_array = torch.full((B, M, 2), -1, dtype=torch.long)
    for i, pairs in enumerate(raw_bt):
        for j, (w, l) in enumerate(pairs):
            bt_array[i, j, 0] = w
            bt_array[i, j, 1] = l

    # 3) **不要** 把它放到 batch["labels"]，而是改成 forward 要的三个参数：
    batch["score_diff_labels"] = batch.pop("score_diff_labels")
    batch["cls_labels"] = batch.pop("cls_labels")
    batch["bt_pairs"] = bt_array  # or 保持原 list-of-list，然后在 forward 再转

    batch["labels"] = (
        batch.pop("score_diff_labels"),
        batch.pop("cls_labels"),
        batch.pop("bt_pairs"),
    )

    return batch


def build_router_input(question_dialogue, answer):
    parts = []
    for msg in question_dialogue:
        if msg["role"] == "user":
            parts.append(f"<|user|>: {msg['content']}")
        elif msg["role"] == "assistant":
            parts.append(f"<|assistant|>: {msg['content']}")

    for msg in answer:
        parts.append(f"<|assistant|>: {msg['content']}")

    return "\n".join(parts)


class PreferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024, target_rm_indices=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_rm_indices = target_rm_indices
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        with open(data_path) as f:
            raw_data = json.load(f)
        return raw_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # # Instruct版本的输入
        # query = item['question']
        # r1 = item['chosen_answer']
        # r2 = item['rejected_answer']
        # input_text_1 = self.tokenizer.apply_chat_template(query + r1, tokenize=False)
        # input_text_2 = self.tokenizer.apply_chat_template(query + r2, tokenize=False)

        # 两路输入
        input_text_1 = build_router_input(item["question"], item["chosen_answer"])
        input_text_2 = build_router_input(item["question"], item["rejected_answer"])
        encoded_1 = self.tokenizer(
            input_text_1,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        encoded_2 = self.tokenizer(
            input_text_2,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # 使用索引列表获取对应位置的标签
        cls_labels = [item["int_labels"][i] for i in self.target_rm_indices]
        score_labels = [item["score_diffs"][i] for i in self.target_rm_indices]
        # cls_labels = [item["masked_int_labels"][i] for i in self.target_rm_indices]
        # score_labels = [item["masked_score_diffs"][i] for i in self.target_rm_indices]
        cls_labels = torch.tensor(cls_labels, dtype=torch.float32)
        score_labels = torch.tensor(score_labels, dtype=torch.float32)

        raw_bt_pairs = item.get("bt_pairs", [])
        global2local = {rm: idx for idx, rm in enumerate(self.target_rm_indices)}
        bt_pairs = []
        for w, l in raw_bt_pairs:
            if w in global2local and l in global2local:
                bt_pairs.append((global2local[w], global2local[l]))

        return {
            "input_ids_1": encoded_1["input_ids"].squeeze(0),
            "attention_mask_1": encoded_1["attention_mask"].squeeze(0),
            "input_ids_2": encoded_2["input_ids"].squeeze(0),
            "attention_mask_2": encoded_2["attention_mask"].squeeze(0),
            "score_diff_labels": score_labels,
            "cls_labels": cls_labels,
            "bt_pairs": bt_pairs
        }


def build_router_input_test(question_dialogue, answer):
    return f"<|user|>: {question_dialogue}\n<|assistant|>: {answer}"


class PreferenceDataset_ood(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024, target_rm_indices=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_rm_indices = target_rm_indices
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        with open(data_path) as f:
            raw_data = json.load(f)
        return raw_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_text_1 = build_router_input_test(item["question"], item["chosen_answer"])
        input_text_2 = build_router_input_test(item["question"], item["rejected_answer"])
        encoded_1 = self.tokenizer(
            input_text_1,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        encoded_2 = self.tokenizer(
            input_text_2,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        cls_labels = [item["int_labels"][i] for i in self.target_rm_indices]
        score_labels = [item["score_diffs"][i] for i in self.target_rm_indices]
        # cls_labels = [item["masked_int_labels"][i] for i in self.target_rm_indices]
        # score_labels = [item["masked_score_diffs"][i] for i in self.target_rm_indices]
        cls_labels = torch.tensor(cls_labels, dtype=torch.float32)
        score_labels = torch.tensor(score_labels, dtype=torch.float32)

        raw_bt_pairs = item.get("bt_pairs", [])
        global2local = {rm: idx for idx, rm in enumerate(self.target_rm_indices)}
        bt_pairs = []
        for w, l in raw_bt_pairs:
            if w in global2local and l in global2local:
                bt_pairs.append((global2local[w], global2local[l]))

        return {
            "input_ids_1": encoded_1["input_ids"].squeeze(0),
            "attention_mask_1": encoded_1["attention_mask"].squeeze(0),
            "input_ids_2": encoded_2["input_ids"].squeeze(0),
            "attention_mask_2": encoded_2["attention_mask"].squeeze(0),
            "score_diff_labels": score_labels,
            "cls_labels": cls_labels,
            "bt_pairs": bt_pairs
        }


def train(args):
    tokenizer = get_tokenizer(args.tokenizer_name)
    config = AutoConfig.from_pretrained(args.model_name)
    model = RewardDiffPredictor(config=config, num_reward_models=args.num_reward_models)
    model.load_qwen_backbone(model_path=config._name_or_path)  # 加载qwen的预训练参数

    total_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(f"Qwen 参数总数（元素个数）: {total_params:,}")
    # model.backbone.print_trainable_parameters()

    # print(">>> Trainable parameter names (requires_grad=True):")
    # for n, p in model.backbone.named_parameters():
    #     if p.requires_grad:
    #         print(n, p.shape)

    model.resize_token_embeddings(len(tokenizer))
    print("Tokenizer vocab size:", len(tokenizer))
    print("Model embed tokens size:", model.get_input_embeddings().num_embeddings)

    train_dataset = PreferenceDataset(args.train_data_path, tokenizer, target_rm_indices=[1, 3, 4, 7])
    val_dataset = PreferenceDataset_ood(args.val_data_path, tokenizer, target_rm_indices=[1, 3, 4, 7])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        # weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        adam_epsilon=args.adam_epsilon,
        save_steps=500,
        save_strategy="steps",
        save_total_limit=None,
        logging_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,  # 是否在训练结束时，用最佳 checkpoint 覆盖当前模型，设为false可以保存最后一个
        metric_for_best_model="router_accuracy",  # 根据加权
        greater_is_better=True,
        # fp16=True,
    )

    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     learning_rate=args.learning_rate,  # 这个可以保留（Trainer 里若我们手动注入优化器会被覆盖）
    #     per_device_train_batch_size=args.batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,  # <-- 新加
    #     num_train_epochs=args.epochs,
    #     adam_epsilon=args.adam_epsilon,
    #     save_steps=100,
    #     save_strategy="steps",
    #     save_total_limit=None,
    #     logging_steps=100,
    #     eval_strategy="steps",
    #     eval_steps=100,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="router_accuracy",
    #     greater_is_better=True,
    #     # fp16=True,
    #     weight_decay=args.weight_decay,  # 可选：也写在 TrainingArguments 中
    # )

    trainer = MultiOutputTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_combined_metrics,
        data_collator=data_collator,
    )

    # backbone_lr = args.backbone_lr
    # head_lr = args.head_lr
    # weight_decay = args.weight_decay
    #
    # # 1) 先收集所有 LayerNorm 模块的参数名，确保能准确排除 weight decay
    # layernorm_param_names = set()
    # for module_name, module in model.named_modules():
    #     if isinstance(module, nn.LayerNorm):
    #         # module.named_parameters(prefix=module_name) 会返回 e.g. ("shared_mlp.2.weight", param)
    #         for pn, _ in module.named_parameters(prefix=module_name):
    #             layernorm_param_names.add(pn)
    #
    # # 2) 按照是否属于 backbone（prefix "backbone") 来分组；并按 decay / no_decay 再细分
    # backbone_decay = []
    # backbone_nodecay = []
    # head_decay = []
    # head_nodecay = []
    #
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         continue
    #
    #     is_backbone = name.startswith("backbone")
    #     is_bias = name.endswith(".bias")
    #     is_layernorm = name in layernorm_param_names
    #
    #     if is_backbone:
    #         if is_bias or is_layernorm:
    #             backbone_nodecay.append(param)
    #         else:
    #             backbone_decay.append(param)
    #     else:
    #         # 非 backbone 都当作 head（包括 shared_mlp 等）
    #         if is_bias or is_layernorm:
    #             head_nodecay.append(param)
    #         else:
    #             head_decay.append(param)
    #
    # # 3) 组装 optimizer_grouped_parameters
    # optimizer_grouped_parameters = []
    # if backbone_decay:
    #     optimizer_grouped_parameters.append({
    #         "params": backbone_decay,
    #         "lr": backbone_lr,
    #         "weight_decay": weight_decay,
    #     })
    # if backbone_nodecay:
    #     optimizer_grouped_parameters.append({
    #         "params": backbone_nodecay,
    #         "lr": backbone_lr,
    #         "weight_decay": 0.0,
    #     })
    # if head_decay:
    #     optimizer_grouped_parameters.append({
    #         "params": head_decay,
    #         "lr": head_lr,
    #         "weight_decay": weight_decay,
    #     })
    # if head_nodecay:
    #     optimizer_grouped_parameters.append({
    #         "params": head_nodecay,
    #         "lr": head_lr,
    #         "weight_decay": 0.0,
    #     })
    #
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=args.adam_epsilon)
    #
    # # 4) scheduler（考虑 gradient_accumulation_steps）
    # num_update_steps_per_epoch = math.ceil(
    #     len(train_dataset) / (args.batch_size * max(1, args.gradient_accumulation_steps)))
    # num_training_steps = int(num_update_steps_per_epoch * args.epochs)
    # num_warmup_steps = int(args.warmup_ratio * num_training_steps) if num_training_steps > 0 else 0
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
    #                                             num_training_steps=num_training_steps)
    #
    # # 5) 注入到 Trainer（最小侵入）
    # trainer.optimizer = optimizer
    # trainer.lr_scheduler = scheduler


    trainer.train()
    trainer.save_model(args.output_dir)  # 保存最佳模型到输出目录
    tokenizer.save_pretrained(args.output_dir)  # 保存 tokenizer 配置（必须）


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--train_data_path", type=str, default=Dir3 + "train_helpsteer_RMBench_only1347.json")
    parser.add_argument("--val_data_path", type=str, default=Dir2 + "all_samples.json")
    parser.add_argument("--output_dir", type=str, default=Dir3 + "output14")

    # parser.add_argument("--backbone_lr", type=float, default=1e-5)
    # parser.add_argument("--head_lr", type=float, default=5e-5)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # parser.add_argument("--weight_decay", type=float, default=0.01)
    # parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1.0e-8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num_reward_models", type=int, default=4)
    args = parser.parse_args()

    train(args)


