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
from transformers import Trainer, TrainingArguments, Qwen2Model, AutoConfig
from torch.utils.data import Dataset
import torch
from offline_model_2encoder_100M import RewardDiffPredictor, get_tokenizer

Dir = "/data/cs.aau.dk/zh45qz/router_data/all_tulu/"
Dir2 = "/data/cs.aau.dk/zh45qz/router_data/rewardbenchV2/"
Dir3 = "/data/cs.aau.dk/zh45qz/router_data/helpsteer3/"
os.environ["WANDB_DISABLED"] = "true"

EXPECTED_SUBSETS = ["Factuality", "Precise IF", "Math", "Safety", "Focus"]

def _canonicalize_subset_name(s: str):
    """把名字规范化为小写且只保留字母数字，便于匹配 'Precise_IF' / 'Precise IF' 等变体。"""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'[^0-9a-z]', '', s)  # keep only lowercase alnum
    return s

# canonical -> id
SUBSET_NAME_TO_ID = {_canonicalize_subset_name(name): i for i, name in enumerate(EXPECTED_SUBSETS)}
# id -> original display name
ID_TO_SUBSET_NAME = {i: name for i, name in enumerate(EXPECTED_SUBSETS)}


def extract_subset_from_pair_id(pair_id: str):
    """
    更稳健地从 pair_id 提取前缀。例如：
      "Precise IF__514-rejected_1" -> "Precise IF"
      "Precise_IF__514-..."      -> "Precise IF"
      "Factuality-123_..."       -> "Factuality"
    不再用 re.split(...) 只取第一个 token，而优先按 "__" 切分；没有 "__" 时
    尝试保留前面可能的整段前缀（去除多余后缀）。
    """
    if not isinstance(pair_id, str) or len(pair_id) == 0:
        return ""
    s = pair_id.strip()
    # 首先以双下划线为界（你数据里多数是这种形式）
    if "__" in s:
        prefix = s.split("__", 1)[0]
    else:
        # 没有 __ 时：尝试取到第一个 '-' 或到末尾（保留空格、下划线，稍后规范化）
        m = re.match(r'^([^\-\|]+)', s)  # 取到第一个 '-'（或其他分隔符）之前的全部
        prefix = m.group(1) if m else s
    # 规范化：把下划线换成空格，去首尾空白
    prefix = prefix.replace('_', ' ').strip()
    return prefix


def map_subset_to_id(raw_subset_name: str):
    """
    将 raw_subset_name 映射到 0..4 的 id（对应 EXPECTED_SUBSETS）。
    规则：
      1) 先 canonicalize（小写 + 去非字母数字）
      2) 精确匹配 canonical key
      3) 如果未命中，尝试 fuzzy：如果 canonical 中包含 expected key 或反向包含，则命中
      4) 仍未命中时返回 -1
    """
    if raw_subset_name is None:
        return -1
    key = _canonicalize_subset_name(raw_subset_name.replace('_', ' ').strip())
    if key in SUBSET_NAME_TO_ID:
        return SUBSET_NAME_TO_ID[key]
    # fuzzy match: try substring matches
    for exp_key, exp_id in SUBSET_NAME_TO_ID.items():
        if exp_key in key or key in exp_key:
            return exp_id
    # nothing matched
    return -1


# class MultiOutputTrainer(Trainer):
#     def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
#         # 标准的 Trainer 会尝试自动处理 labels 和 logits，但你的是 tuple，所以我们自定义
#         has_labels = "labels" in inputs
#         inputs = self._prepare_inputs(inputs)
#
#         with torch.no_grad():
#             if has_labels:
#                 labels = inputs.pop("labels")
#                 outputs = model(**inputs, labels=labels)
#                 loss = outputs.loss
#             else:
#                 labels = None
#                 outputs = model(**inputs)
#                 loss = None
#
#         logits = outputs.logits  # 这里你返回的是 tuple (score_diffs, cls_logits, bt_scores)
#         return (loss, logits, labels)


class MultiOutputTrainer(Trainer):
    def _pop_meta_from_mapping(self, mapping):
        """
        尝试从 mapping-like 对象中删除 meta keys（subset_ids, subset_names, pair_ids）。
        就地修改并返回。
        """
        try:
            if mapping is None:
                return mapping
            if hasattr(mapping, "pop"):
                mapping.pop("subset_ids", None)
                mapping.pop("subset_name", None)
                mapping.pop("subset_names", None)
                mapping.pop("pair_id", None)
                mapping.pop("pair_ids", None)
        except Exception:
            try:
                if "subset_ids" in mapping:
                    del mapping["subset_ids"]
                if "pair_id" in mapping:
                    del mapping["pair_id"]
            except Exception:
                pass
        return mapping

    def training_step(self, model, inputs, *args, **kwargs):
        """
        在训练时也移除 subset_ids 等 meta，防止被传入 model.forward。
        使用可变参数以兼容不同 transformers 版本。
        """
        self._pop_meta_from_mapping(inputs)
        return super().training_step(model, inputs, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        评估时：弹出 subset_ids（整型 tensor）但不要把它放回 labels。
        改为把 subset_ids 作为第四个元素附到 logits 上 (predictions)，
        这样 compute_metrics 能可靠地从 eval_preds.predictions 中读取它。
        """
        # 尽早弹出 meta（subset_ids 是 tensor）
        meta_subset_ids = None
        if isinstance(inputs, dict):
            meta_subset_ids = inputs.pop("subset_ids", None)
            # 也弹出字符串调试字段，避免不必要的传递
            inputs.pop("subset_names", None)
            inputs.pop("pair_ids", None)

        try:
            prepared_inputs = self._prepare_inputs(inputs)
        except Exception:
            prepared_inputs = inputs

        # 再次确保 prepared_inputs 中没有 meta
        if isinstance(prepared_inputs, dict) or hasattr(prepared_inputs, "pop"):
            try:
                prepared_inputs.pop("subset_ids", None)
                prepared_inputs.pop("subset_names", None)
                prepared_inputs.pop("pair_ids", None)
            except Exception:
                pass

        if ignore_keys is not None and isinstance(prepared_inputs, dict):
            for k in ignore_keys:
                prepared_inputs.pop(k, None)

        with torch.no_grad():
            if "labels" in prepared_inputs:
                labels = prepared_inputs.pop("labels")
                outputs = model(**prepared_inputs, labels=labels)
                loss = outputs.loss
            else:
                labels = None
                outputs = model(**prepared_inputs)
                loss = None

        logits = outputs.logits  # 期望是 tuple (score_diffs, cls_logits, bt_scores)

        # —— 这里把 meta 附加到 logits（predictions）上 ——
        if meta_subset_ids is not None:
            # 确保是 tensor
            if not torch.is_tensor(meta_subset_ids):
                try:
                    meta_subset_ids = torch.tensor(meta_subset_ids, dtype=torch.long)
                except Exception:
                    meta_subset_ids = torch.tensor(np.array(meta_subset_ids), dtype=torch.long)
            # 把 meta 移到与模型输出相同的 device（通常 logits[0] 在 model device）
            device = None
            if isinstance(logits, (tuple, list)) and len(logits) > 0 and torch.is_tensor(logits[0]):
                device = logits[0].device
            else:
                try:
                    device = next(model.parameters()).device
                except Exception:
                    device = torch.device("cpu")
            meta_subset_ids = meta_subset_ids.to(device)

            # 如果是标量 -> 扩成一维
            if meta_subset_ids.dim() == 0:
                meta_subset_ids = meta_subset_ids.unsqueeze(0)

            # 确保 logits 是 tuple，然后附加 meta
            if isinstance(logits, tuple):
                logits = tuple(list(logits) + [meta_subset_ids])
            else:
                logits = (logits, meta_subset_ids)

        # 返回时 labels 保持原样（不附 meta）
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
    preds = eval_preds.predictions
    # preds 可能是 tuple/list of arrays/tensors
    subset_arr = None

    # helper: convert tensor/array to numpy
    def to_numpy(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return np.array(x)

    # 首先尝试从 predictions 里读取第4项（我们在 prediction_step 中把 subset_ids 放到这里）
    if isinstance(preds, (list, tuple)) and len(preds) >= 4:
        # 可能是 (score_preds, cls_preds, bt_preds, subset_ids)
        score_preds = to_numpy(preds[0])
        cls_preds = to_numpy(preds[1])
        bt_preds = to_numpy(preds[2])
        subset_arr = to_numpy(preds[3])
    else:
        # 退回到老路径：predictions 应该包含 3 项
        score_preds = to_numpy(preds[0])
        cls_preds = to_numpy(preds[1])
        bt_preds = to_numpy(preds[2])

    # 然后拿 label_ids（前三项），不再指望第四项在 label_ids
    label_ids = eval_preds.label_ids
    if isinstance(label_ids, (list, tuple)) and len(label_ids) >= 3:
        score_labels = to_numpy(label_ids[0])
        cls_labels = to_numpy(label_ids[1])
        bt_array = to_numpy(label_ids[2])
    else:
        # 如果 label_ids 不是 tuple （极不常见），尝试直接转
        # 保持向后兼容：若无法解析，抛出清晰错误提示
        try:
            score_labels, cls_labels, bt_array = label_ids
            score_labels = to_numpy(score_labels)
            cls_labels = to_numpy(cls_labels)
            bt_array = to_numpy(bt_array)
        except Exception as e:
            raise RuntimeError(f"Unexpected label_ids format in compute_combined_metrics: {type(label_ids)}; value={label_ids}") from e

    metrics = {}

    # —— Part 1: BT metrics ——
    if bt_preds is not None and bt_array is not None:
        B, num_rms = bt_preds.shape
        router_pred = np.argmax(bt_preds, axis=1)
        pick_counts = Counter(router_pred.tolist())
        for rm_idx in range(num_rms):
            metrics[f'router_selected_rm{rm_idx}'] = pick_counts.get(rm_idx, 0)

        def router_acc(preds, label_matrix):
            correct = 0
            count = 0
            for idx, p in enumerate(preds):
                row = np.array(label_matrix[idx])
                if row.sum() == 0:
                    continue
                count += 1
                if p < 0 or p >= row.shape[0]:
                    continue
                if int(row[p]) == 1:
                    correct += 1
            return (correct / count) if count > 0 else float('nan')

        metrics['router_accuracy'] = router_acc(router_pred, cls_labels)
        for k in range(num_rms):
            metrics[f'baseline_rm{k}'] = router_acc(np.full(B, k, dtype=int), cls_labels)
        rnd = np.random.randint(0, num_rms, size=B)
        metrics['baseline_random'] = router_acc(rnd, cls_labels)

        # —— per expected subset —— （强制按 EXPECTED_SUBSETS 顺序输出）
        for sid in range(len(EXPECTED_SUBSETS)):
            subset_name = ID_TO_SUBSET_NAME[sid]
            safe_name = re.sub(r'[^0-9a-zA-Z]+', '_', subset_name)
            key_prefix = f"subset_{safe_name}"

            # 找出该 subset 的样本索引（subset_arr 中使用的是 id 或 -1）
            if subset_arr is None:
                idxs = np.array([], dtype=int)
            else:
                arr = np.array(subset_arr).reshape(-1)
                idxs = np.where(arr == sid)[0]

            if idxs.size == 0:
                # 无样本 -> 返回 nan 保证结果条目存在
                metrics[f'{key_prefix}_router_accuracy'] = float('nan')
                for k in range(num_rms):
                    metrics[f'{key_prefix}_baseline_rm{k}'] = float('nan')
                metrics[f'{key_prefix}_baseline_random'] = float('nan')
            else:
                metrics[f'{key_prefix}_router_accuracy'] = router_acc(router_pred[idxs], cls_labels[idxs])
                for k in range(num_rms):
                    metrics[f'{key_prefix}_baseline_rm{k}'] = router_acc(np.full(len(idxs), k, dtype=int), cls_labels[idxs])
                rnd_sub = np.random.randint(0, num_rms, size=len(idxs))
                metrics[f'{key_prefix}_baseline_random'] = router_acc(rnd_sub, cls_labels[idxs])


    return metrics



# def data_collator(features):
#     batch = {}
#
#     # 1) 堆输入和三个任务的 labels
#     tensor_keys = [k for k in features[0].keys()]  # 包含 score_diff_labels, cls_labels, bt_pairs
#     for key in tensor_keys:
#         if key == "bt_pairs":
#             continue
#         batch[key] = torch.stack([f[key] for f in features])
#
#     # 2) bt_pairs -> bt_array
#     raw_bt = [f["bt_pairs"] for f in features]
#     B = len(raw_bt)
#     M = max(len(p) for p in raw_bt)
#     bt_array = torch.full((B, M, 2), -1, dtype=torch.long)
#     for i, pairs in enumerate(raw_bt):
#         for j, (w, l) in enumerate(pairs):
#             bt_array[i, j, 0] = w
#             bt_array[i, j, 1] = l
#
#     # 3) **不要** 把它放到 batch["labels"]，而是改成 forward 要的三个参数：
#     batch["score_diff_labels"] = batch.pop("score_diff_labels")
#     batch["cls_labels"] = batch.pop("cls_labels")
#     batch["bt_pairs"] = bt_array  # or 保持原 list-of-list，然后在 forward 再转
#
#     batch["labels"] = (
#         batch.pop("score_diff_labels"),
#         batch.pop("cls_labels"),
#         batch.pop("bt_pairs"),
#     )
#
#     return batch


def data_collator(features):
    batch = {}

    # 1) 堆输入（跳过某些非-tensor 字段）
    tensor_keys = [k for k in features[0].keys()]
    skip_keys = {"bt_pairs", "subset_name", "pair_id", "subset_id"}
    for key in tensor_keys:
        if key in skip_keys:
            continue
        batch[key] = torch.stack([f[key] for f in features])

    # 2) bt_pairs -> bt_array (和你原来的逻辑)
    raw_bt = [f["bt_pairs"] for f in features]
    B = len(raw_bt)
    M = max((len(p) for p in raw_bt), default=0)
    bt_array = torch.full((B, M, 2), -1, dtype=torch.long)
    for i, pairs in enumerate(raw_bt):
        for j, (w, l) in enumerate(pairs):
            bt_array[i, j, 0] = w
            bt_array[i, j, 1] = l

    # 3) collect subset_ids（整型 tensor），以及 pair_ids/subset_name（如果你想保留）
    batch["subset_ids"] = torch.tensor(
        [f.get("subset_id", -1) for f in features],
        dtype=torch.long
    )
    # 可选保留调试 info（不作为 tensor）
    batch["pair_ids"] = [f.get("pair_id", "") for f in features]
    batch["subset_names"] = [f.get("subset_name", "") for f in features]

    # 4) labels（仍然只包含模型需要的三元组）
    batch["score_diff_labels"] = batch.pop("score_diff_labels")
    batch["cls_labels"] = batch.pop("cls_labels")
    batch["bt_pairs"] = bt_array

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
        self.target_rm_indices = target_rm_indices  # or [0, 1, 2]  # 默认前三个 RM
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        with open(data_path) as f:
            raw_data = json.load(f)

        # 对每个 RM 分别收集其 score_diff（非 -1000）用于计算均值和方差
        score_stats = {i: [] for i in self.target_rm_indices}
        for item in raw_data:
            for i in self.target_rm_indices:
                score = item["score_diffs"][i]
                if score != -1000.0:
                    score_stats[i].append(score)

        # 计算每个 RM 的均值和标准差
        rm_mean_std = {}
        for i in self.target_rm_indices:
            scores = np.array(score_stats[i])
            mean = scores.mean()
            std = scores.std() if scores.std() > 1e-6 else 1.0  # 防止除以0
            rm_mean_std[i] = (mean, std)

        # 对所有样本的 score_diff 进行归一化
        for item in raw_data:
            for i in self.target_rm_indices:
                score = item["score_diffs"][i]
                if score != -1000.0:
                    mean, std = rm_mean_std[i]
                    item["score_diffs"][i] = (score - mean) / std
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

        pair_id = item.get("pair_id", "")
        subset_name = extract_subset_from_pair_id(pair_id)  # e.g. "Focus" or "Precise_IF"
        subset_key = _canonicalize_subset_name(subset_name)  # e.g. "focus" or "preciseif"
        # 如果未识别则返回 -1；compute_combined_metrics 只对 EXPECTED_SUBSETS (0..4) 输出
        # subset_id = SUBSET_NAME_TO_ID.get(subset_key, -1)
        subset_id = map_subset_to_id(subset_name)
        return {
            "input_ids_1": encoded_1["input_ids"].squeeze(0),
            "attention_mask_1": encoded_1["attention_mask"].squeeze(0),
            "input_ids_2": encoded_2["input_ids"].squeeze(0),
            "attention_mask_2": encoded_2["attention_mask"].squeeze(0),
            "score_diff_labels": score_labels,
            "cls_labels": cls_labels,
            "bt_pairs": bt_pairs,
            "pair_id": pair_id,  # 可选：保留调试用
            "subset_name": subset_name,  # 可选：保留调试用
            "subset_id": subset_id,  # **必需**：整型 id，用于 eval 分组
        }


def build_router_input_test(question_dialogue, answer):
    return f"<|user|>: {question_dialogue}\n<|assistant|>: {answer}"


class PreferenceDataset_ood(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024, target_rm_indices=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_rm_indices = target_rm_indices  # or [0, 1, 2]  # 默认前三个 RM
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        with open(data_path) as f:
            raw_data = json.load(f)

        # 对每个 RM 分别收集其 score_diff（非 -1000）用于计算均值和方差
        score_stats = {i: [] for i in self.target_rm_indices}
        for item in raw_data:
            for i in self.target_rm_indices:
                score = item["score_diffs"][i]
                if score != -1000.0:
                    score_stats[i].append(score)

        # 计算每个 RM 的均值和标准差
        rm_mean_std = {}
        for i in self.target_rm_indices:
            scores = np.array(score_stats[i])
            mean = scores.mean()
            std = scores.std() if scores.std() > 1e-6 else 1.0  # 防止除以0
            rm_mean_std[i] = (mean, std)

        # 对所有样本的 score_diff 进行归一化
        for item in raw_data:
            for i in self.target_rm_indices:
                score = item["score_diffs"][i]
                if score != -1000.0:
                    mean, std = rm_mean_std[i]
                    item["score_diffs"][i] = (score - mean) / std
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

        pair_id = item.get("pair_id", "")
        subset_name = extract_subset_from_pair_id(pair_id)  # e.g. "Focus" or "Precise_IF"
        subset_key = _canonicalize_subset_name(subset_name)  # e.g. "focus" or "preciseif"
        # 如果未识别则返回 -1；compute_combined_metrics 只对 EXPECTED_SUBSETS (0..4) 输出
        # subset_id = SUBSET_NAME_TO_ID.get(subset_key, -1)
        subset_id = map_subset_to_id(subset_name)
        return {
            "input_ids_1": encoded_1["input_ids"].squeeze(0),
            "attention_mask_1": encoded_1["attention_mask"].squeeze(0),
            "input_ids_2": encoded_2["input_ids"].squeeze(0),
            "attention_mask_2": encoded_2["attention_mask"].squeeze(0),
            "score_diff_labels": score_labels,
            "cls_labels": cls_labels,
            "bt_pairs": bt_pairs,
            "pair_id": pair_id,  # 可选：保留调试用
            "subset_name": subset_name,  # 可选：保留调试用
            "subset_id": subset_id,  # **必需**：整型 id，用于 eval 分组
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

    from collections import Counter
    cnt = Counter()
    sample_keys = []
    for i in range(len(val_dataset)):
        sid = val_dataset[i].get("subset_id", -1)
        cnt[sid] += 1
        if i < 20:
            # 采样打印前 20 条的 pair_id / subset_name / subset_id，便于人工检查
            sample_keys.append((val_dataset[i].get("pair_id"), val_dataset[i].get("subset_name"), sid))
    print("VAL subset_id distribution (sid -> count):", dict(cnt))
    print("VAL sample pairs (first 20):")
    for p, sname, sid in sample_keys:
        print("  pair_id:", p, "-> subset_name:", sname, "-> sid:", sid)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        # weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        adam_epsilon=args.adam_epsilon,
        save_steps=10,
        save_strategy="steps",
        save_total_limit=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        load_best_model_at_end=True,  # 是否在训练结束时，用最佳 checkpoint 覆盖当前模型，设为false可以保存最后一个
        metric_for_best_model="router_accuracy",  # 根据加权
        greater_is_better=True,
        # fp16=True,
    )

    trainer = MultiOutputTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_combined_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)  # 保存最佳模型到输出目录
    tokenizer.save_pretrained(args.output_dir)  # 保存 tokenizer 配置（必须）


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--train_data_path", type=str, default=Dir3 + "train_helpsteer_RMBench_only1347.json")
    parser.add_argument("--val_data_path", type=str, default=Dir2 + "all_samples.json")
    parser.add_argument("--output_dir", type=str, default=Dir3 + "output5")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1.0e-8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num_reward_models", type=int, default=4)
    args = parser.parse_args()

    train(args)


