# evaluate.py
"""
载入训练完成并保存到 output_dir 的 RewardDiffPredictor 模型，
然后在 val_data（OOD）上按 pair_id 前缀分组并分别评估 router 的指标。

用法示例:
python evaluate.py --model_dir /path/to/output_dir --val_data_path /path/to/ood_v2_test.json \
    --tokenizer_name HuggingFaceTB/SmolLM2-135M --batch_size 16 --device cuda

注意: 需要 offline_model.py 在同一目录或可导入路径中（包含 RewardDiffPredictor, get_tokenizer）。
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig
from tqdm import tqdm
from offline_model_2encoder_100M2 import RewardDiffPredictor, get_tokenizer

Dir = "/data/cs.aau.dk/zh45qz/router_data/helpsteer3/"
Dir2 = "/data/cs.aau.dk/zh45qz/router_data/rewardbenchV2/"

# ------------------ util ------------------
def extract_subset_from_pair_id(pair_id: str):
    if not isinstance(pair_id, str) or len(pair_id) == 0:
        return "unknown"
    if "__" in pair_id:
        return pair_id.split("__", 1)[0]
    return re.split(r'[_\-\s]', pair_id, maxsplit=1)[0]

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------ Eval Dataset (简化版，基于你的 PreferenceDataset_ood) ------------------
class EvalDataset(Dataset):
    def __init__(self, raw_items, tokenizer, max_length=1024, target_rm_indices=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_rm_indices = list(target_rm_indices)
        self.raw = raw_items

        # optional normalization (kept for compatibility)
        score_stats = {i: [] for i in self.target_rm_indices}
        for item in self.raw:
            for i in self.target_rm_indices:
                if "score_diffs" in item and len(item["score_diffs"])>i:
                    s = item["score_diffs"][i]
                    if s != -1000.0:
                        score_stats[i].append(s)
        rm_mean_std = {}
        for i in self.target_rm_indices:
            arr = np.array(score_stats[i]) if len(score_stats[i])>0 else np.array([0.0])
            mean = arr.mean() if arr.size>0 else 0.0
            std = arr.std() if arr.size>0 and arr.std()>1e-6 else 1.0
            rm_mean_std[i] = (mean, std)
        for item in self.raw:
            if "score_diffs" in item:
                for i in self.target_rm_indices:
                    s = item["score_diffs"][i]
                    if s != -1000.0:
                        m, sd = rm_mean_std[i]
                        try:
                            item["score_diffs"][i] = (s - m) / sd
                        except Exception:
                            item["score_diffs"][i] = 0.0

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        item = self.raw[idx]
        pair_id = item.get("pair_id", "")
        subset = extract_subset_from_pair_id(pair_id)
        # print(subset)
        # print('='*200)

        input_text_1 = f"<|user|>: {item['question']}\n<|assistant|>: {item['chosen_answer']}"
        input_text_2 = f"<|user|>: {item['question']}\n<|assistant|>: {item['rejected_answer']}"

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
        cls_labels = torch.tensor(cls_labels, dtype=torch.long)

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
            "cls_labels": cls_labels,
            "bt_pairs": bt_pairs,
            "pair_id": pair_id,
            "subset": subset
        }

# collate_fn 与 train 中 data_collator 行为一致（但这里只需要 cls_labels 与 bt_pairs）
def collate_fn(batch):
    batch_out = {}
    batch_out["input_ids_1"] = torch.stack([b["input_ids_1"] for b in batch])
    batch_out["attention_mask_1"] = torch.stack([b["attention_mask_1"] for b in batch])
    batch_out["input_ids_2"] = torch.stack([b["input_ids_2"] for b in batch])
    batch_out["attention_mask_2"] = torch.stack([b["attention_mask_2"] for b in batch])
    batch_out["cls_labels"] = torch.stack([b["cls_labels"] for b in batch]).float()

    raw_bt = [b["bt_pairs"] for b in batch]
    B = len(raw_bt)
    M = max((len(p) for p in raw_bt), default=0)
    bt_array = torch.full((B, M, 2), -1, dtype=torch.long)
    for i, pairs in enumerate(raw_bt):
        for j, (w, l) in enumerate(pairs):
            bt_array[i, j, 0] = w
            bt_array[i, j, 1] = l
    batch_out["bt_pairs"] = bt_array

    batch_out["pair_ids"] = [b["pair_id"] for b in batch]
    batch_out["subset_names"] = [b["subset"] for b in batch]
    return batch_out

# ------------------ 评估计算函数 ------------------
def router_acc(preds, label_matrix):
    correct = 0
    count = 0
    for idx, p in enumerate(preds):
        row = label_matrix[idx]
        if np.sum(row) == 0:
            continue
        count += 1
        if p < 0 or p >= row.shape[0]:
            continue
        if int(row[p]) == 1:
            correct += 1
    return (correct / count) if count > 0 else float("nan")


def majority_voting_acc(label_matrix, rng):
    """
    多数投票baseline：对于每个query，如果多数RM标记为1则正确，多数为0则错误，平局时随机决定
    """
    correct = 0
    count = 0
    for idx in range(label_matrix.shape[0]):
        row = label_matrix[idx]
        if np.sum(row) == 0:
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
            if hasattr(rng, "random") and callable(rng.random):
                # 对于 np.random.Generator
                if rng.random() > 0.5:
                    correct += 1
            else:
                # 对于传统 np.random
                if np.random.rand() > 0.5:
                    correct += 1

    return (correct / count) if count > 0 else float("nan")


def compute_router_metrics(all_preds, all_cls_labels, num_rms, rng):
    """
    all_preds: np.array shape (N,) predicted rm index (local 0..num_rms-1)
    all_cls_labels: np.array shape (N, num_rms) of 0/1
    rng: numpy random generator (np.random.default_rng) or anything
    returns dict of metrics
    """
    metrics = {}
    pick_counts = Counter(all_preds.tolist())
    for rm_idx in range(num_rms):
        metrics[f'router_selected_rm{rm_idx}'] = int(pick_counts.get(rm_idx, 0))
    metrics['router_accuracy'] = float(router_acc(all_preds, all_cls_labels))

    # Single RM baselines
    for k in range(num_rms):
        metrics[f'baseline_rm{k}'] = float(router_acc(np.full(len(all_preds), k, dtype=int), all_cls_labels))

    # Random baseline
    if hasattr(rng, "integers"):
        rnd = rng.integers(0, num_rms, size=len(all_preds))
    else:
        rnd = np.random.randint(0, num_rms, size=len(all_preds))
    metrics['baseline_random'] = float(router_acc(rnd, all_cls_labels))

    # Majority voting baseline (新增)
    metrics['baseline_majority_voting'] = float(majority_voting_acc(all_cls_labels, rng))

    return metrics

# def compute_router_metrics(all_preds, all_cls_labels, num_rms, rng):
#     """
#     all_preds: np.array shape (N,) predicted rm index (local 0..num_rms-1)
#     all_cls_labels: np.array shape (N, num_rms) of 0/1
#     rng: numpy random generator (np.random.default_rng) or anything
#     returns dict of metrics
#     """
#     metrics = {}
#     pick_counts = Counter(all_preds.tolist())
#     for rm_idx in range(num_rms):
#         metrics[f'router_selected_rm{rm_idx}'] = int(pick_counts.get(rm_idx, 0))
#     metrics['router_accuracy'] = float(router_acc(all_preds, all_cls_labels))
#     for k in range(num_rms):
#         metrics[f'baseline_rm{k}'] = float(router_acc(np.full(len(all_preds), k, dtype=int), all_cls_labels))
#
#     # 兼容 np.random.Generator（有 integers）和旧版 np.random (randint)
#     if hasattr(rng, "integers"):
#         rnd = rng.integers(0, num_rms, size=len(all_preds))
#     else:
#         rnd = np.random.randint(0, num_rms, size=len(all_preds))
#
#     metrics['baseline_random'] = float(router_acc(rnd, all_cls_labels))
#     return metrics

def evaluate_single_group(items, tokenizer, model, device, args, rng):
    if len(items) == 0:
        return {"note": "empty_subset"}
    ds = EvalDataset(items, tokenizer, max_length=args.max_length, target_rm_indices=args.target_rm_indices)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=True)

    all_preds = []
    all_cls = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc="batches", leave=False):
            input_ids_1 = batch["input_ids_1"].to(device)
            attention_mask_1 = batch["attention_mask_1"].to(device)
            input_ids_2 = batch["input_ids_2"].to(device)
            attention_mask_2 = batch["attention_mask_2"].to(device)

            outputs = model(
                input_ids_1=input_ids_1,
                attention_mask_1=attention_mask_1,
                input_ids_2=input_ids_2,
                attention_mask_2=attention_mask_2
            )
            logits_tuple = outputs.logits
            if isinstance(logits_tuple, (list, tuple)) and len(logits_tuple) >= 3:
                bt_scores = logits_tuple[2]
            else:
                bt_scores = outputs.bt_scores

            preds = torch.argmax(bt_scores, dim=1).cpu().numpy()
            cls_labels = batch["cls_labels"].cpu().numpy().astype(int)

            all_preds.append(preds)
            all_cls.append(cls_labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_cls = np.concatenate(all_cls, axis=0)
    num_rms = all_cls.shape[1]
    metrics = compute_router_metrics(all_preds, all_cls, num_rms, rng)
    return metrics

def load_model_from_dir(model_dir, config, num_reward_models):
    # 首选直接使用 from_pretrained（它能识别 model.safetensors）
    try:
        print("尝试 RewardDiffPredictor.from_pretrained(...) 加载模型（优先）...")
        model = RewardDiffPredictor.from_pretrained(model_dir, config=config, num_reward_models=num_reward_models)
        print("from_pretrained 成功。")
        return model
    except Exception as e:
        print("from_pretrained 失败，退回到手动加载 state_dict。错误：", e)

    # 手动加载：优先处理 model.safetensors，否则尝试 pytorch_model.bin
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pt_path = os.path.join(model_dir, "pytorch_model.bin")
    sd = None
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file as load_safetensors
            print("使用 safetensors 载入 model.safetensors ...")
            sd = load_safetensors(safetensors_path, device="cpu")
        except Exception as e:
            print("safetensors 加载失败，请确认已安装 safetensors（pip install safetensors）。错误：", e)
    elif os.path.exists(pt_path):
        print("加载 pytorch_model.bin ...")
        sd = torch.load(pt_path, map_location="cpu")
    else:
        raise FileNotFoundError("未找到 model.safetensors 或 pytorch_model.bin，请确认 model_dir 是否包含保存的模型文件。")

    # 如果 sd 包含 {'state_dict': {...}} 的包装，解包
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # instantiate model and load
    model = RewardDiffPredictor(config=config, num_reward_models=num_reward_models)
    # 尝试剥离常见前缀
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        new_sd[new_k] = v
    try:
        model.load_state_dict(new_sd, strict=False)
    except Exception as e:
        print("load_state_dict 仍然失败，尝试直接加载原始 sd（strict=False）。错误：", e)
        model.load_state_dict(sd, strict=False)
    return model

def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("Using device:", device)
    set_seed(args.seed)

    # load tokenizer: 优先尝试 model_dir（训练时保存的 tokenizer）
    try:
        tokenizer = get_tokenizer(args.model_dir)
        print("使用 model_dir 下的 tokenizer。")
    except Exception:
        tokenizer = get_tokenizer(args.tokenizer_name)
        print("使用 tokenizer_name 指定的 tokenizer。")

    # load val data
    with open(args.val_data_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        raw_items = raw["data"]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        # 找到第一个是 list 的 value
        raw_items = []
        for v in (raw.values() if isinstance(raw, dict) else []):
            if isinstance(v, list):
                raw_items = v
                break

    groups = defaultdict(list)
    for it in raw_items:
        pid = it.get("pair_id", "")
        subset = extract_subset_from_pair_id(pid)
        groups[subset].append(it)

    eval_subsets = sorted(groups.keys()) if not args.subsets else [s for s in [x.strip() for x in args.subsets.split(",")] if s in groups]
    print("将评估的子集：", eval_subsets)

    # load config and model
    config = AutoConfig.from_pretrained(args.model_dir)
    model = load_model_from_dir(args.model_dir, config=config, num_reward_models=args.num_reward_models)
    model.to(device)
    model.eval()

    rng = np.random.default_rng(args.seed)
    results = {}

    # overall
    print("评估总体...")
    results["overall"] = evaluate_single_group(raw_items, tokenizer, model, device, args, rng)
    # per subset
    for s in eval_subsets:
        items = groups[s]
        print(f"评估子集 {s} (n={len(items)}) ...")
        results[s] = evaluate_single_group(items, tokenizer, model, device, args, rng)

    out_path = os.path.join(args.model_dir, args.output_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("评估完成，结果保存在：", out_path)

# ------------------ CLI ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=Dir + "output7/checkpoint-3000")
    parser.add_argument("--val_data_path", type=str, default=Dir2 + "all_samples.json")
    parser.add_argument("--tokenizer_name", type=str, default=Dir + "output7")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsets", type=str, default=None, help="Factuality, Precise IF, Math, Safety, Focus")
    parser.add_argument("--output_name", type=str, default=Dir + "eval.json")
    parser.add_argument("--num_reward_models", type=int, default=4)
    parser.add_argument("--target_rm_indices", type=int, nargs="+", default=[1,3,4,7])
    args = parser.parse_args()
    evaluate(args)
