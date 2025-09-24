import os
import time
import yaml
import json
import math
import random
import gc
from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed.rpc as rpc
from offline_model_2encoder_100M2 import RewardDiffPredictor, get_tokenizer


# config_loader
def load_config(config_file):
    """Load the YAML configuration file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


class DatasetManagerTraining:
    def __init__(self, seed=42):
        self.filtered_data = []
        self.seed = seed

        dataset_path = "/data/cs.aau.dk/zh45qz/router_data/rewardbenchV2/all_samples.json"
        # dataset_path = "/data/cs.aau.dk/zh45qz/router_data/RM_Bench/all_samples.json"
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        indices_to_keep = [1, 3, 4, 7]
        for sample in data:
            filtered_int_labels = [sample['int_labels'][i] for i in indices_to_keep]
            if 1 not in filtered_int_labels:
                continue
            filtered_sample = {
                "id": sample["id"],
                "question": sample["question"],
                "chosen_answer": sample["chosen_answer"],
                "rejected_answer": sample["rejected_answer"],
                "int_labels": filtered_int_labels
            }
            self.filtered_data.append(filtered_sample)
        random.shuffle(self.filtered_data)
        print(f'len filtered train: {len(self.filtered_data)}')

    def get_train_data(self, dataset_name):
        return self.filtered_data


class DatasetManagerMMLU:
    def __init__(self, dataset_paths, dev_size=0.1, seed=42):
        self.datasets = {}
        self.dev_size = float(dev_size)
        self.seed = seed
        for name, path in dataset_paths.items():
            dataset_id = path.split("hf:", 1)[1]
            ds = load_dataset(dataset_id, name="all", split="test")  # e.g. cais/mmlu
            examples = []
            for ex in ds:
                q = ex.get('question', '')
                choices = ex.get('choices', '')
                ans = ex.get('answer', '')
                gold = choices[ans]
                question_with_choices = q + "\n" + "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
                examples.append({'question': question_with_choices, 'original_q': q, 'choices': choices, 'answer': str(gold)})

            train_examples, test_examples = train_test_split(
                examples, test_size=0.2, random_state=self.seed)
            train_examples, dev_examples = train_test_split(
                train_examples, test_size=self.dev_size, random_state=self.seed)
            print(f'len train: {len(train_examples)}, len dev: {len(dev_examples)}, len test: {len(test_examples)}', flush=True)
            print(test_examples[900])
            self.datasets[name] = {'train': train_examples[:10000], 'dev': dev_examples, 'test': test_examples}

    def get_train_data(self, dataset_name):
        return self.datasets[dataset_name]['train']

    def get_dev_data(self, dataset_name):
        return self.datasets[dataset_name]['dev']

    def get_test_data(self, dataset_name):
        return self.datasets[dataset_name]['test']


class DatasetManager:
    """
    简洁版：支持从 Hugging Face 直接加载数据集（通过 'hf:dataset_id' 约定）
    或从本地 json/jsonl 文件加载。目标：返回包含 'question' 和 'answer' 的 dev 列表。
    用法示例:
      dm = DatasetManager({'gsm8k': 'hf:openai/gsm8k'}, dev_size=0.1)
      dev = dm.get_dev_data('gsm8k')   # list of dicts, 每条至少含 'question' 和 'answer'
    """
    def __init__(self, dataset_paths, dev_size=0.1, seed=42):
        self.datasets = {}
        self.dev_size = float(dev_size)
        self.seed = seed
        for name, path in dataset_paths.items():
            if isinstance(path, str) and path.startswith("hf:"):
                dataset_id = path.split("hf:", 1)[1]
                ds = load_dataset(dataset_id, 'main')  # e.g. "openai/gsm8k"
                train_examples = list(ds['train'])
                test_examples = list(ds['test']) if 'test' in ds else []
            else:
                # 本地 json 或 jsonl（每行一个 json object 或整个 array）
                with open(path, 'r', encoding='utf-8') as f:
                    txt = f.read().strip()
                    if txt.startswith('['):
                        raw = json.loads(txt)
                    else:
                        raw = [json.loads(l) for l in txt.splitlines() if l.strip()]
                train_examples = raw
                test_examples = []

            # 最小化归一：保证每条都有 question/answer 字段（answer 允许为空）
            def _norm(ex):
                q = ex.get('question') or ex.get('question_text') or ex.get('problem') or ""
                a = ex.get('answer', "")
                return {**ex, 'question': str(q).strip(), 'answer': str(a).strip()}

            train_examples = [_norm(x) for x in train_examples]
            test_examples = [_norm(x) for x in test_examples]

            # 如果 HF 没有 test split，则从 train 中切一部分作为 test
            if not test_examples:
                train_examples, test_examples = train_test_split(
                    train_examples, test_size=0.2, random_state=self.seed)

            # 从剩下的 train 中切出 dev
            train_examples, dev_examples = train_test_split(
                train_examples, test_size=self.dev_size, random_state=self.seed)
            print(f'len train: {len(train_examples)}, len dev: {len(dev_examples)}, len test: {len(test_examples)}', flush=True)

            self.datasets[name] = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}

    def get_train_data(self, dataset_name):
        return self.datasets[dataset_name]['train']

    def get_dev_data(self, dataset_name):
        return self.datasets[dataset_name]['dev']

    def get_test_data(self, dataset_name):
        return self.datasets[dataset_name]['test']


# class BayesianRouter:
#     def __init__(self, d, rm_embeddings, sigma2=1.0, lambda0=1.0, tol_eig=1e-10):
#         """
#         d: 上下文向量维度
#         rm_embeddings: (K, d) 预训练嵌入，作为 prior mean
#         sigma2: 噪声方差，贝叶斯线性模型的方差
#         lambda0: prior precision scalar
#         tol_eig: 特征值下界比例，用于数值稳定
#         """
#         self.d = d
#         self.K = rm_embeddings.shape[0]
#         self.sigma2 = float(sigma2)
#         self.lambda0 = float(lambda0)
#         self.tol_eig = float(tol_eig)
#
#         # 全部用双精度
#         rm = rm_embeddings.double()
#         # # 暂时禁用先验知识
#         # rm = torch.zeros_like(rm_embeddings, dtype=torch.float64)
#
#         # 信息形式先验：Λ = λ0 I,  b = λ0 μ0
#         I = torch.eye(d, dtype=torch.float64, device=rm.device)
#         self.Lambda = I.unsqueeze(0).repeat(self.K, 1, 1) * self.lambda0  # (K,d,d)
#         self.b = (self.lambda0 * rm).clone()  # (K,d)
#         self.mu = rm.clone()  # (K,d)
#
#         # Welford 在线归一化
#         self.count = 0
#         self.mean = 0.0
#         self.M2 = 0.0
#
#     def normalize_reward(self, r: float) -> float:
#         self.count += 1
#         delta = r - self.mean
#         self.mean += delta / self.count
#         delta2 = r - self.mean
#         self.M2 += delta * delta2
#         if self.count < 2:
#             return 0.0
#         std = (self.M2 / (self.count - 1)) ** 0.5
#         return (r - self.mean) / (std + 1e-8)
#
#     def select_arm(self, context: torch.Tensor) -> int:
#         x = context.reshape(-1).double()
#         scores = torch.empty(self.K, dtype=torch.float64, device=x.device)
#
#         for k in range(self.K):
#             # 1) 对称化精度矩阵
#             A = 0.5 * (self.Lambda[k] + self.Lambda[k].mT)
#             # 2) 谱分解
#             eig, Q = torch.linalg.eigh(A)
#             # 3) 下界
#             eps = self.tol_eig * eig.max()
#             eig_clamped = torch.clamp(eig, min=eps)
#             # 4) 采样：w = μ + Q diag(1/sqrt(eig_clamped)) z
#             z = torch.randn(self.d, device=x.device, dtype=torch.float64)
#             inv_sqrt = eig_clamped.rsqrt()  # 1/sqrt(lambda_i)
#             y = (Q * inv_sqrt.unsqueeze(0)) @ (Q.t() @ z)
#             w = self.mu[k] + y
#             scores[k] = w.dot(x)
#
#         return int(scores.argmax().item())
#
#     def update(self, k: int, context: torch.Tensor, reward: float, normalized: bool = False):
#         """
#         如果 normalized=True，就直接把 reward 当作已经标准化（z-score）的数使用，
#         否则再通过内部 normalize_reward 做（目前我们推荐尽量传 normalized=True）。
#         """
#         if not normalized:
#             r_norm = self.normalize_reward(reward)
#         else:
#             # 直接使用（并保证为 float64/float32 与 self.mu 对齐）
#             r_norm = float(reward)
#
#         x = context.reshape(-1).to(self.mu.device).to(self.mu.dtype)
#         # 信息形式更新（与原逻辑一致）
#         self.Lambda[k] += torch.ger(x, x) / self.sigma2
#         self.b[k] += (r_norm / self.sigma2) * x
#
#         # 重算后验均值 μ = Λ^{-1} b （保持你原来的谱分解方法或改为更稳定的 cholesky）
#         A = 0.5 * (self.Lambda[k] + self.Lambda[k].mT)
#         eig, Q = torch.linalg.eigh(A)
#         eps = self.tol_eig * eig.max()
#         eig_clamped = torch.clamp(eig, min=eps)
#         inv = 1.0 / eig_clamped
#         self.mu[k] = (Q * inv.unsqueeze(0)) @ (Q.t() @ self.b[k])


class BayesianRouter:
    def __init__(self, d, rm_embeddings, sigma2=1.0, lambda0=1.0, tol_eig=1e-10, min_history_for_quantile=20):
        """
        d: 上下文向量维度
        rm_embeddings: (K, d) 预训练嵌入，作为 prior mean
        sigma2: 噪声方差，贝叶斯线性模型的方差
        lambda0: prior precision scalar
        tol_eig: 特征值下界比例，用于数值稳定
        min_history_for_quantile: 计算分位数所需的最小历史记录长度
        """
        self.d = d
        self.K = rm_embeddings.shape[0]
        self.sigma2 = float(sigma2)
        self.lambda0 = float(lambda0)
        self.tol_eig = float(tol_eig)
        self.min_history_for_quantile = min_history_for_quantile

        # # 全部用双精度
        # rm = rm_embeddings.double()
        # 暂时禁用先验知识
        rm = torch.zeros_like(rm_embeddings, dtype=torch.float64)

        # 信息形式先验：Λ = λ0 I,  b = λ0 μ0
        I = torch.eye(d, dtype=torch.float64, device=rm.device)
        self.Lambda = I.unsqueeze(0).repeat(self.K, 1, 1) * self.lambda0  # (K,d,d)
        self.b = (self.lambda0 * rm).clone()  # (K,d)
        self.mu = rm.clone()  # (K,d)

        self.reward_history = []

    def normalize_reward(self, r: float) -> float:
        self.reward_history.append(r)

        # 如果历史记录太短，无法稳定计算分位数，返回一个中性值（例如0.5）
        if len(self.reward_history) < self.min_history_for_quantile:
            return 0.5

        # 计算20%和80%分位数
        # 注意：需要将list转换为tensor来使用quantile方法
        history_tensor = torch.tensor(self.reward_history, dtype=torch.float64)
        q_lo = torch.quantile(history_tensor, 0.2)
        q_hi = torch.quantile(history_tensor, 0.8)

        # 处理分母为0的边缘情况（例如，所有历史奖励都相同）
        if q_hi <= q_lo:
            # 如果r高于这个单一值，返回1，否则返回0
            return 1.0 if r > q_lo else 0.0

        # 应用论文中的归一化和裁剪逻辑
        if r < q_lo:
            return 0.0
        elif r > q_hi:
            return 1.0
        else:
            return (r - q_lo) / (q_hi - q_lo)

    def select_arm(self, context: torch.Tensor) -> int:
        x = context.reshape(-1).double()
        scores = torch.empty(self.K, dtype=torch.float64, device=x.device)

        for k in range(self.K):
            A = 0.5 * (self.Lambda[k] + self.Lambda[k].mT)
            eig, Q = torch.linalg.eigh(A)
            eps = self.tol_eig * eig.max()
            eig_clamped = torch.clamp(eig, min=eps)
            z = torch.randn(self.d, device=x.device, dtype=torch.float64)
            inv_sqrt = eig_clamped.rsqrt()
            y = (Q * inv_sqrt.unsqueeze(0)) @ (Q.t() @ z)
            w = self.mu[k] + y
            scores[k] = w.dot(x)

        return int(scores.argmax().item())

    def update(self, k: int, context: torch.Tensor, reward: float, normalized: bool = False):
        if not normalized:
            r_norm = self.normalize_reward(reward)
        else:
            r_norm = float(reward)

        x = context.reshape(-1).to(self.mu.device).to(self.mu.dtype)
        self.Lambda[k] += torch.ger(x, x) / self.sigma2
        self.b[k] += (r_norm / self.sigma2) * x

        A = 0.5 * (self.Lambda[k] + self.Lambda[k].mT)
        eig, Q = torch.linalg.eigh(A)
        eps = self.tol_eig * eig.max()
        eig_clamped = torch.clamp(eig, min=eps)
        inv = 1.0 / eig_clamped
        self.mu[k] = (Q * inv.unsqueeze(0)) @ (Q.t() @ self.b[k])


# llm_trainer
class LLMTrainer:
    def __init__(self, model, optimizer, tokenizer, grad_accum_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = next(self.model.parameters()).device
        self.grad_accum_steps = grad_accum_steps
        self.accum_step = 0  # 当前累积步数计数器
        self.optimizer.zero_grad()  # 初始清空梯度
        self.dpo_beta = 1.0

        # 初始用 policy adapter
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.set_adapter("policy")
            self.model.module.train()
        else:
            self.model.set_adapter("policy")
            self.model.train()

    def _tokenize(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

    # def _log_probs_answer_only(self, prompts, answers):
    #     # prompts, answers: lists of strings
    #     joint_texts = [p + a for p, a in zip(prompts, answers)]
    #     joint = self._tokenize(joint_texts)  # tensors on device
    #     # prompt lengths (non-padded token counts) — compute per-sample without padding
    #     prefix_lens = torch.tensor(
    #         [len(self.tokenizer(p, add_special_tokens=True)['input_ids']) for p in prompts],
    #         device=self.device, dtype=torch.long
    #     )
    #
    #     outputs = self.model(input_ids=joint["input_ids"], attention_mask=joint["attention_mask"])
    #     logits = outputs.logits  # (B, T, V)
    #     shift_logits = logits[:, :-1, :]
    #     shift_labels = joint["input_ids"][:, 1:]
    #     shift_mask = joint["attention_mask"][:, 1:]  # valid positions (1 for real tokens)
    #
    #     log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    #     selected = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    #
    #     B, Tm1 = selected.shape
    #     ans_mask = torch.zeros_like(selected, dtype=shift_mask.dtype)
    #
    #     joint_nonpad = joint["attention_mask"].sum(dim=1)  # prompt+answer length (tokens)
    #     pad_counts = (joint["attention_mask"].size(1) - joint_nonpad).long()  # per-sample pad count
    #     for b in range(B):
    #         s = (pad_counts[b] + prefix_lens[b] - 1).item()  # index in shift_labels
    #         s = max(0, min(s, Tm1 - 1))
    #         ans_mask[b, s:] = 1
    #
    #     mask = shift_mask * ans_mask
    #     denom = mask.sum(dim=1).clamp_min(1.0)
    #     # average log-prob per token (answer tokens)
    #     return (selected * mask).sum(dim=1)  # / denom

    def _log_probs_answer_only(self, prompts, answers):
        # prompts, answers: lists of strings — prompts should be the exact strings used during generation
        joint_texts = [p + a for p, a in zip(prompts, answers)]
        joint = self._tokenize(joint_texts)  # tensors on device

        # prompt lengths in tokens (without padding), computed via tokenizer on CPU or same tokenizer
        # NOTE: use tokenizer.encode to get exact token counts (avoid batch padding interference)
        prefix_lens = []
        for p in prompts:
            toks = self.tokenizer(p, add_special_tokens=True)['input_ids']
            prefix_lens.append(len(toks))
        prefix_lens = torch.tensor(prefix_lens, device=self.device, dtype=torch.long)  # (B,)

        outputs = self.model(input_ids=joint["input_ids"], attention_mask=joint["attention_mask"])
        logits = outputs.logits  # (B, T, V)

        shift_logits = logits[:, :-1, :].float()  # cast to float32 for stable softmax
        shift_labels = joint["input_ids"][:, 1:]
        shift_mask = joint["attention_mask"][:, 1:]  # (B, T-1)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)  # float32
        selected = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        B, Tm1 = selected.shape
        ans_mask = torch.zeros_like(selected, dtype=shift_mask.dtype)

        joint_nonpad = joint["attention_mask"].sum(dim=1)  # tokens count in joint (prompt+answer)
        pad_counts = torch.full_like(joint_nonpad, joint["attention_mask"].size(1)) - joint_nonpad
        # For each sample, answer tokens start at: pad_count + prefix_len - 1 (because labels are shifted by 1)
        for b in range(B):
            s = int(pad_counts[b].item() + prefix_lens[b].item() - 1)
            s = max(0, min(s, Tm1 - 1))
            ans_mask[b, s:] = 1

        mask = shift_mask * ans_mask
        denom = mask.sum(dim=1).clamp_min(1.0)  # per-sample answer length
        # average log-prob per answer token
        avg_logp = (selected * mask).sum(dim=1)  # / denom
        return avg_logp  # (B,)

    # def dpo_loss(self, queries, y_w, y_l):
    #     if isinstance(self.model, DistributedDataParallel):
    #         self.model.module.set_adapter("policy")
    #     else:
    #         self.model.set_adapter("policy")
    #     # === 改为只对 answer tokens 计分 ===
    #     logp_w = self._log_probs_answer_only(queries, y_w)  # (N,)
    #     logp_l = self._log_probs_answer_only(queries, y_l)  # (N,)
    #
    #     if isinstance(self.model, DistributedDataParallel):
    #         self.model.module.set_adapter("reference")
    #     else:
    #         self.model.set_adapter("reference")
    #
    #     with torch.no_grad():
    #         ref_logp_w = self._log_probs_answer_only(queries, y_w)  # (N,)
    #         ref_logp_l = self._log_probs_answer_only(queries, y_l)  # (N,)
    #
    #     if isinstance(self.model, DistributedDataParallel):
    #         self.model.module.set_adapter("policy")
    #     else:
    #         self.model.set_adapter("policy")
    #
    #     diff = (logp_w - ref_logp_w) - (logp_l - ref_logp_l)  # (N,)
    #     diff = torch.clamp(diff, min=-5.0, max=5.0)
    #     return -torch.log(torch.sigmoid(diff)).mean()

    def dpo_loss(self, prompts, y_w, y_l):
        """
        返回: (loss_tensor_for_backward, per_sample_loss_tensor_detached)
        per_sample_loss_tensor_detached shape (B,)
        """
        model_core = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

        # ensure policy adapter active for forward with grad
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.set_adapter("policy")
        else:
            self.model.set_adapter("policy")

        was_training = model_core.training
        model_core.eval()  # keep in eval, but enable_grad to compute grads only where needed

        with torch.enable_grad():
            logp_w = self._log_probs_answer_only(prompts, y_w)  # (B,)
            logp_l = self._log_probs_answer_only(prompts, y_l)  # (B,)

        # reference (no grad)
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.set_adapter("reference")
        else:
            self.model.set_adapter("reference")
        model_core.eval()
        with torch.no_grad():
            ref_logp_w = self._log_probs_answer_only(prompts, y_w)
            ref_logp_l = self._log_probs_answer_only(prompts, y_l)

        # restore adapter
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.set_adapter("policy")
        else:
            self.model.set_adapter("policy")
        if was_training:
            model_core.train()

        diff = (logp_w - ref_logp_w) - (logp_l - ref_logp_l)
        diff = torch.clamp(diff, min=-50.0, max=50.0)
        beta = self.dpo_beta
        scaled = beta * diff
        per_sample_loss = -torch.log(torch.sigmoid(scaled) + 1e-12)  # (B,)
        loss = per_sample_loss.mean()
        return loss, per_sample_loss.detach().clone()

    def nll_loss(self, y_list):
        # y_list: list of strings (answers)
        inputs = self._tokenize(y_list)  # returns tensors on device
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = inputs['input_ids'][:, 1:]
        shift_mask = inputs['attention_mask'][:, 1:]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        selected = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        denom = shift_mask.sum(dim=1).clamp_min(1.0)
        seq_logp = (selected * shift_mask).sum(dim=1)  # / denom
        return -seq_logp.mean()

    # def train_step(self, preference_pairs):
    #     total_loss_sum = 0.0
    #     total_samples = 0
    #     chunk_size = max(1, math.ceil(len(preference_pairs) / self.grad_accum_steps))
    #     chunks = [preference_pairs[i:i + chunk_size] for i in range(0, len(preference_pairs), chunk_size)]
    #     num_chunks = len(chunks)
    #
    #     for chunk in chunks:
    #         queries, ys_w, ys_l = zip(*chunk)
    #         loss = self.dpo_loss(list(queries), list(ys_w), list(ys_l))
    #         # 固定以 grad_accum_steps 缩放
    #         loss_to_backprop = loss / float(self.grad_accum_steps)
    #         loss_to_backprop.backward()
    #         total_loss_sum += loss.item() * len(chunk)
    #         total_samples += len(chunk)
    #
    #         self.accum_step += 1
    #         if self.accum_step % self.grad_accum_steps == 0:
    #             torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=1.0)
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
    #             self.accum_step = 0
    #
    #     # flush remaining
    #     if self.accum_step > 0:
    #         torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=1.0)
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()
    #         self.accum_step = 0

    def train_step(self, preference_pairs):
        """
        preference_pairs: list of (prompt, y_w, y_l)
        返回:
          - avg_loss (float)
          - per_pair_losses (list of floats)  与 preference_pairs 一一对应
        """
        total_loss_sum = 0.0
        total_samples = 0
        per_pair_losses = []

        micro_batch_size = max(1, math.ceil(len(preference_pairs) / self.grad_accum_steps))
        chunks = [preference_pairs[i:i + micro_batch_size] for i in range(0, len(preference_pairs), micro_batch_size)]

        for idx, chunk in enumerate(chunks):
            prompts, ys_w, ys_l = zip(*chunk)
            loss_tensor, per_sample_loss_tensor = self.dpo_loss(list(prompts), list(ys_w), list(ys_l))
            # scale by grad_accum_steps for backward
            loss_tensor = loss_tensor / float(self.grad_accum_steps)
            loss_tensor.backward()

            # accumulate numeric sums (use per_sample_loss_tensor which was detached)
            total_loss_sum += float(per_sample_loss_tensor.sum().item())
            total_samples += per_sample_loss_tensor.numel()
            per_pair_losses.extend([float(x) for x in per_sample_loss_tensor.tolist()])

            if (idx + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # leftover step (if chunks % grad_accum_steps != 0)
        if (len(chunks) % self.grad_accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss_sum / max(1, total_samples)
        return avg_loss, per_pair_losses


def generate_offline_prior(queries, resp_as, resp_bs):
    # 1) 一次 rpc 批量 encode -> 返回 shape (N, D)
    batch_emb = rpc.rpc_sync(
        to="worker0",
        func=rpc_offline_batch_encode,
        args=(queries, resp_as, resp_bs)
    )  # CPU tensor

    # 2) 一次 rpc 批量 logits -> 返回 shape (N, K)
    batch_logits = rpc.rpc_sync(
        to="worker0",
        func=rpc_offline_batch_logits,
        args=(batch_emb,)
    )

    return batch_emb, batch_logits


def generate_preference_pairs_multi_rm(pair_queries, resp_as, resp_bs, rm_labels, pair_rm_indices):
    # 对每个query，都选择了一个rm，
    sel_prefs = []
    sel_pair_indices = []
    sel_pair_rm_indices = []
    i = 0
    for (query, resp_a, resp_b, rm_label_list, rm_indice) in zip(pair_queries, resp_as, resp_bs, rm_labels, pair_rm_indices):
        if rm_label_list[rm_indice] == 1:
            sel_prefs.append((query, resp_a, resp_b))
        else:
            sel_prefs.append((query, resp_b, resp_a))
        sel_pair_indices.append(i)
        i += 1
        sel_pair_rm_indices.append(rm_indice)

    return sel_prefs, sel_pair_indices, sel_pair_rm_indices


# Global storage for reward models
_reward_models = []


# reward_model
class RewardModel:
    def __init__(self, name, model, tokenizer, device):
        self.name = name
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def prepare_batch_inputs(self, queries, responses):
        formatted_prompts = []

        for q, r in zip(queries, responses):
            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": r}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
            formatted_prompts.append(prompt)

        tokenized_inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        return tokenized_inputs

    def batch_score(self, queries, responses):
        inputs = self.prepare_batch_inputs(queries, responses)
        # print('before model', flush=True)
        with torch.no_grad():
            batch_rewards = self.model(inputs['input_ids'],
                                       attention_mask=inputs["attention_mask"])[0].cpu()
        # print('after model', flush=True)

        return [reward.item() for reward in batch_rewards]


class ProxyRewardModel:
    """Client-side proxy: forwards batch_score requests via RPC to rank0."""

    def __init__(self, idx):
        self.idx = idx

    def batch_score(self, queries, responses):
        # Remote call to server process
        # print('proxy', flush=True)
        return rpc.rpc_sync(
            to="worker0",
            func=_remote_batch_score,
            args=(self.idx, queries, responses)
        )


# # 文件顶部（或合适位置）插入
# import threading
# _rpc_forward_lock = threading.Lock()
def _remote_batch_score(rm_idx, queries, responses, chunk_size: int = 8, min_chunk: int = 1, verbose: bool = False):
    model = _reward_models[rm_idx]

    # fast path
    N = len(queries)
    if N == 0:
        return []

    results = []
    # serialize all forward passes on rank0 to avoid concurrent overlapping activations
    # with _rpc_forward_lock:
    i = 0
    cur_chunk = max(1, int(chunk_size))
    # loop until we've processed all items
    while i < N:
        end = min(i + cur_chunk, N)
        q_chunk = queries[i:end]
        r_chunk = responses[i:end]

        try:
            # Use inference_mode for max memory savings (no autograd bookkeeping)
            with torch.inference_mode():
                # RewardModel.batch_score already does prepare_batch_inputs and model forward
                chunk_scores = model.batch_score(q_chunk, r_chunk)
                # normalize / convert to floats
                chunk_scores = [float(x) for x in chunk_scores]
            # append results and advance
            results.extend(chunk_scores)
            i = end
            # reset chunk size to initial for next window (if we had reduced it earlier)
            cur_chunk = max(cur_chunk, int(chunk_size))
        except RuntimeError as e:
            # detect OOM-like errors and try to recover by shrinking chunk
            msg = str(e).lower()
            is_oom = ("out of memory" in msg) or ("cuda" in msg and "out" in msg)
            if is_oom:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                # shrink chunk and retry this window
                if cur_chunk > min_chunk:
                    new_chunk = max(min_chunk, cur_chunk // 2)
                    if verbose:
                        print(f"[_remote_batch_score] OOM while scoring chunk size {cur_chunk}. "
                              f"Retrying with chunk {new_chunk}.", flush=True)
                    cur_chunk = new_chunk
                    # do not advance i; retry same window with smaller chunk
                    continue
                else:
                    # if already at min_chunk, re-raise so caller sees failure
                    if verbose:
                        print(f"[_remote_batch_score] OOM at min_chunk={min_chunk}. Raising.", flush=True)
                    raise
            else:
                # not OOM -> re-raise
                raise
        finally:
            # best-effort cleanup after each chunk
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    return results


def load_reward_models(model_names, device, multi_gpu=False, rpc_info=None):
    """
    Load RewardModel instances on rank0, or create proxies on other ranks.
    """
    global _reward_models
    _reward_models = []  # 这是个对所有进程可见的全局变量？

    # Determine if this is the rank0 process
    if rpc_info is not None:
        is_leader = (rpc_info.get("rank", -1) == 0)
    else:
        info = rpc.get_worker_info()
        is_leader = (info and info.name == "worker0")

    if is_leader:
        # Get the list of available devices (GPUs) if multi_gpu is enabled
        # 有可能一个gpu塞不下所有的reward models，比如(8+7+3+2)*2 = 40G参数，再加额外的就装不下了，所以multi_gpu还是需要的；
        devices = [torch.device(f"cuda:{i}") for i in
                   range(torch.cuda.device_count())] if multi_gpu and torch.cuda.is_available() else [device]
        for idx, name in enumerate(model_names):
            # Determine the device to load this model on (distribute across available devices if multi_gpu)
            device_for_model = devices[idx % len(devices)]

            model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(name)
            # Create a RewardModel instance and add it to the list
            _reward_models.append(RewardModel(name, model, tokenizer, device_for_model))
    else:
        # On other ranks, create proxies
        for idx in range(len(model_names)):
            _reward_models.append(ProxyRewardModel(idx))

    return _reward_models


# ===== RPC Handlers =====
_bayes_router = None
_embed = None
_offline_router_model = None
_offline_router_tokenizer = None
# Per-arm moving-average baselines (maintained on rank0)
_arm_baseline_alpha = None
_arm_baselines_mean = None  # list of floats (NaN if uninit)
_arm_baselines_sq = None  # list of floats (NaN if uninit)
_arm_baseline_clip_z = 2.0  # 可配置的 clipping
_arm_baseline_eps = 1e-6
# per-arm sample counts + minimum samples before allowing bayes update
_arm_counts = None
_arm_min_count = 50  # 可以调整为 5/10 等


def rpc_init_globals(num_reward_models: int,
                     offline_router_path: str,
                     sigma2: float = 1.0,
                     lambda0: float = 1.0,
                     device: torch.device = torch.device("cuda:0"),
                     baseline_alpha: float = 0.01):
    global _bayes_router, _embed, _arm_baseline_alpha, _arm_baselines_mean, _arm_baselines_sq
    rm_embeddings = rpc_init_offline_router(offline_router_path, num_reward_models, device)
    d = rm_embeddings.shape[1]
    _bayes_router = BayesianRouter(d, rm_embeddings, sigma2=sigma2, lambda0=lambda0)
    _bayes_router._decomp_cache = [None] * _bayes_router.K
    # import threading
    # _bayes_router._decomp_lock = threading.Lock()

    # per-arm EMA baseline init
    _arm_baseline_alpha = float(baseline_alpha)
    _arm_baselines_mean = [float('nan')] * num_reward_models
    _arm_baselines_sq = [float('nan')] * num_reward_models
    # 初始化每臂样本计数，供 min_count 逻辑使用
    global _arm_counts
    _arm_counts = [0] * num_reward_models
    print(f"[Rank0] Initialized {_bayes_router.K} arm baselines (alpha={_arm_baseline_alpha}).")


# def rpc_select_arms(contexts, priors=None, beta=0.0):
#     # 如果传了 priors（offline logits），直接用 priors 做选择（argmax per row）
#     if priors is not None:
#         # priors expected shape (N, K) tensor on CPU
#         if isinstance(priors, torch.Tensor):
#             if priors.numel() == 0:
#                 return []
#             # per-row argmax -> list of ints
#             return [int(x) for x in priors.argmax(dim=1).tolist()]
#         else:
#             # defensive: handle list/ndarray
#             arr = torch.tensor(priors)
#             if arr.numel() == 0:
#                 return []
#             return [int(x) for x in arr.argmax(dim=1).tolist()]
#
#     # # 否则回退到原来的 BayesianRouter 行为
#     # if contexts is None:
#     #     return []
#     # ctxs = contexts.reshape(contexts.shape[0], -1)
#     # chosen = []
#     # for i in range(ctxs.shape[0]):
#     #     ctx = ctxs[i]
#     #     try:
#     #         ctx = ctx.to(_bayes_router.mu.device).to(_bayes_router.mu.dtype)
#     #     except Exception:
#     #         pass
#     #     idx = _bayes_router.select_arm(ctx)
#     #     chosen.append(int(idx))
#     # return chosen


# 替换当前的 rpc_select_arms 为下面版本（放在 utils.py 的 RPC handler 区域）
def rpc_select_arms(contexts, priors=None, beta=0.0):
    """
    Vectorized selection:
      - contexts: CPU tensor shape (N, d) or (N,1,d) etc.
      - returns: list of chosen arm indices length N
    Key idea: for each arm k, compute a decomposition (cholesky or eig) ONCE,
      draw N samples y_k_j (j=0..N-1) from N(0, A^{-1}), then w_k_j = mu_k + y_k_j,
      score_k_j = w_k_j dot x_j, finally choose argmax_k score_k_j for each j.
    """
    if contexts is None:
        return []
    # ensure contexts is 2D (N, d)
    ctxs = contexts.reshape(contexts.shape[0], -1)

    # Move contexts to router device/dtype
    try:
        device = _bayes_router.mu.device
        dtype = _bayes_router.mu.dtype
        ctxs = ctxs.to(device).to(dtype)
    except Exception:
        device = ctxs.device
        dtype = ctxs.dtype

    N, d = ctxs.shape
    K = _bayes_router.K

    # prepare scores matrix K x N
    scores = torch.empty((K, N), dtype=dtype, device=device)

    # Ensure we have cache structures on _bayes_router (will be created in rpc_init_globals)
    # _bayes_router._decomp_cache: list of per-arm cached decomposition dicts or None
    # We attempt cholesky (fast) and fallback to eig if needed.
    eye = torch.eye(d, dtype=dtype, device=device)

    for k in range(K):
        # retrieve cached decomposition if available
        decomp = None
        if hasattr(_bayes_router, "_decomp_cache") and _bayes_router._decomp_cache[k] is not None:
            decomp = _bayes_router._decomp_cache[k]

        # Ensure Lambda/mu on right device/dtype
        A = 0.5 * (_bayes_router.Lambda[k] + _bayes_router.Lambda[k].mT)
        if A.device != device:
            A = A.to(device)
        if A.dtype != dtype:
            A = A.to(dtype)

        if decomp is None:
            # Try Cholesky (fast) with small jitter; fallback to eig
            jitter = max(_bayes_router.tol_eig, 1e-8)
            success = True
            try:
                L = torch.linalg.cholesky(A + jitter * eye)  # lower-triangular
                # cache L for later reuse
                decomp = {"mode": "chol", "L": L}
            except RuntimeError:
                # fallback to eigh and clamp, store Q and inv_sqrt for reuse
                eig, Q = torch.linalg.eigh(A)
                eps = _bayes_router.tol_eig * eig.max()
                eig_clamped = torch.clamp(eig, min=eps)
                inv_sqrt = eig_clamped.rsqrt()
                decomp = {"mode": "eig", "Q": Q, "inv_sqrt": inv_sqrt}

            # store cache
            if hasattr(_bayes_router, "_decomp_cache"):
                _bayes_router._decomp_cache[k] = decomp

        # Now sample N y's vectorized according to decomp
        if decomp["mode"] == "chol":
            L = decomp["L"]  # lower tri, shape (d,d)
            # draw Z ~ N(0,I) shape (N, d)
            Z = torch.randn((N, d), device=device, dtype=dtype)
            # Solve L^T * y.T = Z.T  => y = solve_triangular(L.T, Z.T).T
            # use solve_triangular (fast, batched)
            y = torch.linalg.solve_triangular(L.transpose(0, 1), Z.T, upper=True).T  # shape (N,d)
        else:
            Q = decomp["Q"]
            inv_sqrt = decomp["inv_sqrt"]
            Z = torch.randn((N, d), device=device, dtype=dtype)  # N x d
            # compute y = (Q * inv_sqrt.unsqueeze(0)) @ (Q.t() @ Z.T)  -> d x N then transpose
            temp = Q.t() @ Z.T  # d x N
            y = ((Q * inv_sqrt.unsqueeze(0)) @ temp).T  # N x d

        # compute w = mu_k + y  (mu_k shape (d,))
        mu_k = _bayes_router.mu[k].to(device=device, dtype=dtype)
        w = mu_k.unsqueeze(0) + y  # N x d

        # compute per-sample dot(w_j, x_j)
        # elementwise multiply and sum over dim
        scores[k] = (w * ctxs).sum(dim=1)

    # Now scores is K x N, choose argmax over rows per column
    chosen = torch.argmax(scores, dim=0)  # length N
    return [int(x) for x in chosen.tolist()]


# def rpc_update_arm(arm, context, loss):
#     ctx = context.flatten()
#     k = int(arm)
#     loss_f = float(loss)
#
#     global _arm_baselines_mean, _arm_baselines_sq, _arm_baseline_alpha, _arm_baseline_clip_z, _arm_baseline_eps
#     global _arm_counts, _arm_min_count
#
#     if _arm_baselines_mean is None:
#         raise RuntimeError(...)
#
#     old_mean = _arm_baselines_mean[k]
#     old_sq = _arm_baselines_sq[k]
#
#     # --- If first observation for this arm: init baseline and SKIP bayes update (preserve original behavior) ---
#     if math.isnan(old_mean):
#         _arm_baselines_mean[k] = loss_f
#         _arm_baselines_sq[k] = loss_f * loss_f
#         # initialize count to 1 for this arm
#         if _arm_counts is None:
#             _arm_counts = [0] * len(_arm_baselines_mean)
#         _arm_counts[k] = 1
#         print(f"[rpc_update_arm] arm={k} first observation: init baseline to {loss_f:.4f}; skipping bayes update.",
#               flush=True)
#         return
#
#     # --- Ensure counts structure exists (safety) ---
#     if _arm_counts is None:
#         _arm_counts = [0] * len(_arm_baselines_mean)
#
#     # increment sample count for this arm
#     _arm_counts[k] += 1
#
#     # ---------- If not enough samples observed yet, only update baseline and skip bayes update ----------
#     if _arm_counts[k] < _arm_min_count:
#         a = _arm_baseline_alpha
#         _arm_baselines_mean[k] = (1.0 - a) * old_mean + a * loss_f
#         _arm_baselines_sq[k] = (1.0 - a) * old_sq + a * (loss_f * loss_f)
#         print(
#             f"[rpc_update_arm] arm={k} sample_count={_arm_counts[k]} < min_count({_arm_min_count}), baseline updated only.",
#             flush=True)
#         return
#     # ---------- end min_count guard ----------
#
#     # 以下为原有逻辑（正常计算 z, 更新 EMA, 再调用 _bayes_router.update）
#     raw = old_mean - loss_f
#     var = max(old_sq - old_mean * old_mean, 0.0)
#     std = math.sqrt(var + _arm_baseline_eps)
#     z = raw / std
#     z = max(min(z, _arm_baseline_clip_z), -_arm_baseline_clip_z)
#     reward = float(z)
#     a = _arm_baseline_alpha
#     _arm_baselines_mean[k] = (1.0 - a) * old_mean + a * loss_f
#     _arm_baselines_sq[k] = (1.0 - a) * old_sq + a * (loss_f * loss_f)
#
#     # ensure ctx dtype/device same as _bayes_router
#     if hasattr(_bayes_router, "mu"):
#         try:
#             ctx = ctx.to(_bayes_router.mu.device).to(_bayes_router.mu.dtype)
#         except Exception:
#             pass
#
#     # print(f'===reward: {reward}', flush=True)
#     _bayes_router.update(k, ctx, float(reward), normalized=True)
#     # _bayes_router.update(k, ctx, -loss_f, normalized=True)
#     # invalidate cache for arm k
#     try:
#         if hasattr(_bayes_router, "_decomp_cache"):
#             # with _bayes_router._decomp_lock:
#             _bayes_router._decomp_cache[k] = None
#     except Exception:
#         pass


# def rpc_update_arm(arm, context, loss_or_reward, is_reward: bool = False):
#     """
#     如果 is_reward==False: 保持原有语义，loss_or_reward 是 per-pair loss（旧逻辑）
#     如果 is_reward==True: loss_or_reward 被视为 already-normalized reward (higher better),
#                         跳过 baseline/z-score 流程，直接调用 _bayes_router.update(k, ctx, reward, normalized=True)
#     """
#     ctx = context.flatten()
#     k = int(arm)
#
#     global _arm_baselines_mean, _arm_baselines_sq, _arm_baseline_alpha, _arm_baseline_clip_z, _arm_baseline_eps
#     global _arm_counts, _arm_min_count
#
#     if is_reward:
#         reward = float(loss_or_reward)
#         # 最小裁剪（用你现有的 clip 值）
#         try:
#             clip_val = float(_arm_baseline_clip_z)
#         except Exception:
#             clip_val = 2.0
#         reward = max(min(reward, clip_val), -clip_val)
#
#         # ensure ctx dtype/device same as _bayes_router
#         if hasattr(_bayes_router, "mu"):
#             try:
#                 ctx = ctx.to(_bayes_router.mu.device).to(_bayes_router.mu.dtype)
#             except Exception:
#                 pass
#
#         # 直接更新 router（normalized=True 表示传入的 reward 已经是可用尺度）
#         _bayes_router.update(k, ctx, float(reward), normalized=True)
#
#         # invalidate cache for arm k
#         try:
#             if hasattr(_bayes_router, "_decomp_cache"):
#                 _bayes_router._decomp_cache[k] = None
#         except Exception:
#             pass
#         # 最小日志
#         print(f"[rpc_update_arm] arm={k} got ADV reward={reward:.4f} (is_reward=True).", flush=True)
#         return
#
#     # ------------------- 原有逻辑（不变），loss_or_reward 表示 per-pair loss -------------------
#     loss_f = float(loss_or_reward)
#
#     if _arm_baselines_mean is None:
#         raise RuntimeError(...)
#
#     old_mean = _arm_baselines_mean[k]
#     old_sq = _arm_baselines_sq[k]
#
#     # --- If first observation for this arm: init baseline and SKIP bayes update (preserve original behavior) ---
#     if math.isnan(old_mean):
#         _arm_baselines_mean[k] = loss_f
#         _arm_baselines_sq[k] = loss_f * loss_f
#         # initialize count to 1 for this arm
#         if _arm_counts is None:
#             _arm_counts = [0] * len(_arm_baselines_mean)
#         _arm_counts[k] = 1
#         print(f"[rpc_update_arm] arm={k} first observation: init baseline to {loss_f:.4f}; skipping bayes update.",
#               flush=True)
#         return
#
#     # --- Ensure counts structure exists (safety) ---
#     if _arm_counts is None:
#         _arm_counts = [0] * len(_arm_baselines_mean)
#
#     # increment sample count for this arm
#     _arm_counts[k] += 1
#
#     # ---------- If not enough samples observed yet, only update baseline and skip bayes update ----------
#     if _arm_counts[k] < _arm_min_count:
#         a = _arm_baseline_alpha
#         _arm_baselines_mean[k] = (1.0 - a) * old_mean + a * loss_f
#         _arm_baselines_sq[k] = (1.0 - a) * old_sq + a * (loss_f * loss_f)
#         print(
#             f"[rpc_update_arm] arm={k} sample_count={_arm_counts[k]} < min_count({_arm_min_count}), baseline updated only.",
#             flush=True)
#         return
#     # ---------- end min_count guard ----------
#
#     # 以下为原有逻辑（正常计算 z, 更新 EMA, 再调用 _bayes_router.update）
#     raw = old_mean - loss_f
#     var = max(old_sq - old_mean * old_mean, 0.0)
#     std = math.sqrt(var + _arm_baseline_eps)
#     z = raw / std
#     z = max(min(z, _arm_baseline_clip_z), -_arm_baseline_clip_z)
#     reward = float(z)
#     a = _arm_baseline_alpha
#     _arm_baselines_mean[k] = (1.0 - a) * old_mean + a * loss_f
#     _arm_baselines_sq[k] = (1.0 - a) * old_sq + a * (loss_f * loss_f)
#
#     # ensure ctx dtype/device same as _bayes_router
#     if hasattr(_bayes_router, "mu"):
#         try:
#             ctx = ctx.to(_bayes_router.mu.device).to(_bayes_router.mu.dtype)
#         except Exception:
#             pass
#
#     # _bayes_router.update(k, ctx, float(reward), normalized=True)
#
#     _bayes_router.update(k, ctx, float(-loss_f), normalized=False)
#
#     # invalidate cache for arm k
#     try:
#         if hasattr(_bayes_router, "_decomp_cache"):
#             _bayes_router._decomp_cache[k] = None
#     except Exception:
#         pass


# def rpc_update_arm(arm, context, loss_or_reward):
#     ctx = context.flatten()
#     k = int(arm)
#     loss_f = float(loss_or_reward)
#     # ensure ctx dtype/device same as _bayes_router
#     if hasattr(_bayes_router, "mu"):
#         try:
#             ctx = ctx.to(_bayes_router.mu.device).to(_bayes_router.mu.dtype)
#         except Exception:
#             pass
#     _bayes_router.update(k, ctx, float(-loss_f), normalized=False)
#
#     # invalidate cache for arm k
#     try:
#         if hasattr(_bayes_router, "_decomp_cache"):
#             _bayes_router._decomp_cache[k] = None
#     except Exception:
#         pass


def rpc_update_arm(arm, context, loss_or_reward, is_reward: bool = False):
    """
    If is_reward is False (default), the third arg is interpreted as a loss and
    we convert to reward = -loss (legacy behavior).
    If is_reward is True, the third arg is interpreted directly as reward (advantage).
    """
    ctx = context.flatten()
    k = int(arm)
    val = float(loss_or_reward)
    # interpret val as reward or convert from loss
    if is_reward:
        reward = val
    else:
        reward = -val

    # ensure ctx dtype/device same as _bayes_router
    if hasattr(_bayes_router, "mu"):
        try:
            ctx = ctx.to(_bayes_router.mu.device).to(_bayes_router.mu.dtype)
        except Exception:
            pass
    _bayes_router.update(k, ctx, reward, normalized=False)

    # invalidate cache for arm k
    try:
        if hasattr(_bayes_router, "_decomp_cache"):
            _bayes_router._decomp_cache[k] = None
    except Exception:
        pass



def rpc_init_offline_router(model_path: str,
                            num_reward_models: int,
                            device: torch.device = torch.device("cuda:0")):
    global _offline_router_model, _offline_router_tokenizer
    _offline_router_tokenizer = get_tokenizer(model_path)
    _offline_config = AutoConfig.from_pretrained(os.path.join(model_path, "checkpoint-3000"))
    _offline_router_model = RewardDiffPredictor.from_pretrained(
        os.path.join(model_path, "checkpoint-3000"),  # 这里之前没改？
        config=_offline_config,
        num_reward_models=num_reward_models,
        trust_remote_code=True
    ).to(device)
    _offline_router_model.eval()
    rm_emb_norm = _offline_router_model.get_rm_embeddings().cpu()
    return rm_emb_norm.cpu()


def rpc_offline_batch_encode(queries, resp_as, resp_bs, chunk_size: int = 16):
    """
    在 rank0 执行：把大输入分成 chunk，逐块在 _offline_router_model 上计算 embedding，
    每块用 no_grad，完成后把 emb.cpu() 存回，并释放 cuda cache。
    返回: Tensor (N, D) on CPU
    """
    total = len(queries)
    if total == 0:
        return torch.empty((0, _offline_router_model.get_embedding_dim()), dtype=torch.float32)

    out_chunks = []
    # ensure model.device is defined
    dev = _offline_router_model.device
    for i in range(0, total, chunk_size):
        q_chunk = queries[i:i + chunk_size]
        a_chunk = resp_as[i:i + chunk_size]
        b_chunk = resp_bs[i:i + chunk_size]
        with torch.no_grad():
            emb_chunk = _offline_router_model.batch_encode(
                q_chunk, a_chunk, b_chunk, _offline_router_tokenizer,
                max_length=1024, device=dev
            )
        out_chunks.append(emb_chunk.cpu())
        # free CUDA cache proactively
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(out_chunks, dim=0)


def rpc_offline_batch_logits(embeddings: torch.Tensor, chunk_size: int = 16):
    """
    embeddings: Tensor on CPU (N, D) — 分块把数据送回 model 获取 logits（N, K）
    """
    if embeddings.numel() == 0:
        return torch.empty((0, _offline_router_model.num_reward_models), dtype=torch.float32)
    emb_t = embeddings  # CPU
    dev = _offline_router_model.device
    outs = []
    N = emb_t.shape[0]
    for i in range(0, N, chunk_size):
        chunk = emb_t[i:i + chunk_size].to(dev)
        with torch.no_grad():
            logits_chunk = _offline_router_model.get_bt_logits_from_embedding(chunk)
        outs.append(logits_chunk.cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(outs, dim=0)

