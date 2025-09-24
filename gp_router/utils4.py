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


# class DatasetManagerTraining:
#     def __init__(self, prompts_path, test_size=0.2, dev_size=0.25):
#         self.prompts_path = prompts_path
#         self.test_size = test_size
#         self.dev_size = dev_size
#         self.datasets = {}
#         self.load_and_split_datasets()
#
#     def load_and_split_datasets(self):
#         # Load the dataset from JSON file
#         with open(self.prompts_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         random.shuffle(data)
#         data_test = data[-10:]
#         data_train_val = data[:80]
#
#         train_data, dev_data = train_test_split(
#             data_train_val,
#             test_size=self.dev_size,
#             random_state=42
#         )
#         final_dataset = {'train': train_data, 'dev': dev_data, 'test': data_test}
#         self.datasets['iter3-20k'] = final_dataset
#
#     def get_train_data(self, dataset_name):
#         """Returns the training dataset for the specified dataset."""
#         return self.datasets[dataset_name]['train']
#
#     def get_dev_data(self, dataset_name):
#         """Returns the development (dev) dataset for the specified dataset."""
#         return self.datasets[dataset_name]['dev']
#
#     def get_test_data(self, dataset_name):
#         """Returns the test dataset for the specified dataset."""
#         return self.datasets[dataset_name]['test']


class DatasetManagerInstruct:
    def __init__(self, prompts_path, use_alpaca_eval_test=True, hf_config_name="alpaca_eval_gpt4_baseline",
                 test_size=0.2, dev_size=0.25):
        """
        prompts_path: 你的本地 JSON（原来用的）用来生成 train/dev
        use_alpaca_eval_test: 如果 True，用 AlpacaEval-2（HF）作为 test set
        hf_config_name: HF dataset config; AlpacaEval-2 对应 "alpaca_eval_gpt4_baseline"
        """
        self.prompts_path = prompts_path
        self.use_alpaca_eval_test = use_alpaca_eval_test
        self.hf_config_name = hf_config_name
        self.test_size = test_size
        self.dev_size = dev_size
        self.datasets = {}
        self.load_and_split_datasets()

    def load_alpaca_eval_prompts(self):
        # trust_remote_code=True 很重要（dataset 脚本中有自定义代码）
        ds = load_dataset("tatsu-lab/alpaca_eval", name=self.hf_config_name, split="eval",
                          use_auth_token=None,  # 如果需要 HF token, 在环境中设置 HUGGINGFACEHUB_API_TOKEN
                          trust_remote_code=True)
        # dataset entries have 'instruction' (already merged instruction+input when applicable)
        prompts = [ex["instruction"] for ex in ds]
        return prompts

    def load_and_split_datasets(self):
        # load your local prompts (same as before)
        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        random.shuffle(data)

        if self.use_alpaca_eval_test:
            alpaca_prompts = self.load_alpaca_eval_prompts()
            # build test entries in the format alpaca_eval expects: dicts with 'instruction'
            data_test = [{"question": instr} for instr in alpaca_prompts]
            data_train_val = data  # use all your local data for train+dev (or trim if you want)
        else:
            data_test = data[-10:]
            data_train_val = data[:80]

        train_data, dev_data = train_test_split(
            data_train_val,
            test_size=self.dev_size,
            random_state=42
        )
        final_dataset = {'train': train_data[:3000], 'dev': dev_data, 'test': data_test}
        self.datasets['iter3-20k'] = final_dataset

    def get_train_data(self, dataset_name):
        return self.datasets[dataset_name]['train']

    def get_dev_data(self, dataset_name):
        return self.datasets[dataset_name]['dev']

    def get_test_data(self, dataset_name):
        return self.datasets[dataset_name]['test']


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

            self.datasets[name] = {'train': train_examples[:1600], 'dev': dev_examples, 'test': test_examples}

    def get_train_data(self, dataset_name):
        return self.datasets[dataset_name]['train']

    def get_dev_data(self, dataset_name):
        return self.datasets[dataset_name]['dev']

    def get_test_data(self, dataset_name):
        return self.datasets[dataset_name]['test']


# # dataset_manager
# class DatasetManager:
#     def __init__(self, dataset_paths, test_size=0.2, dev_size=0.25):
#         self.dataset_paths = dataset_paths
#         self.test_size = test_size
#         self.dev_size = dev_size
#         self.datasets = {}
#         self.load_and_split_datasets()
#
#     def load_and_split_datasets(self):
#         for dataset_name, dataset_path in self.dataset_paths.items():
#             if not os.path.exists(dataset_path):
#                 raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
#
#             # Load the dataset from JSON file
#             with open(dataset_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#
#             data_test = data[2290:]  # 测试集 = 490条
#             data_train_val = data[:2290]  # 训练+验证集 = 2290条
#             # question一般多长？
#
#             # 2. 使用sklearn划分训练/验证集
#             train_data, dev_data = train_test_split(
#                 data_train_val,
#                 test_size=self.dev_size,  # 例如0.1
#                 random_state=42
#             )
#
#             # 3. 组合最终数据集
#             final_dataset = {'train': train_data[:500], 'dev': dev_data, 'test': data_test}
#             self.datasets[dataset_name] = final_dataset
#
#     def get_train_data(self, dataset_name):
#         """Returns the training dataset for the specified dataset."""
#         return self.datasets[dataset_name]['train']
#
#     def get_dev_data(self, dataset_name):
#         """Returns the development (dev) dataset for the specified dataset."""
#         return self.datasets[dataset_name]['dev']
#
#     def get_test_data(self, dataset_name):
#         """Returns the test dataset for the specified dataset."""
#         return self.datasets[dataset_name]['test']


class BayesianRouter:
    def __init__(self, d, rm_embeddings, sigma2=1.0, lambda0=1.0, tol_eig=1e-10):
        """
        d: 上下文向量维度
        rm_embeddings: (K, d) 预训练嵌入，作为 prior mean
        sigma2: 噪声方差，贝叶斯线性模型的方差
        lambda0: prior precision scalar
        tol_eig: 特征值下界比例，用于数值稳定
        """
        self.d = d
        self.K = rm_embeddings.shape[0]
        self.sigma2 = float(sigma2)
        self.lambda0 = float(lambda0)
        self.tol_eig = float(tol_eig)

        # # 全部用双精度
        # rm = rm_embeddings.double()
        # 暂时禁用先验知识
        rm = torch.zeros_like(rm_embeddings, dtype=torch.float64)

        # 信息形式先验：Λ = λ0 I,  b = λ0 μ0
        I = torch.eye(d, dtype=torch.float64, device=rm.device)
        self.Lambda = I.unsqueeze(0).repeat(self.K, 1, 1) * self.lambda0  # (K,d,d)
        self.b = (self.lambda0 * rm).clone()  # (K,d)
        self.mu = rm.clone()  # (K,d)

        # Welford 在线归一化
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def normalize_reward(self, r: float) -> float:
        self.count += 1
        delta = r - self.mean
        self.mean += delta / self.count
        delta2 = r - self.mean
        self.M2 += delta * delta2
        if self.count < 2:
            return 0.0
        std = (self.M2 / (self.count - 1)) ** 0.5
        return (r - self.mean) / (std + 1e-8)

    def select_arm(self, context: torch.Tensor) -> int:
        x = context.reshape(-1).double()
        scores = torch.empty(self.K, dtype=torch.float64, device=x.device)

        for k in range(self.K):
            # 1) 对称化精度矩阵
            A = 0.5 * (self.Lambda[k] + self.Lambda[k].mT)
            # 2) 谱分解
            eig, Q = torch.linalg.eigh(A)
            # 3) 下界
            eps = self.tol_eig * eig.max()
            eig_clamped = torch.clamp(eig, min=eps)
            # 4) 采样：w = μ + Q diag(1/sqrt(eig_clamped)) z
            z = torch.randn(self.d, device=x.device, dtype=torch.float64)
            inv_sqrt = eig_clamped.rsqrt()  # 1/sqrt(lambda_i)
            y = (Q * inv_sqrt.unsqueeze(0)) @ (Q.t() @ z)
            w = self.mu[k] + y
            scores[k] = w.dot(x)

        return int(scores.argmax().item())

    def update(self, k: int, context: torch.Tensor, reward: float, normalized: bool = False):
        """
        如果 normalized=True，就直接把 reward 当作已经标准化（z-score）的数使用，
        否则再通过内部 normalize_reward 做（目前我们推荐尽量传 normalized=True）。
        """
        if not normalized:
            r_norm = self.normalize_reward(reward)
        else:
            # 直接使用（并保证为 float64/float32 与 self.mu 对齐）
            r_norm = float(reward)

        x = context.reshape(-1).to(self.mu.device).to(self.mu.dtype)
        # 信息形式更新（与原逻辑一致）
        self.Lambda[k] += torch.ger(x, x) / self.sigma2
        self.b[k] += (r_norm / self.sigma2) * x

        # 重算后验均值 μ = Λ^{-1} b （保持你原来的谱分解方法或改为更稳定的 cholesky）
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


def generate_offline_prior(query_response_pairs):
    """
    输入: query_response_pairs = [(q, resp), ...] where resp are candidate responses (strings).
    输出:
      - queries, resp_as, resp_bs: 列表，表示每一个 candidate pair (q, a, b)
      - batch_emb: torch.Tensor shape (N, D)  —— encoder 为每个 pair 生成的 embedding（CPU）
      - batch_logits: torch.Tensor shape (N, K) —— offline-router 对每对的 K 个 rm logits（CPU）
    说明：
      - 这个函数仍然按 query 生成所有 i<j 的 candidate pairs（为后续由 router 为每个 pair 选 arm）
      - 不再把所有 pair 的平均 embedding 当作 batch context 返回（那会导致单一 RM），
        而是返回每对的 embedding 以便为每对选 RM。
    """
    # 先按 query 分组
    grouped = defaultdict(list)
    for q, resp in query_response_pairs:
        grouped[q].append(resp)

    # 收集所有 (q, a, b) 三元组（i < j）
    queries, resp_as, resp_bs = [], [], []
    for q, resp_list in grouped.items():
        L = len(resp_list)
        for i in range(L):
            for j in range(i + 1, L):
                queries.append(q)
                resp_as.append(resp_list[i])
                resp_bs.append(resp_list[j])

    if len(queries) == 0:
        # no candidate pairs
        return [], [], [], torch.empty(0), torch.empty(0)

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

    return queries, resp_as, resp_bs, batch_emb, batch_logits


# def generate_preference_pairs(reward_model, query_response_pairs, pairs_per_query=16, tie_eps=1e-3):
#     grouped = defaultdict(list)
#     for query, response in query_response_pairs:
#         grouped[query].append(response)
#
#     pairs = []
#     for query, responses in grouped.items():
#         # 去重（按文本），保留顺序
#         uniq = []
#         seen = set()
#         for r in responses:
#             k = r.strip()
#             if k not in seen:
#                 seen.add(k)
#                 uniq.append(r)
#         n = len(uniq)
#         if n < 2:
#             continue
#
#         scores = reward_model.batch_score([query] * n, uniq)
#         # build all unordered pairs (i<j)
#         candidate_pairs = []
#         for i, j in combinations(range(n), 2):
#             si, sj = scores[i], scores[j]
#             if abs(si - sj) <= tie_eps:
#                 continue  # 平局跳过
#             if si > sj:
#                 winner, loser = uniq[i], uniq[j]
#             else:
#                 winner, loser = uniq[j], uniq[i]
#             candidate_pairs.append((query, winner, loser))
#         if not candidate_pairs:
#             continue
#         k = min(pairs_per_query, len(candidate_pairs))
#         sampled = random.sample(candidate_pairs, k)
#         pairs.extend(sampled)
#     return pairs


def generate_preference_pairs(reward_model,
                              query_response_pairs,
                              pairs_per_query=1,
                              tie_eps=1e-2,
                              max_trials_per_pair=20,
                              rng_seed=None):
    """
    新策略（批量评分 + 随机尝试构造 pair）：
    - query_response_pairs: list of (query, response) -- 可包含同一 query 的多条 response（通常来自 generate_responses）
    - pairs_per_query: 每个 question 希望构造的最大偏好对数量（默认 1）
    - tie_eps: 两个 response 分数差小于等于该阈值视为平局，需再采样
    - max_trials_per_pair: 单个 question 为构造一个 pair 最多尝试次数（防止死循环）
    - rng_seed: 可选随机种子，便于复现
    返回: list of (query, preferred_response, less_preferred_response)
    """
    # 1) group responses by query, preserve order and de-duplicate textually
    grouped = defaultdict(list)
    for q, r in query_response_pairs:
        grouped[q].append(r)

    uniq_grouped = {}
    for q, responses in grouped.items():
        uniq = []
        seen = set()
        for r in responses:
            k = r.strip()
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        if len(uniq) >= 2:
            uniq_grouped[q] = uniq

    if not uniq_grouped:
        return []

    # 2) prepare one big batch to score all (query, uniq_response) pairs at once
    all_queries = []
    all_responses = []
    idx_info = []  # (query, start_index, n_responses_for_query)
    for q, uniq in uniq_grouped.items():
        start = len(all_queries)
        all_queries.extend([q] * len(uniq))
        all_responses.extend(uniq)
        idx_info.append((q, start, len(uniq)))

    # call reward model once for the whole batch (efficient for RPC/proxy)
    scores = reward_model.batch_score(all_queries, all_responses)  # list of floats, aligned to all_responses
    # safety: ensure it's a list
    scores = list(scores)

    rng = random.Random(rng_seed)

    pairs = []
    for q, start, n in idx_info:
        uniq = uniq_grouped[q]
        score_slice = scores[start:start + n]
        indices = list(range(n))

        produced = 0
        trials = 0
        while produced < pairs_per_query and trials < max_trials_per_pair:
            trials += 1
            # sample two distinct candidates
            if n < 2:
                break
            i, j = rng.sample(indices, 2)
            si, sj = score_slice[i], score_slice[j]
            if abs(si - sj) > tie_eps:
                # we have a decisive pair
                if si > sj:
                    pairs.append((q, uniq[i], uniq[j]))
                else:
                    pairs.append((q, uniq[j], uniq[i]))
                produced += 1
                continue

            # otherwise，尝试采一个第三条来和 i/j 中的一个构成对（如果存在 third）
            if n >= 3:
                remaining = [x for x in indices if x not in (i, j)]
                k = rng.choice(remaining)
                # try (i, k) then (j, k)
                if abs(score_slice[i] - score_slice[k]) > tie_eps:
                    if score_slice[i] > score_slice[k]:
                        pairs.append((q, uniq[i], uniq[k]))
                    else:
                        pairs.append((q, uniq[k], uniq[i]))
                    produced += 1
                    continue
                if abs(score_slice[j] - score_slice[k]) > tie_eps:
                    if score_slice[j] > score_slice[k]:
                        pairs.append((q, uniq[j], uniq[k]))
                    else:
                        pairs.append((q, uniq[k], uniq[j]))
                    produced += 1
                    continue

            # 如果没有找到，循环继续（直到达到 max_trials_per_pair）
        # end while
    # end for each question
    return pairs


# Rank0: single RPC handler that accepts per-rm unique lists and scores each rm's list
def rpc_batch_score_per_rm(per_rm_qs, per_rm_rs, rm_indices, chunk_size: int = 16):
    """
    per_rm_qs: list of lists, per_rm_qs[j] is list of queries for rm_indices[j]
    per_rm_rs: list of lists, per_rm_rs[j] is list of responses for rm_indices[j]
    rm_indices: list of rm indices (integers)
    Returns: results: list of lists: results[j] is list of floats = scores for per_rm_qs[j]/per_rm_rs[j]
    """
    global _reward_models   #, _rpc_forward_lock
    K = len(rm_indices)
    results = []
    for j, rm_idx in enumerate(rm_indices):
        q_list = per_rm_qs[j]
        r_list = per_rm_rs[j]
        Mj = len(q_list)
        if Mj == 0:
            results.append([])
            continue
        scores_j = []
        # score in chunks
        for s in range(0, Mj, chunk_size):
            e = min(Mj, s + chunk_size)
            q_chunk = q_list[s:e]
            r_chunk = r_list[s:e]
            # single lock region per chunk (still fine, now chunk is meaningful)
            # with _rpc_forward_lock:
            with torch.inference_mode():
                chunk_scores = _reward_models[rm_idx].batch_score(q_chunk, r_chunk)
                # convert to floats
                scores_j.extend([float(x) for x in chunk_scores])

            # minimal cleanup per chunk (not per small RPC)
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    # optional: don't always empty cache; keep it if memory stable
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
        results.append(scores_j)
    return results


def generate_preference_pairs_multi_rm(reward_models, pair_queries, resp_as, resp_bs, pair_rm_indices,
                                      pairs_per_query=1, tie_eps=1e-2, max_trials_per_query=100,
                                      rng_seed=None, score_chunk=16):
    if not pair_queries:
        return [], [], []

    N = len(pair_queries)
    assert len(resp_as) == N and len(resp_bs) == N and len(pair_rm_indices) == N

    # map rm -> pair indices (those pairs assigned to this rm)
    rm_to_pair_idxs = defaultdict(list)
    for idx, rm_idx in enumerate(pair_rm_indices):
        rm_to_pair_idxs[int(rm_idx)].append(idx)

    # For each rm, build its unique (q,resp) list and mapping from pair idx -> local unique idx
    selected_rm_list = list(rm_to_pair_idxs.keys())
    per_rm_qs = []
    per_rm_rs = []
    # for reconstructing scores: map pair i -> (rm_local_col, local_idx_a, local_idx_b)
    pair_to_rm_local = {}  # i -> (col_j, a_local_idx, b_local_idx)

    for col_j, rm_idx in enumerate(selected_rm_list):
        indices = rm_to_pair_idxs[rm_idx]
        unique_map = {}
        unique_qs = []
        unique_rs = []
        pair_a_uidx = {}
        pair_b_uidx = {}
        for i in indices:
            q = pair_queries[i]
            a = resp_as[i]
            b = resp_bs[i]
            key_a = (q, a)
            if key_a not in unique_map:
                unique_map[key_a] = len(unique_qs)
                unique_qs.append(q)
                unique_rs.append(a)
            pair_a_uidx[i] = unique_map[key_a]

            key_b = (q, b)
            if key_b not in unique_map:
                unique_map[key_b] = len(unique_qs)
                unique_qs.append(q)
                unique_rs.append(b)
            pair_b_uidx[i] = unique_map[key_b]

            # record mapping from pair i -> (rm_column, will fill local idxes after loop)
        per_rm_qs.append(unique_qs)
        per_rm_rs.append(unique_rs)
        # store pair->local indices for these pairs
        for i in indices:
            pair_to_rm_local[i] = (col_j, pair_a_uidx[i], pair_b_uidx[i])

    # If nothing to score:
    if all(len(x) == 0 for x in per_rm_qs):
        return [], [], []

    # Single RPC: rank0 will score each rm's local unique lists and return per-rm lists
    results_per_rm = rpc.rpc_sync(
        to="worker0",
        func=rpc_batch_score_per_rm,
        args=(per_rm_qs, per_rm_rs, selected_rm_list, score_chunk)
    )  # results_per_rm is list of lists, len = len(selected_rm_list)

    # Build score_a / score_b for each original pair index i
    score_a = [None] * N
    score_b = [None] * N
    for i in range(N):
        if i not in pair_to_rm_local:
            # this pair wasn't assigned to any RM? shouldn't happen
            score_a[i] = 0.0
            score_b[i] = 0.0
            continue
        col_j, a_local, b_local = pair_to_rm_local[i]
        score_list = results_per_rm[col_j]
        # safety bounds
        score_a[i] = float(score_list[a_local]) if a_local < len(score_list) else 0.0
        score_b[i] = float(score_list[b_local]) if b_local < len(score_list) else 0.0

    # Now proceed with your original selection logic (q_to_pair_idxs etc.)
    q_to_pair_idxs = defaultdict(list)
    for idx, q in enumerate(pair_queries):
        q_to_pair_idxs[q].append(idx)

    rng = random.Random(rng_seed)
    selected_pairs = []
    selected_pair_indices = []
    selected_pair_rm_indices = []

    for q, idxs in q_to_pair_idxs.items():
        if not idxs:
            continue
        produced = 0
        trials = 0
        while produced < pairs_per_query and trials < max_trials_per_query:
            trials += 1
            i = rng.choice(idxs)
            sa = score_a[i]
            sb = score_b[i]
            if sa is None or sb is None:
                continue
            if abs(sa - sb) > tie_eps:
                if sa > sb:
                    selected_pairs.append((q, resp_as[i], resp_bs[i]))
                else:
                    selected_pairs.append((q, resp_bs[i], resp_as[i]))
                selected_pair_indices.append(i)
                selected_pair_rm_indices.append(int(pair_rm_indices[i]))
                produced += 1
                continue
            # tie -> continue sampling
            continue

    return selected_pairs, selected_pair_indices, selected_pair_rm_indices


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


# def _remote_batch_score(rm_idx, queries, responses):
#     """
#     RPC handler on rank0: invoke real RewardModel.batch_score
#     """
#     model = _reward_models[rm_idx]
#     # print('_remote', flush=True)
#     return model.batch_score(queries, responses)


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
_arm_min_count = 5  # 可以调整为 5/10 等


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


def rpc_update_arm(arm, context, loss):
    ctx = context.flatten()
    k = int(arm)
    loss_f = float(loss)

    global _arm_baselines_mean, _arm_baselines_sq, _arm_baseline_alpha, _arm_baseline_clip_z, _arm_baseline_eps
    global _arm_counts, _arm_min_count

    if _arm_baselines_mean is None:
        raise RuntimeError(...)

    old_mean = _arm_baselines_mean[k]
    old_sq = _arm_baselines_sq[k]

    # --- If first observation for this arm: init baseline and SKIP bayes update (preserve original behavior) ---
    if math.isnan(old_mean):
        _arm_baselines_mean[k] = loss_f
        _arm_baselines_sq[k] = loss_f * loss_f
        # initialize count to 1 for this arm
        if _arm_counts is None:
            _arm_counts = [0] * len(_arm_baselines_mean)
        _arm_counts[k] = 1
        print(f"[rpc_update_arm] arm={k} first observation: init baseline to {loss_f:.4f}; skipping bayes update.",
              flush=True)
        return

    # --- Ensure counts structure exists (safety) ---
    if _arm_counts is None:
        _arm_counts = [0] * len(_arm_baselines_mean)

    # increment sample count for this arm
    _arm_counts[k] += 1

    # ---------- If not enough samples observed yet, only update baseline and skip bayes update ----------
    if _arm_counts[k] < _arm_min_count:
        a = _arm_baseline_alpha
        _arm_baselines_mean[k] = (1.0 - a) * old_mean + a * loss_f
        _arm_baselines_sq[k] = (1.0 - a) * old_sq + a * (loss_f * loss_f)
        print(
            f"[rpc_update_arm] arm={k} sample_count={_arm_counts[k]} < min_count({_arm_min_count}), baseline updated only.",
            flush=True)
        return
    # ---------- end min_count guard ----------

    # 以下为原有逻辑（正常计算 z, 更新 EMA, 再调用 _bayes_router.update）
    raw = old_mean - loss_f
    var = max(old_sq - old_mean * old_mean, 0.0)
    std = math.sqrt(var + _arm_baseline_eps)
    z = raw / std
    z = max(min(z, _arm_baseline_clip_z), -_arm_baseline_clip_z)
    reward = float(z)
    a = _arm_baseline_alpha
    _arm_baselines_mean[k] = (1.0 - a) * old_mean + a * loss_f
    _arm_baselines_sq[k] = (1.0 - a) * old_sq + a * (loss_f * loss_f)

    # ensure ctx dtype/device same as _bayes_router
    if hasattr(_bayes_router, "mu"):
        try:
            ctx = ctx.to(_bayes_router.mu.device).to(_bayes_router.mu.dtype)
        except Exception:
            pass

    # print(f'===reward: {reward}', flush=True)
    _bayes_router.update(k, ctx, float(reward), normalized=True)
    # _bayes_router.update(k, ctx, -loss_f, normalized=True)
    # invalidate cache for arm k
    try:
        if hasattr(_bayes_router, "_decomp_cache"):
            # with _bayes_router._decomp_lock:
            _bayes_router._decomp_cache[k] = None
    except Exception:
        pass


def rpc_init_offline_router(model_path: str,
                            num_reward_models: int,
                            device: torch.device = torch.device("cuda:0")):
    global _offline_router_model, _offline_router_tokenizer
    _offline_router_tokenizer = get_tokenizer(model_path)
    _offline_config = AutoConfig.from_pretrained(os.path.join(model_path, "checkpoint-3500"))
    _offline_router_model = RewardDiffPredictor.from_pretrained(
        os.path.join(model_path, "checkpoint-3500"),  # 这里之前没改？
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

