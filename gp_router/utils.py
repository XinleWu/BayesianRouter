import time
import yaml
import os
import json
import math
import random
from collections import defaultdict
from torch.distributions import MultivariateNormal
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed.rpc as rpc
from offline_model_2encoder_100M import RewardDiffPredictor, get_tokenizer


# config_loader
def load_config(config_file):
    """Load the YAML configuration file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


# dataset_manager
class DatasetManager:
    def __init__(self, dataset_paths, test_size=0.2, dev_size=0.25, prompt_type="reasoning"):
        self.dataset_paths = dataset_paths
        self.test_size = test_size
        self.dev_size = dev_size
        self.prompt_type = prompt_type
        self.datasets = {}
        self.load_and_split_datasets()

    def load_and_split_datasets(self):
        for dataset_name, dataset_path in self.dataset_paths.items():
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

            # # Load the dataset from JSON file
            # breakpoint()
            # dataset = load_dataset("json", data_files=dataset_path)['train']
            #
            # # Split the dataset into train, test, and dev sets
            # dataset = dataset.train_test_split(test_size=self.test_size)
            # dataset['dev'] = dataset['train'].train_test_split(test_size=self.dev_size)['test']
            # self.datasets[dataset_name] = dataset

            # Load the dataset from JSON file
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # # 1. 固定划分测试集（后490条）
            # data_test = data[2290:]  # 测试集 = 490条
            # data_train_val = data[:2290]  # 训练+验证集 = 2290条

            data_test = data[2290:]  # 测试集 = 490条
            data_train_val = data[:600]  # 训练+验证集 = 2290条

            # 2. 使用sklearn划分训练/验证集
            train_data, dev_data = train_test_split(
                data_train_val,
                test_size=self.dev_size,  # 例如0.1
                random_state=42
            )

            # 3. 组合最终数据集
            final_dataset = {'train': train_data, 'dev': dev_data, 'test': data_test}
            self.datasets[dataset_name] = final_dataset

    def get_train_data(self, dataset_name):
        """Returns the training dataset for the specified dataset."""
        return self.datasets[dataset_name]['train']

    def get_dev_data(self, dataset_name):
        """Returns the development (dev) dataset for the specified dataset."""
        return self.datasets[dataset_name]['dev']

    def get_test_data(self, dataset_name):
        """Returns the test dataset for the specified dataset."""
        return self.datasets[dataset_name]['test']


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

    # def update(self, k: int, context: torch.Tensor, reward: float):
    #     r_norm = self.normalize_reward(reward)
    #     x = context.reshape(-1).double()
    #
    #     # 信息形式更新
    #     self.Lambda[k] += torch.ger(x, x) / self.sigma2
    #     self.b[k] += (r_norm / self.sigma2) * x
    #
    #     # 重算后验均值 μ = Λ^{-1} b  用谱分解+下界+三角求解
    #     A = 0.5*(self.Lambda[k] + self.Lambda[k].mT)
    #     eig, Q = torch.linalg.eigh(A)
    #     eps = self.tol_eig * eig.max()
    #     eig_clamped = torch.clamp(eig, min=eps)
    #     inv = 1.0 / eig_clamped
    #     # μ = Q diag(inv) Q^T b
    #     self.mu[k] = (Q * inv.unsqueeze(0)) @ (Q.t() @ self.b[k])

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grad_accum_steps = grad_accum_steps
        self.accum_step = 0  # 当前累积步数计数器
        self.optimizer.zero_grad()  # 初始清空梯度

        # 初始用 policy adapter
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.set_adapter("policy")
            self.model.module.train()
        else:
            self.model.set_adapter("policy")
            self.model.train()

    def _tokenize(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

    def _log_probs(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift logits and labels for LM
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        selected = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        return (selected * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)

    # === NEW: only score answer tokens given the prompt ===
    def _log_probs_answer_only(self, prompts, answers):
        # 1) tokenize joint: prompt+answer
        joint_texts = [p + a for p, a in zip(prompts, answers)]
        joint = self._tokenize(joint_texts)  # to(self.device) inside

        # 2) tokenize prompt alone to get prefix lengths (including special tokens)
        prompt_tok = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        # 用 attention_mask 统计每条样本的有效 token 数
        prefix_lens = prompt_tok["attention_mask"].sum(dim=1)  # (B,)

        # 3) 前向并构造 shift 后的位置
        outputs = self.model(input_ids=joint["input_ids"], attention_mask=joint["attention_mask"])
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = joint["input_ids"][:, 1:]
        shift_mask = joint["attention_mask"][:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        selected = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        # 4) 构造 answer-only 掩码：只保留 labels 属于 answer 的位置
        B, Tm1 = selected.shape
        ans_mask = torch.zeros_like(selected)
        # 对每个样本：从 (prefix_len-1) 开始是 answer 的 labels
        for b in range(B):
            s = int(prefix_lens[b].item()) - 1
            # 保险起见裁剪到合法区间
            s = max(0, min(s, Tm1 - 1))
            ans_mask[b, s:] = 1

        mask = shift_mask * ans_mask
        denom = mask.sum(dim=1).clamp_min(1)  # 防止极端截断导致除 0
        print("DEBUG denom stats", denom.min().item(), denom.median().item(), denom.max().item())
        return (selected * mask).sum(dim=1) / (denom.sqrt())

    # def dpo_loss(self, queries, y_w, y_l):
    #     # Construct full prompt+response strings
    #     prompts_w = [q + r for q, r in zip(queries, y_w)]
    #     prompts_l = [q + r for q, r in zip(queries, y_l)]
    #
    #     if isinstance(self.model, DistributedDataParallel):
    #         self.model.module.set_adapter("policy")
    #     else:
    #         self.model.set_adapter("policy")
    #     inputs_w = self._tokenize(prompts_w)
    #     inputs_l = self._tokenize(prompts_l)
    #     logp_w = self._log_probs(inputs_w["input_ids"], inputs_w["attention_mask"])
    #     logp_l = self._log_probs(inputs_l["input_ids"], inputs_l["attention_mask"])
    #
    #     if isinstance(self.model, DistributedDataParallel):
    #         self.model.module.set_adapter("reference")
    #     else:
    #         self.model.set_adapter("reference")
    #     with torch.no_grad():
    #         ref_logp_w = self._log_probs(inputs_w["input_ids"], inputs_w["attention_mask"])
    #         ref_logp_l = self._log_probs(inputs_l["input_ids"], inputs_l["attention_mask"])
    #
    #     # 切回 policy 以便下一次生成 & 训练
    #     if isinstance(self.model, DistributedDataParallel):
    #         self.model.module.set_adapter("policy")
    #     else:
    #         self.model.set_adapter("policy")
    #
    #     # DPO loss
    #     diff = (logp_w - ref_logp_w) - (logp_l - ref_logp_l)
    #     loss = -torch.log(torch.sigmoid(diff)).mean()
    #     return loss

    def dpo_loss_per_sample(self, queries, y_w, y_l):
        # prompts_w = [q + r for q, r in zip(queries, y_w)]
        # prompts_l = [q + r for q, r in zip(queries, y_l)]
        #
        # if isinstance(self.model, DistributedDataParallel):
        #     self.model.module.set_adapter("policy")
        # else:
        #     self.model.set_adapter("policy")
        #
        # inputs_w = self._tokenize(prompts_w)
        # inputs_l = self._tokenize(prompts_l)
        #
        # logp_w = self._log_probs(inputs_w["input_ids"], inputs_w["attention_mask"])  # (N,)
        # logp_l = self._log_probs(inputs_l["input_ids"], inputs_l["attention_mask"])  # (N,)
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.set_adapter("policy")
        else:
            self.model.set_adapter("policy")
        # === 改为只对 answer tokens 计分 ===
        logp_w = self._log_probs_answer_only(queries, y_w)  # (N,)
        logp_l = self._log_probs_answer_only(queries, y_l)  # (N,)

        if isinstance(self.model, DistributedDataParallel):
            self.model.module.set_adapter("reference")
        else:
            self.model.set_adapter("reference")

        # with torch.no_grad():
        #     ref_logp_w = self._log_probs(inputs_w["input_ids"], inputs_w["attention_mask"])  # (N,)
        #     ref_logp_l = self._log_probs(inputs_l["input_ids"], inputs_l["attention_mask"])  # (N,)
        with torch.no_grad():
            ref_logp_w = self._log_probs_answer_only(queries, y_w)  # (N,)
            ref_logp_l = self._log_probs_answer_only(queries, y_l)  # (N,)

        if isinstance(self.model, DistributedDataParallel):
            self.model.module.set_adapter("policy")
        else:
            self.model.set_adapter("policy")

        diff = (logp_w - ref_logp_w) - (logp_l - ref_logp_l)  # (N,)
        diff = torch.clamp(diff, min=-5.0, max=5.0)
        return -torch.log(torch.sigmoid(diff))  # (N,)

    # def nll_loss(self, y):
    #     inputs = self._tokenize(y)
    #     logp = self._log_probs(inputs["input_ids"], inputs["attention_mask"])
    #     return -logp.mean()

    def nll_loss_per_sample(self, ys_w):
        inputs = self._tokenize(list(ys_w))
        logp = self._log_probs(inputs["input_ids"], inputs["attention_mask"])  # (N,)
        return -logp  # (N,)

    # def compute_preference_loss(self, preference_pairs):
    #     if not preference_pairs:
    #         return 0.0
    #     # Unzip preference triplets
    #     queries, ys_w, ys_l = zip(*preference_pairs)
    #     # DPO loss
    #     loss_dpo = self.dpo_loss(list(queries), list(ys_w), list(ys_l))
    #     # NLL loss on winners
    #     loss_nll = self.nll_loss(list(ys_w))
    #     total_loss = loss_dpo + loss_nll
    #     return total_loss.item()

    # def train_step(self, preference_pairs):
    #     self.model.train()
    #     print(f'len preference pairs: {len(preference_pairs)}', flush=True)
    #
    #     # 将偏好对分成多个微批次，这是为啥？chunk_size只能是1？
    #     total_loss = 0.0
    #     chunk_size = max(1, len(preference_pairs) // self.grad_accum_steps)
    #     chunks = [preference_pairs[i:i+chunk_size] for i in range(0, len(preference_pairs), chunk_size)]
    #     for chunk in chunks:
    #         if not chunk:
    #             continue
    #         queries, ys_w, ys_l = zip(*chunk)
    #         loss_dpo = self.dpo_loss(list(queries), list(ys_w), list(ys_l))
    #         loss_nll = self.nll_loss(list(ys_w))
    #         loss = loss_dpo + loss_nll
    #         scaled_loss = loss / len(chunks)
    #         scaled_loss.backward()
    #         total_loss += loss.item()
    #
    #         self.accum_step += 1
    #         if self.accum_step % self.grad_accum_steps == 0:
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
    #             self.accum_step = 0
    #     return total_loss / len(preference_pairs) * chunk_size

    def train_step(self, preference_pairs):
        self.model.train()
        if not preference_pairs:
            return 0.0, []  # === 改动：同时返回 per-pair losses

        print(f'len preference pairs: {len(preference_pairs)}', flush=True)
        queries, ys_w, ys_l = zip(*preference_pairs)

        # 1) 一次性向量化得到每个样本的 loss
        loss_dpo_all = self.dpo_loss_per_sample(list(queries), list(ys_w), list(ys_l))  # (N,)
        # loss_nll_all = self.nll_loss_per_sample(list(ys_w))  # (N,)
        loss_all = loss_dpo_all  # + loss_nll_all  # (N,)

        # 2) 仍按你原先的 chunk 规则做 backward + grad_accum（语义等价）
        N = len(preference_pairs)
        chunk_size = max(1, N // self.grad_accum_steps)
        total_loss = 0.0

        # 计算实际的 chunk 数，以便判断最后一个 chunk
        n_chunks = math.ceil(N / chunk_size)
        chunk_idx = 0

        for i in range(0, N, chunk_size):
            chunk = loss_all[i:i + chunk_size]  # (m,)
            loss = chunk.mean()  # (scalar)
            # 用 n_chunks 做缩放（与你原来缩放逻辑等价）
            scaled_loss = loss / n_chunks

            # 对除最后一个 chunk 外的 backward 保留计算图
            retain = (chunk_idx != n_chunks - 1)
            scaled_loss.backward(retain_graph=retain)
            total_loss += loss.item()
            self.accum_step += 1
            if self.accum_step % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accum_step = 0
            chunk_idx += 1

        # 3) 返回与原来近似的 avg（保持打印/收敛逻辑），以及逐样本 loss 列表
        avg_loss = total_loss / (N // chunk_size if N % chunk_size == 0 else len(range(0, N, chunk_size)))
        per_pair_losses = [float(x) for x in loss_dpo_all.detach().cpu().tolist()]
        return avg_loss, per_pair_losses


def generate_offline_prior(query_response_pairs):
    # 先按 query 分组
    grouped = defaultdict(list)
    for q, resp in query_response_pairs:
        grouped[q].append(resp)

    # 收集所有 (q, a, b) 三元组
    queries, resp_as, resp_bs = [], [], []
    for q, resp_list in grouped.items():
        L = len(resp_list)
        for i in range(L):
            for j in range(i + 1, L):
                queries.append(q)
                resp_as.append(resp_list[i])
                resp_bs.append(resp_list[j])

    # 1) 一次 rpc 批量 encode
    t1 = time.time()
    batch_emb = rpc.rpc_sync(
        to="worker0",
        func=rpc_offline_batch_encode,
        args=(queries, resp_as, resp_bs)
    )  # 返回 shape (N, D)
    # print(f'batch emb time: {time.time()-t1}', flush=True)

    # 2) 一次 rpc 批量 logits
    batch_logits = rpc.rpc_sync(
        to="worker0",
        func=rpc_offline_batch_logits,
        args=(batch_emb,)
    )  # 返回 shape (N, K)

    # 3) 计算 context
    # 平均所有 embedding，得到 (1, D)
    context = batch_emb.mean(dim=0, keepdim=True)

    # 4) 计算 prior
    probs = torch.sigmoid(batch_logits)  # (N, K)
    prior = (probs > 0.5).float().mean(dim=0, keepdim=True)  # (1, K)

    return context, prior


# def generate_preference_pairs(reward_model, query_response_pairs, pairs_per_query=4, top_k=4, mid_start=8, mid_end=12):
#     """
#     Generate preference pairs using Top-K vs Mid-to-High-K strategy.
#     If reward_model is a ProxyRewardModel, batch_score will be forwarded via RPC.
#     """
#     grouped = defaultdict(list)
#     for query, response in query_response_pairs:
#         grouped[query].append(response)
#
#     pairs = []
#     for query, response_list in grouped.items():
#         if len(response_list) < mid_end:  # 应该不会发生吧？
#             continue
#
#         scores = reward_model.batch_score([query] * len(response_list), response_list)
#         scored = list(zip(response_list, scores))
#         sorted_resps = sorted(scored, key=lambda x: x[1], reverse=True)
#
#         top_candidates = sorted_resps[:top_k]
#         mid_candidates = sorted_resps[mid_start:mid_end]
#         for _ in range(pairs_per_query):
#             top = random.choice(top_candidates)
#             mid = random.choice(mid_candidates)
#             # if top[1] == mid[1]:  # 这种情况应该不多
#             #     continue
#             preferred, less_preferred = (top[0], mid[0]) if top[1] > mid[1] else (mid[0], top[0])
#             pairs.append((query, preferred, less_preferred))
#
#     return pairs


def generate_preference_pairs(
        reward_model,
        query_response_pairs,
        pairs_per_query=4,
        top_k=4,
        mid_start=8,
        mid_end=12,
        return_pairs_only: bool = False  # === NEW: 仅采样，不贴 winner/loser 标签
):
    """
    这个版本逐pair计算reward?
    当 return_pairs_only=True 时，返回形如 [(q, a, b), ...] 的“未贴标签”的pair，
    仍然使用指定 reward_model 的排序来进行 top-mid 采样；否则行为与原来一致，
    返回 [(q, preferred, less_preferred), ...]。
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for query, response in query_response_pairs:
        grouped[query].append(response)

    pairs = []
    for query, response_list in grouped.items():
        if len(response_list) < mid_end:  # 应该不会发生吧？
            continue

        scores = reward_model.batch_score([query] * len(response_list), response_list)
        scored = list(zip(response_list, scores))
        sorted_resps = sorted(scored, key=lambda x: x[1], reverse=True)

        top_candidates = sorted_resps[:top_k]
        mid_candidates = sorted_resps[mid_start:mid_end]

        for _ in range(pairs_per_query):  # 是否会导致重复？？？要不要把batch size搞大点？
            top = random.choice(top_candidates)
            mid = random.choice(mid_candidates)
            if return_pairs_only:
                # 只返回 (q, a, b)，不决定谁赢
                pairs.append((query, top[0], mid[0]))
            else:
                preferred, less_preferred = (top[0], mid[0]) if top[1] > mid[1] else (mid[0], top[0])
                pairs.append((query, preferred, less_preferred))

    return pairs


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

        with torch.no_grad():
            batch_rewards = self.model(inputs['input_ids'],
                                       attention_mask=inputs["attention_mask"])[0].cpu()

        return [reward.item() for reward in batch_rewards]


class ProxyRewardModel:
    """Client-side proxy: forwards batch_score requests via RPC to rank0."""

    def __init__(self, idx):
        self.idx = idx

    def batch_score(self, queries, responses):
        # Remote call to server process
        return rpc.rpc_sync(
            to="worker0",
            func=_remote_batch_score,
            args=(self.idx, queries, responses)
        )


def _remote_batch_score(rm_idx, queries, responses):
    """
    RPC handler on rank0: invoke real RewardModel.batch_score
    """
    model = _reward_models[rm_idx]
    return model.batch_score(queries, responses)


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

    # per-arm EMA baseline init
    _arm_baseline_alpha = float(baseline_alpha)
    _arm_baselines_mean = [float('nan')] * num_reward_models
    _arm_baselines_sq = [float('nan')] * num_reward_models
    # 初始化每臂样本计数，供 min_count 逻辑使用
    global _arm_counts
    _arm_counts = [0] * num_reward_models
    print(f"[Rank0] Initialized {_bayes_router.K} arm baselines (alpha={_arm_baseline_alpha}).")


def rpc_select_arm(context, prior=None, beta=0.0):
    ctx = context.flatten()
    return _bayes_router.select_arm(ctx)
    # pri = prior.squeeze(0)
    # return int(torch.argmax(pri).item())


# def rpc_update_arm(arm, context, reward):
#     ctx = context.flatten()
#     _bayes_router.update(arm, ctx, reward)

# def rpc_update_arm(arm, context, loss):  # online版本
#     """
#     在 rank0：
#     - 计算 per-arm raw = old_mean - loss（若 old_mean 未初始化，则 reward=0）
#     - 用 per-arm EMA 二阶矩估计 std -> z = raw / std
#     - clip z -> reward = z
#     - 更新 per-arm EMA mean & sq（用 loss）
#     - 把 reward（已标准化）传给 _bayes_router.update(..., normalized=True)
#     """
#     ctx = context.flatten()
#     k = int(arm)
#     loss_f = float(loss)
#
#     global _arm_baselines_mean, _arm_baselines_sq, _arm_baseline_alpha, _arm_baseline_clip_z, _arm_baseline_eps
#     if _arm_baselines_mean is None:
#         raise RuntimeError("Arm baselines not initialized; call rpc_init_globals first on rank0.")
#
#     old_mean = _arm_baselines_mean[k]
#     old_sq = _arm_baselines_sq[k]
#
#     if math.isnan(old_mean):
#         # 首次观测：初始化均值和二阶矩，reward 设为 0
#         reward = 0.0
#         _arm_baselines_mean[k] = loss_f
#         _arm_baselines_sq[k] = loss_f * loss_f
#     else:
#         raw = old_mean - loss_f  # positive => better than history
#         # 估计方差（基于 EMA second moment）
#         var = max(old_sq - old_mean * old_mean, 0.0)
#         std = math.sqrt(var + _arm_baseline_eps)
#         z = raw / std
#         # clip z
#         z = max(min(z, _arm_baseline_clip_z), -_arm_baseline_clip_z)
#         reward = float(z)
#
#         # 更新 EMA（对原始 loss 做 EMA）
#         a = _arm_baseline_alpha
#         _arm_baselines_mean[k] = (1.0 - a) * old_mean + a * loss_f
#         _arm_baselines_sq[k] = (1.0 - a) * old_sq + a * (loss_f * loss_f)
#
#     # 确保 context 在和 _bayes_router 相同的 device/dtype
#     if hasattr(_bayes_router, "mu"):
#         try:
#             ctx = ctx.to(_bayes_router.mu.device).to(_bayes_router.mu.dtype)
#         except Exception:
#             # 如果 ctx 是 numpy/CPU tensor，这里可能不必要，但包个 try 以稳健
#             pass
#
#     # 把 reward（已经是标准化 z）送入 Bayes router，并告诉它 reward 已经标准化
#     # 这里假定我们已给 BayesianRouter.update 增加 normalized 参数（见下面）
#     print(f'===reward: {reward}', flush=True)
#     _bayes_router.update(k, ctx, float(reward), normalized=True)


# def rpc_update_arm(arm, context, loss):  # offline+online版本
#     ctx = context.flatten()
#     k = int(arm)
#     loss_f = float(loss)
#
#     global _arm_baselines_mean, _arm_baselines_sq, _arm_baseline_alpha, _arm_baseline_clip_z, _arm_baseline_eps
#     if _arm_baselines_mean is None:
#         raise RuntimeError(...)
#
#     old_mean = _arm_baselines_mean[k]
#     old_sq = _arm_baselines_sq[k]
#
#     # ---------- NEW: if first obs, only init baselines and SKIP bayes update ----------
#     if math.isnan(old_mean):
#         # 初始化 EMA baseline，但不要用 reward=0 去更新 bayes（避免只增 Λ）
#         _arm_baselines_mean[k] = loss_f
#         _arm_baselines_sq[k] = loss_f * loss_f
#         print(f"[rpc_update_arm] arm={k} first observation: init baseline to {loss_f:.4f}; skipping bayes update.", flush=True)
#         return
#     # ---------- end NEW ----------
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
#     print(f'===reward: {reward}', flush=True)
#     _bayes_router.update(k, ctx, float(reward), normalized=True)


def rpc_update_arm(arm, context, loss):  # offline+online，且有min_count版本
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

    print(f'===reward: {reward}', flush=True)
    _bayes_router.update(k, ctx, float(reward), normalized=True)


def rpc_init_offline_router(model_path: str,
                            num_reward_models: int,
                            device: torch.device = torch.device("cuda:0")):
    global _offline_router_model, _offline_router_tokenizer
    _offline_router_tokenizer = get_tokenizer(model_path)
    _offline_config = AutoConfig.from_pretrained(model_path)
    _offline_router_model = RewardDiffPredictor.from_pretrained(
        model_path,
        config=_offline_config,
        num_reward_models=num_reward_models,
        trust_remote_code=True
    ).to(device)
    _offline_router_model.eval()
    rm_emb_norm = _offline_router_model.get_rm_embeddings().cpu()
    return rm_emb_norm.cpu()


def rpc_offline_batch_encode(queries, resp_as, resp_bs):
    emb = _offline_router_model.batch_encode(
        queries, resp_as, resp_bs,
        _offline_router_tokenizer,
        max_length=1024,  # 注意1024
        device=_offline_router_model.device
    )
    return emb.cpu()


def rpc_offline_batch_logits(embeddings: torch.Tensor):
    emb_t = embeddings.to(_offline_router_model.device)
    logits = _offline_router_model.get_bt_logits_from_embedding(emb_t)
    return logits.cpu()
