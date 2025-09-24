import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Model, Qwen2PreTrainedModel, AutoModel, AutoConfig, AutoTokenizer, PreTrainedModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, List
from peft import get_peft_model, LoraConfig, TaskType


@dataclass
class PredictorOutput(ModelOutput):
    score_diffs: torch.FloatTensor = None  # 回归输出
    cls_logits: Optional[torch.FloatTensor] = None  # 分类输出
    bt_scores: Optional[torch.FloatTensor] = None  # BT输出
    loss: Optional[torch.FloatTensor] = None


class RewardDiffPredictor(PreTrainedModel):
    def __init__(self, config, num_reward_models: int):
        super().__init__(config)

        self.backbone = AutoModel.from_config(config)
        self.num_reward_models = num_reward_models

        # reg head
        self.reg_vectors = nn.Parameter(torch.randn(num_reward_models, config.hidden_size))
        self.reg_bias = nn.Parameter(torch.zeros(num_reward_models))
        # cls head
        self.cls_vectors = nn.Parameter(torch.randn(num_reward_models, config.hidden_size))
        self.cls_bias = nn.Parameter(torch.zeros(num_reward_models))
        self.bt_vectors = nn.Parameter(torch.randn(num_reward_models, config.hidden_size))

        self.shared_mlp = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            # nn.Dropout(0.1)
        )

        nn.init.kaiming_uniform_(self.reg_vectors, a=math.sqrt(5))
        nn.init.zeros_(self.reg_bias)
        nn.init.kaiming_uniform_(self.cls_vectors, a=math.sqrt(5))
        nn.init.zeros_(self.cls_bias)
        nn.init.kaiming_uniform_(self.bt_vectors, a=math.sqrt(5))
        # nn.init.zeros_(self.bt_bias)

        # self.criterion_reg = nn.SmoothL1Loss(reduction="mean")
        self.criterion_reg = nn.L1Loss(reduction="mean")
        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.lambda_reg = 0.0
        self.lambda_cls = 0.0
        self.lambda_bt = 1.0

    def load_qwen_backbone(self, model_path: str):
        # backbone = AutoModel.from_pretrained(model_path, config=self.config)
        # lora_config = LoraConfig(
        #     r=8,
        #     lora_alpha=16,
        #     target_modules=["q_proj", "k_proj"],
        #     lora_dropout=0.1,
        #     bias="none",
        #     task_type=TaskType.FEATURE_EXTRACTION
        # )
        # peft_model = get_peft_model(backbone, lora_config)
        # self.backbone = peft_model
        backbone = AutoModel.from_pretrained(model_path, config=self.config)
        self.backbone = backbone

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, D)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        # 计算每个样本最后一个非 PAD token 的索引
        lengths = attention_mask.sum(dim=1)  # (B,) 每个样本有效 token 数
        last_idx = (lengths - 1).clamp(min=0)  # 防止为 0
        # 使用 torch.arange 按行索引
        last_hidden = hidden_states[torch.arange(batch_size, device=hidden_states.device), last_idx]  # (B, D)
        return last_hidden

    # def encode(self, input_ids, attention_mask):
    #     outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
    #     hidden_states = outputs.last_hidden_state  # (B, L, D)
    #     attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
    #     masked_hidden = hidden_states * attention_mask_expanded
    #     sum_hidden = masked_hidden.sum(dim=1)  # (B, D)
    #     sum_mask = attention_mask.sum(dim=1, keepdim=True).to(hidden_states.dtype)  # (B,1)
    #     # avoid division by zero
    #     sum_mask = sum_mask.clamp_min(1.0)
    #     mean_pooled = sum_hidden / sum_mask
    #     return mean_pooled

    def forward(
        self,
        input_ids_1,
        attention_mask_1,
        input_ids_2,
        attention_mask_2,
        score_diff_labels: Optional[torch.Tensor] = None,
        cls_labels: Optional[torch.Tensor] = None,
        bt_pairs: Optional[List[List[int]]] = None,
        labels=None,
    ):
        if labels is not None:
            score_diff_labels, cls_labels, bt_pairs = labels

        h1 = self.encode(input_ids_1, attention_mask_1)
        h2 = self.encode(input_ids_2, attention_mask_2)

        h_mean = (h1 + h2) / 2
        h_diff = torch.abs(h1 - h2)
        h_cat = torch.cat([h_mean, h_diff], dim=-1)
        h_shared = self.shared_mlp(h_cat)
        h_norm = F.normalize(h_shared, p=2, dim=1)  # (B, D)

        reg_vec_norm = F.normalize(self.reg_vectors, p=2, dim=1)
        cls_vec_norm = F.normalize(self.cls_vectors, p=2, dim=1)
        bt_vec_norm = F.normalize(self.bt_vectors, p=2, dim=1)

        score_diffs = torch.matmul(h_norm, reg_vec_norm.t()) + self.reg_bias
        cls_logits = torch.matmul(h_norm, cls_vec_norm.t()) + self.cls_bias
        bt_scores = torch.matmul(h_norm, bt_vec_norm.t())

        # --- 在计算 loss 之前，构造一个依赖于模型输出的零张量，保证计算图不被断开 ---
        zero = h_norm.sum() * 0.0  # = 0.0 but requires_grad True (依赖于 h_shared)
        regression_loss = zero
        cls_loss = zero
        bt_loss = zero
        if score_diff_labels is not None:
            mask_reg = (score_diff_labels != -1000.0)
            if mask_reg.any():
                # regression_loss = F.smooth_l1_loss(score_diffs[mask_reg], score_diff_labels[mask_reg],
                #                                    reduction="mean")
                regression_loss = F.l1_loss(score_diffs[mask_reg], score_diff_labels[mask_reg],
                                            reduction="mean")

        if cls_labels is not None:
            mask_cls = (cls_labels != -1000)
            if mask_cls.any():
                cls_loss = F.binary_cross_entropy_with_logits(cls_logits[mask_cls], cls_labels[mask_cls],
                                                              reduction="mean")

        if bt_pairs is not None:
            bt_pairs = bt_pairs.to(bt_scores.device)
            B, max_pairs, _ = bt_pairs.shape
            valid = (bt_pairs[..., 0] >= 0) & (bt_pairs[..., 1] >= 0)
            if valid.any():
                w_idx = bt_pairs[..., 0].clamp(min=0)
                l_idx = bt_pairs[..., 1].clamp(min=0)
                s_w = bt_scores.gather(1, w_idx)
                s_l = bt_scores.gather(1, l_idx)
                per_pair_loss = F.softplus(-(s_w - s_l))  # (B, P)
                per_pair_loss = per_pair_loss * valid.float()
                cnt = valid.sum(dim=1).clamp(min=1).float()
                per_sample_loss = per_pair_loss.sum(dim=1) / cnt
                bt_loss = per_sample_loss.mean()
            else:
                bt_loss = zero
        else:
            bt_loss = zero

        loss = self.lambda_cls * cls_loss + self.lambda_reg * regression_loss + self.lambda_bt * bt_loss

        output = PredictorOutput(
            score_diffs=score_diffs,
            cls_logits=cls_logits,
            bt_scores=bt_scores,
            loss=loss,
        )
        output.logits = (score_diffs, cls_logits, bt_scores)
        return output

    @torch.no_grad()
    def batch_encode(
            self,
            questions: List[str],
            answers_A: List[str],
            answers_B: List[str],
            tokenizer: AutoTokenizer,
            max_length: int = 1024,
            device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # 设备
        if device is None:
            device = next(self.parameters()).device

        assert len(questions) == len(answers_A) == len(answers_B), \
            "questions、answers_A、answers_B 的长度必须一致。"
        B = len(questions)
        if B == 0:
            return torch.empty(0, self.config.hidden_size, device=device)

        # 与 build_router_input 等价的组装（支持问题串里已含多轮对话的情况）
        def _pair_text(q: str, a: str) -> str:
            q = q.strip()
            a = a.strip()
            # 若问题串里已经包含 <|user|>:/<|assistant|>:，则视作现成对话，仅在末尾追加一条 assistant
            if "<|user|>:" in q or "<|assistant|>:" in q:
                if q and not q.endswith("\n"):
                    q = q + "\n"
                return f"{q}<|assistant|>: {a}"
            # 否则按单轮拼：<|user|> + <|assistant|>
            return f"<|user|>: {q}\n<|assistant|>: {a}"

        texts_A = [_pair_text(q, a) for q, a in zip(questions, answers_A)]
        texts_B = [_pair_text(q, b) for q, b in zip(questions, answers_B)]

        tok_kwargs = dict(
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encoded_1 = tokenizer(texts_A, **tok_kwargs)
        encoded_2 = tokenizer(texts_B, **tok_kwargs)

        input_ids_1 = encoded_1["input_ids"].to(device)
        attention_mask_1 = encoded_1["attention_mask"].to(device)
        input_ids_2 = encoded_2["input_ids"].to(device)
        attention_mask_2 = encoded_2["attention_mask"].to(device)

        # 与 forward 完全一致的两路编码与融合
        h1 = self.encode(input_ids_1, attention_mask_1)  # (B, D)
        h2 = self.encode(input_ids_2, attention_mask_2)  # (B, D)

        h_mean = (h1 + h2) / 2
        h_diff = torch.abs(h1 - h2)
        h_cat = torch.cat([h_mean, h_diff], dim=-1)  # (B, 2D)
        h_shared = self.shared_mlp(h_cat)  # (B, D)
        h_norm = F.normalize(h_shared, p=2, dim=1)  # (B, D)
        return h_norm

    @torch.no_grad()
    def get_bt_logits_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        输入必须是已经 L2 归一化过的 h_norm，例如 batch_encode 的输出。
        返回 (B, K) 的打分。
        """
        bt_vec_norm = F.normalize(self.bt_vectors, p=2, dim=1)  # (K, D)
        device = bt_vec_norm.device

        x = embedding
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, D)
        x = x.to(device=device, dtype=bt_vec_norm.dtype)

        logits = torch.matmul(x, bt_vec_norm.t())  # (B, K)
        return logits

    @torch.no_grad()
    def get_rm_embeddings(self) -> torch.Tensor:
        return F.normalize(self.bt_vectors.detach(), p=2, dim=1)


def get_tokenizer(tokenizer_name, pad_token_if_none="<|pad|>"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # tokenizer.truncation_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({"pad_token": pad_token_if_none})  # 有必要吗？

    return tokenizer
