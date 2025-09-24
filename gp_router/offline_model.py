import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Model, Qwen2PreTrainedModel, AutoTokenizer
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PredictorOutput(ModelOutput):
    score_diffs: torch.FloatTensor = None  # 回归输出
    direction_logits: Optional[torch.FloatTensor] = None  # 分类输出
    loss: Optional[torch.FloatTensor] = None


class RewardDiffPredictor(Qwen2PreTrainedModel):
    def __init__(self, config, num_reward_models: int):
        super().__init__(config)
        self.qwen = Qwen2Model(config)
        self.num_reward_models = num_reward_models

        # 每个 RM 学习一个回归向量和一个偏置
        self.reg_vectors = nn.Parameter(torch.randn(num_reward_models, config.hidden_size))
        self.reg_bias = nn.Parameter(torch.zeros(num_reward_models))
        # 每个 RM 学习一个分类向量和一个偏置
        self.cls_vectors = nn.Parameter(torch.randn(num_reward_models, config.hidden_size))
        self.cls_bias = nn.Parameter(torch.zeros(num_reward_models))

        nn.init.kaiming_uniform_(self.reg_vectors, a=math.sqrt(5))
        nn.init.zeros_(self.reg_bias)
        nn.init.kaiming_uniform_(self.cls_vectors, a=math.sqrt(5))  # 初始化是导致性能下降的原因吗？
        nn.init.zeros_(self.cls_bias)

        # 损失函数
        self.criterion_reg = nn.SmoothL1Loss(reduction="mean")
        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.lambda_cls = 1.0

    def load_qwen_backbone(self, model_path: str):
        qwen = Qwen2Model.from_pretrained(model_path)
        self.qwen.load_state_dict(qwen.state_dict())

    def get_input_embeddings(self):
        return self.qwen.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.qwen.set_input_embeddings(new_embeddings)

    # def forward(
    #     self,
    #     input_ids: torch.Tensor,
    #     attention_mask: torch.Tensor,
    #     score_diff_labels: Optional[torch.Tensor] = None,
    #     cls_labels: Optional[torch.Tensor] = None,
    # ):
    #     outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
    #     hidden_states = outputs.last_hidden_state  # (B, T, D)
    #
    #     # mean pooling：
    #     attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())  # (B, T, D)
    #     masked_hidden = hidden_states * attention_mask_expanded  # (B, T, D)
    #     sum_hidden = masked_hidden.sum(dim=1)  # (B, D)
    #     sum_mask = attention_mask.sum(dim=1, keepdim=True)  # (B, 1)
    #     mean_pooled = sum_hidden / sum_mask  # (B, D)
    #
    #     # # 对 RM embeddings 做 L2 归一化
    #     reg_vec_norm = F.normalize(self.reg_vectors, p=2, dim=1)  # (K, D)
    #     cls_vec_norm = F.normalize(self.cls_vectors, p=2, dim=1)  # (K, D)
    #
    #     score_diffs = torch.matmul(mean_pooled, reg_vec_norm.t()) + self.reg_bias
    #     direction_logits = torch.matmul(mean_pooled, cls_vec_norm.t()) + self.cls_bias
    #
    #     loss = None
    #     if score_diff_labels is not None:
    #         # regression_loss = self.criterion_reg(score_diffs, score_diff_labels)
    #         loss_l1 = torch.abs(score_diffs - score_diff_labels)
    #         sign_mismatch = torch.sign(score_diffs) != torch.sign(score_diff_labels)
    #         penalty = torch.where(sign_mismatch, 1.0, 1.0)
    #         regression_loss = (loss_l1 * penalty).mean()
    #
    #         classification_loss = self.criterion_cls(direction_logits, cls_labels)
    #         loss = regression_loss + self.lambda_cls * classification_loss
    #
    #     return PredictorOutput(
    #         score_diffs=score_diffs,
    #         direction_logits=direction_logits,
    #         loss=loss,
    #     )

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            score_diff_labels: Optional[torch.Tensor] = None,
            cls_labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        masked_hidden = hidden_states * attention_mask_expanded
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True)
        mean_pooled = sum_hidden / sum_mask

        reg_vec_norm = F.normalize(self.reg_vectors, p=2, dim=1)
        cls_vec_norm = F.normalize(self.cls_vectors, p=2, dim=1)

        score_diffs = torch.matmul(mean_pooled, reg_vec_norm.t()) + self.reg_bias
        direction_logits = torch.matmul(mean_pooled, cls_vec_norm.t()) + self.cls_bias

        loss = None
        if score_diff_labels is not None and cls_labels is not None:
            # 回归损失（SmoothL1 自带 mask 会出问题，因此手写）
            mask_reg = (score_diff_labels != -1000.0)
            if mask_reg.any():
                regression_loss = F.smooth_l1_loss(score_diffs[mask_reg], score_diff_labels[mask_reg], reduction="mean")
            else:
                regression_loss = 0.0

            # 分类损失（BCEWithLogitsLoss 不支持 ignore_index，所以我们手动 mask）
            mask_cls = (cls_labels != -1000)
            if mask_cls.any():
                cls_loss = F.binary_cross_entropy_with_logits(
                    direction_logits[mask_cls], cls_labels[mask_cls], reduction="mean"
                )
            else:
                cls_loss = 0.0

            # loss = regression_loss + self.lambda_cls * cls_loss
            loss = cls_loss

        return PredictorOutput(
            score_diffs=score_diffs,
            direction_logits=direction_logits,
            loss=loss,
        )

    @torch.no_grad()
    def encode(self, question: str, answer_A: str, answer_B: str,
               tokenizer: AutoTokenizer, max_length: int = 512, device: Optional[str] = None):
        # 构建文本、tokenize 并移动到 device
        text = f"Question: {question} Answer_A: {answer_A} Answer_B: {answer_B}"
        tok = tokenizer(text, truncation=True, max_length=max_length,
                        padding='max_length', return_tensors='pt')
        input_ids = tok['input_ids']
        attn_mask = tok['attention_mask']
        if device:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
        outputs = self.qwen(input_ids=input_ids, attention_mask=attn_mask)
        hidden = outputs.last_hidden_state
        mask = attn_mask.unsqueeze(-1).expand(hidden.size())
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return emb  # shape (1, D)

    @torch.no_grad()
    def batch_encode(
            self,
            questions: List[str],
            answers_A: List[str],
            answers_B: List[str],
            tokenizer: AutoTokenizer,
            max_length: int = 512,
            device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        texts = [
            f"Question: {q} Answer_A: {a} Answer_B: {b}"
            for q, a, b in zip(questions, answers_A, answers_B)
        ]
        tok = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        t0 = time.time()
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        # print(f'qwen time: {time.time() - t0}', flush=True)
        hidden = outputs.last_hidden_state  # (B, T, D)
        mask = attention_mask.unsqueeze(-1).expand(hidden.size())
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (B, D)
        return emb

    @torch.no_grad()
    def get_score_diff_from_embedding(self, embedding: torch.Tensor):
        return torch.matmul(embedding, self.reg_vectors.t()) + self.reg_bias

    @torch.no_grad()
    def get_cls_logits_from_embedding(self, embedding: torch.Tensor):
        cls_vec_norm = F.normalize(self.cls_vectors, p=2, dim=1)  # (K, D)
        return torch.matmul(embedding, cls_vec_norm.t()) + self.cls_bias

    @torch.no_grad()
    def get_rm_embeddings(self) -> torch.Tensor:
        return self.cls_vectors.detach()


def get_tokenizer(tokenizer_name, pad_token_if_none="<|pad|>"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # tokenizer.truncation_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({"pad_token": pad_token_if_none})  # 有必要吗？

    return tokenizer
