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
        # BT head不需要bias
        self.bt_vectors = nn.Parameter(torch.randn(num_reward_models, config.hidden_size))

        self.shared_mlp = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )

        nn.init.kaiming_uniform_(self.reg_vectors, a=math.sqrt(5))
        nn.init.zeros_(self.reg_bias)
        nn.init.kaiming_uniform_(self.cls_vectors, a=math.sqrt(5))
        nn.init.zeros_(self.cls_bias)
        nn.init.kaiming_uniform_(self.bt_vectors, a=math.sqrt(5))

        # self.criterion_reg = nn.SmoothL1Loss(reduction="mean")
        self.criterion_reg = nn.L1Loss(reduction="mean")
        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.lambda_reg = 0.6
        self.lambda_cls = 0.9
        self.lambda_bt = 1.0

    def load_qwen_backbone(self, model_path: str):
        backbone = AutoModel.from_pretrained(model_path, config=self.config)
        self.backbone = backbone

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        masked_hidden = hidden_states * attention_mask_expanded
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True)
        mean_pooled = sum_hidden / sum_mask
        return mean_pooled  # (B, D)

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
        else:
            score_diff_labels = cls_labels = bt_pairs = None

        h1 = self.encode(input_ids_1, attention_mask_1)
        h2 = self.encode(input_ids_2, attention_mask_2)
        h_mean = (h1 + h2) / 2
        h_diff = torch.abs(h1 - h2)
        h_cat = torch.cat([h_mean, h_diff], dim=-1)
        h_shared = self.shared_mlp(h_cat)

        reg_vec_norm = F.normalize(self.reg_vectors, p=2, dim=1)
        cls_vec_norm = F.normalize(self.cls_vectors, p=2, dim=1)
        bt_vec_norm = F.normalize(self.bt_vectors, p=2, dim=1)

        score_diffs = torch.matmul(h_shared, reg_vec_norm.t()) + self.reg_bias
        cls_logits = torch.matmul(h_shared, cls_vec_norm.t()) + self.cls_bias
        bt_scores = torch.matmul(h_shared, bt_vec_norm.t())

        regression_loss = torch.tensor(0.0, device=h_shared.device)
        cls_loss = torch.tensor(0.0, device=h_shared.device)
        bt_loss = torch.tensor(0.0, device=h_shared.device)
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
            # bt_pairs: Tensor of shape (B, max_pairs, 2), with padding value -1
            B, max_pairs, _ = bt_pairs.shape
            total_pairs = 0
            for i in range(B):
                for j in range(max_pairs):
                    w = bt_pairs[i, j, 0].item()
                    l = bt_pairs[i, j, 1].item()
                    # 只处理有效对，遇到 padding (-1) 就跳出本样本后续
                    if w < 0 or l < 0:
                        break
                    total_pairs += 1
                    s_i = bt_scores[i, w]
                    s_j = bt_scores[i, l]
                    bt_loss = bt_loss - torch.log(torch.sigmoid(s_i - s_j) + 1e-12)
            # 归一化：只有当确实有成对样本时才除，这里的归一化有必要吗？
            if total_pairs > 0:
                bt_loss = bt_loss / total_pairs

        # if bt_pairs is not None:
        #     # bt_pairs: Tensor of shape (B, max_pairs, 2), with padding value -1
        #     valid_mask = (bt_pairs[:, :, 0] >= 0) & (bt_pairs[:, :, 1] >= 0)
        #     if valid_mask.any():
        #         batch_indices, pair_indices = torch.nonzero(valid_mask, as_tuple=True)
        #         w_indices = bt_pairs[batch_indices, pair_indices, 0]
        #         l_indices = bt_pairs[batch_indices, pair_indices, 1]
        #
        #         s_i = bt_scores[batch_indices, w_indices]
        #         s_j = bt_scores[batch_indices, l_indices]
        #         diffs = s_i - s_j
        #         bt_loss = -torch.log(torch.sigmoid(diffs) + 1e-12).mean()

        loss = self.lambda_cls * cls_loss + self.lambda_reg * regression_loss + self.lambda_bt * bt_loss

        # return PredictorOutput(
        #     score_diffs=score_diffs,
        #     cls_logits=cls_logits,
        #     bt_scores=bt_scores,
        #     loss=loss,
        # )

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
            max_length: int = 512,
            device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        texts_A = [q + "\n" + a for q, a in zip(questions, answers_A)]
        texts_B = [q + "\n" + b for q, b in zip(questions, answers_B)]

        enc_A = tokenizer(
            texts_A,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        enc_B = tokenizer(
            texts_B,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

        if device is not None:
            enc_A = {k: v.to(device) for k, v in enc_A.items()}
            enc_B = {k: v.to(device) for k, v in enc_B.items()}
            self.to(device)

        h1 = self.encode(enc_A['input_ids'], enc_A['attention_mask'])  # (B, D)
        h2 = self.encode(enc_B['input_ids'], enc_B['attention_mask'])  # (B, D)
        h_mean = (h1 + h2) / 2
        h_diff = torch.abs(h1 - h2)
        h_cat = torch.cat([h_mean, h_diff], dim=-1)  # (B, 2*D)
        return h_cat

    @torch.no_grad()
    def get_score_diff_from_embedding(self, embedding: torch.Tensor):
        reg_vec_norm = F.normalize(self.reg_vectors, p=2, dim=1)
        return torch.matmul(embedding, reg_vec_norm.t()) + self.reg_bias

    @torch.no_grad()
    def get_cls_logits_from_embedding(self, embedding: torch.Tensor):
        cls_vec_norm = F.normalize(self.cls_vectors, p=2, dim=1)  # (K, D)
        return torch.matmul(embedding, cls_vec_norm.t()) + self.cls_bias

    @torch.no_grad()
    def get_rm_embeddings(self) -> torch.Tensor:
        return F.normalize(self.cls_vectors.detach(), p=2, dim=1)


def get_tokenizer(tokenizer_name, pad_token_if_none="<|pad|>"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # tokenizer.truncation_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({"pad_token": pad_token_if_none})  # 有必要吗？

    return tokenizer
