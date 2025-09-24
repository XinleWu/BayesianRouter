# import os
# import json
# from collections import defaultdict
# import argparse
# import torch
# import torch.nn.functional as F
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
# from datasets import load_dataset
#
# from offline_model import RewardDiffPredictor, get_tokenizer
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_reward_models = 3
# Dir = "/data/cs.aau.dk/zh45qz/router_data/rewardbench_behaviour/"
# model_path = "/data/cs.aau.dk/zh45qz/router_data/all_tulu/out_emb_norm"
# offline_router_tokenizer = get_tokenizer(model_path)
# offline_config = AutoConfig.from_pretrained(model_path)
# offline_router_model = RewardDiffPredictor.from_pretrained(
#         model_path,
#         config=offline_config,
#         num_reward_models=num_reward_models,
#         trust_remote_code=True
#     ).to(device)
# offline_router_model.eval()
#
#
# def offline_batch_encode(queries, resp_as, resp_bs):
#     emb = offline_router_model.batch_encode(
#         queries, resp_as, resp_bs,
#         offline_router_tokenizer,
#         max_length=512,
#         device=offline_router_model.device
#     )
#     return emb.cpu()
#
#
# def offline_batch_logits(embeddings: torch.Tensor):
#     emb_t = embeddings.to(offline_router_model.device)
#     logits = offline_router_model.get_cls_logits_from_embedding(emb_t)
#     return logits.cpu()
#
#
# def load_reward_bench():
#     data = load_dataset("allenai/reward-bench")["filtered"]  # .shuffle(seed=39)
#     eval_data = []
#     for example in data:
#         eval_data.append({
#             "id": f"{example['id']}",
#             "question": example["prompt"],
#             "chosen_answer": example["chosen"],
#             "rejected_answer": example["rejected"]
#         })
#     return eval_data
#
#
# data = load_reward_bench()
# # data需要分成mini-batch再执行后续推理
# batch_embs = offline_batch_encode()
# batch_logits = offline_batch_logits(batch_embs)
# batch_probs = torch.sigmoid(batch_logits)
# # argmax对每个数据选prob最大的reward model
# # 根据选择的reward model index，获取对每个数据，选中的RM的预测结果，然后和真实标签对比计算准确率
#
# # 从router需要得到一个字典，key是id，value是reward model index，然后根据id取下面的preds中找对应RM的preds，得到一个新的字典，key是id，value是preds
# with open(os.path.join(Dir, "Gemma-2B-rewardmodel-baseline.json"), "r", encoding="utf-8") as f:
#     Gemma_preds = json.load(f)
# with open(os.path.join(Dir, "Mistral-RM-for-RAFT-GSHF-v0.json"), "r", encoding="utf-8") as f:
#     Mistral_preds = json.load(f)
# with open(os.path.join(Dir, "Skywork-Reward-Llama-3.1-8B-v0.2.json"), "r", encoding="utf-8") as f:
#     Skywork_preds = json.load(f)


import os
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset
from tqdm import tqdm

from offline_model import RewardDiffPredictor, get_tokenizer

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_reward_models = 3
Dir = "/data/cs.aau.dk/zh45qz/router_data/rewardbench_behaviour/"
model_path = "/data/cs.aau.dk/zh45qz/router_data/all_tulu/out_emb_norm"
batch_size = 32

# Load Router
offline_router_tokenizer = get_tokenizer(model_path)
offline_config = AutoConfig.from_pretrained(model_path)
offline_router_model = RewardDiffPredictor.from_pretrained(
    model_path,
    config=offline_config,
    num_reward_models=num_reward_models,
    trust_remote_code=True
).to(device)
offline_router_model.eval()

# Load RM predictions
with open(os.path.join(Dir, "Gemma-2B-rewardmodel-baseline.json"), "r") as f:  # 75.40%
    Gemma_preds = json.load(f)
with open(os.path.join(Dir, "Mistral-RM-for-RAFT-GSHF-v0.json"), "r") as f:   # 82.27%
    Mistral_preds = json.load(f)
with open(os.path.join(Dir, "Skywork-Reward-Llama-3.1-8B-v0.2.json"), "r") as f:  # 94.03%
    Skywork_preds = json.load(f)
correct=0
for e in Mistral_preds.values():
    if e == 1:
        correct+=1
print('='*20)
print(correct/len(Mistral_preds))


all_rm_preds = [Gemma_preds, Mistral_preds, Skywork_preds]

# Load reward-bench
def load_reward_bench():
    data = load_dataset("allenai/reward-bench")["filtered"]
    eval_data = []
    for example in data:
        eval_data.append({
            "id": str(example['id']),
            "question": example["prompt"],
            "chosen_answer": example["chosen"],
            "rejected_answer": example["rejected"]
        })
    return eval_data

# Batch encoding
def offline_batch_encode(queries, resp_as, resp_bs):
    emb = offline_router_model.batch_encode(
        queries, resp_as, resp_bs,
        offline_router_tokenizer,
        max_length=512,
        device=offline_router_model.device
    )
    return emb.cpu()

def offline_batch_logits(embeddings: torch.Tensor):
    emb_t = embeddings.to(offline_router_model.device)
    logits = offline_router_model.get_cls_logits_from_embedding(emb_t)
    return logits.cpu()

# Main process
data = load_reward_bench()
mix_preds = {}
correct = 0
total = 0

for i in tqdm(range(0, len(data), batch_size)):
    batch = data[i:i+batch_size]
    questions = [item["question"] for item in batch]
    chosen_answers = [item["chosen_answer"] for item in batch]
    rejected_answers = [item["rejected_answer"] for item in batch]
    ids = [item["id"] for item in batch]

    # Get logits & probs
    batch_embs = offline_batch_encode(questions, chosen_answers, rejected_answers)
    batch_logits = offline_batch_logits(batch_embs)
    batch_probs = torch.sigmoid(batch_logits)  # [B, num_RMs]

    # Select RM with highest prob per sample
    # best_rm_idxs = torch.argmax(batch_probs, dim=1).tolist()
    best_rm_idxs = torch.argmax(batch_probs[:, :2], dim=1).tolist()

    for j, sample_id in enumerate(ids):
        best_rm = best_rm_idxs[j]
        rm_pred = all_rm_preds[best_rm].get(sample_id, 0)  # default to 0 if not found
        mix_preds[sample_id] = rm_pred
        correct += int(rm_pred == 1)
        total += 1

# # Save predictions
# with open(os.path.join(Dir, "mix_preds.json"), "w") as f:
#     json.dump(mix_preds, f, indent=2)

# Accuracy
accuracy = correct / total
print(f"Router-based mix prediction accuracy: {accuracy:.4f}")  # 93.53



oracle_correct = 0
for item in data:
    sample_id = item["id"]
    rm_votes = [rm.get(sample_id, 0) for rm in all_rm_preds]
    if any(v == 1 for v in rm_votes):
        oracle_correct += 1
oracle_accuracy = oracle_correct / total
print(f"Oracle (upper bound) accuracy: {oracle_accuracy:.4f}")  # 97.72





