import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import PeftModel
from math_verify import parse, verify
from policy_model import setup_lora_model, ResponseGenerator


# class DatasetManager:
#     def __init__(self):
#
#     def get_test_data
#
#
# def load_policy_for_inference(saved_dir: str,
#                               base_model_name_or_path: str,
#                               device: str = "cuda:0",
#                               dtype=torch.float16,
#                               adapter_name: str = "policy"):
#     # 1) 找到 adapter 子目录（PEFT 通常放在 saved_dir/policy）
#     adapter_path = saved_dir
#     maybe_policy = os.path.join(saved_dir, "policy")
#     if os.path.isdir(maybe_policy):
#         adapter_path = maybe_policy
#
#     # 2) 加载 tokenizer（从 saved_dir）
#     tokenizer = AutoTokenizer.from_pretrained(saved_dir)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "left"  # decoder-only 必须左填充
#
#     # 3) 加载 base model（优先尝试 device_map="auto" + 指定 dtype）
#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_name_or_path,
#         torch_dtype=dtype,
#         device_map={
#             "": 0
#         }  # 如果机器上有加速库，会自动把模型放到合适设备
#     )
#
#     # 4) 把 adapter (LoRA) 应用到 base
#     model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=adapter_name, torch_dtype=dtype)
#     model.set_adapter(adapter_name)
#     model.eval()
#
#     # 如果 device_map="auto" 成功，model 已经分配好；否则把整个 model 移到 device
#     try:
#         model.to(device)
#     except Exception:
#         pass
#
#     return model, tokenizer
#
#
# def generate_response_test(response_generator, dataset_manager, out_path,
#                            dataset_name: str = 'mt-bench',
#                            batch_size: int = 16,
#                            max_new_tokens: int = 256,
#                            chunk_size: int = 4):
#     # unwrap DDP if needed (so generate works)
#     orig_model = getattr(response_generator, "model", None)
#     unwrapped = False
#     try:
#         if isinstance(orig_model, DDP):
#             response_generator.model = orig_model.module
#             unwrapped = True
#
#         test_set = dataset_manager.get_test_data(dataset_name)
#         n = len(test_set)
#         print(f'len test: {n}')
#
#         with open(out_path, "w", encoding="utf-8") as fout:
#             for i in range(0, n, batch_size):
#                 batch = test_set[i:i + batch_size]
#                 queries = []
#                 for ex in batch:
#                     if "question" in ex:
#                         queries.append(ex["question"])
#
#                 prompts = [response_generator.generate_prompt(q) for q in queries]
#                 outputs = response_generator.generate_responses(
#                     batch=queries,
#                     n_responses=1,
#                     prompts=prompts,
#                     override_temperature=0.0,
#                     do_sample_override=False,
#                     max_new_tokens=max_new_tokens,
#                     chunk_size=chunk_size,
#                 )
#
#                 for (q_out, p_out, resp), ex in zip(outputs, batch):
#                     result = {
#                         "instruction": q_out,
#                         "output": resp
#                     }
#                     fout.write(json.dumps(result, ensure_ascii=False) + "\n")
#     finally:
#         if unwrapped:
#             response_generator.model = orig_model
#
#
# saved_dir = "/data/cs.aau.dk/zh45qz/router_data/output_policy/both"  # 你保存的目录（含 policy/）
# base_model = "RLHFlow/LLaMA3-SFT"  # 训练时用的 base 模型名或本地路径
# model, tokenizer = load_policy_for_inference(saved_dir, base_model, device="cuda:3", dtype=torch.float16)
# dataset_manager = DatasetManager("HuggingFaceH4/mt_bench_prompts")
#
# response_generator = ResponseGenerator(model, tokenizer)
# generate_response_test(response_generator, dataset_manager, out_path='output_full/both.json')




import os
import json
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP

#### Dataset manager ####
class DatasetManager:
    def __init__(self, dataset_name: str = "HuggingFaceH4/mt_bench_prompts", split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        ds = load_dataset(dataset_name)
        # 取指定 split（mt-bench_prompts 只有 train split）
        self.dataset = ds[split]
        print(f'len data: {len(self.dataset)}')

    def get_test_data(self):
        # 返回一个可迭代的 dataset（huggingface Dataset）
        return self.dataset

def load_policy_for_inference(saved_dir: str,
                              base_model_name_or_path: str,
                              device: str = "cuda:0",
                              dtype=torch.float16,
                              adapter_name: str = "policy",
                              use_adapter: bool = True):
    # tokenizer 一定从 base 模型加载
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # load base model (try device_map="auto" first)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
        )
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=dtype,
        )
        try:
            base_model.to(device)
        except Exception:
            pass

    if use_adapter:
        adapter_path = saved_dir
        maybe_policy = os.path.join(saved_dir, "policy")
        if os.path.isdir(maybe_policy):
            adapter_path = maybe_policy

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            adapter_name=adapter_name,
            torch_dtype=dtype,
        )
        # ensure adapter set (PeftModel should have this)
        try:
            model.set_adapter(adapter_name)
        except Exception:
            pass
        # try move to device if not already placed
        try:
            model.to(device)
        except Exception:
            pass
    else:
        model = base_model
        try:
            model.to(device)
        except Exception:
            pass

    model.eval()
    return model, tokenizer



#### 辅助：从任意嵌套返回结构中找第一个字符串（用于提取生成结果） ####
def find_first_string(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for v in x.values():
            s = find_first_string(v)
            if s:
                return s
    if isinstance(x, (list, tuple)):
        for v in x:
            s = find_first_string(v)
            if s:
                return s
    return None

#### 辅助：把 history + 当前 user_turn 转为 prompt 字符串（简单 role 标记） ####
def build_prompt_from_history(history, user_turn):
    # history: list of {"role": "user"/"assistant", "content": "..."}
    parts = []
    for m in history:
        role = m.get("role", "user")
        txt = m.get("content", "")
        # 简单 role prefix（你可以改成模型训练时用的格式）
        prefix = "User:" if role == "user" else "Assistant:"
        parts.append(f"{prefix} {txt}")
    parts.append(f"User: {user_turn}")
    # 加上 assistant 的空位，让模型以 assistant 的角色回复（有些 prompt template 会需要）
    parts.append("Assistant:")
    return "\n".join(parts)

#### 主流程：针对每条 mt-bench prompt 进行多轮生成并写出 JSONL ####
def generate_response_test(response_generator, dataset_manager, out_path,
                           dataset_name: str = 'mt-bench',
                           batch_size: int = 16,
                           max_new_tokens: int = 256,
                           chunk_size: int = 4):
    orig_model = getattr(response_generator, "model", None)
    unwrapped = False
    try:
        if isinstance(orig_model, DDP):
            response_generator.model = orig_model.module
            unwrapped = True

        test_set = dataset_manager.get_test_data()
        n = len(test_set)
        print(f'len test: {n}')

        with open(out_path, "w", encoding="utf-8") as fout:
            for i in range(0, n, batch_size):
                batch = test_set[i:i + batch_size]

                for ex_idx, ex in enumerate(batch):
                    prompts = ex["prompt"]  # MT-Bench 多轮 prompt
                    conversation = []
                    history_texts = []  # 用于构建多轮上下文

                    for user_turn in prompts:
                        queries = [user_turn]

                        # 构建 prompt 时加入 history_texts
                        prompts_for_model = []
                        for q in queries:
                            # 把 history 拼接到当前 user_turn
                            if history_texts:
                                prompt_text = "\n".join(history_texts + [q])
                            else:
                                prompt_text = q
                            prompts_for_model.append(prompt_text)

                        outputs = response_generator.generate_responses(
                            batch=queries,
                            n_responses=1,
                            max_input_length=400,
                            prompts=prompts_for_model,
                            override_temperature=0.0,
                            do_sample_override=False,
                            max_new_tokens=max_new_tokens,
                            chunk_size=chunk_size,
                        )

                        # 保留原来的解包方式
                        _, _, resp_text = outputs

                        conversation.append({"role": "user", "content": user_turn})
                        conversation.append({"role": "assistant", "content": resp_text})
                        history_texts.append(user_turn)
                        history_texts.append(resp_text)

                    # 保存整条多轮对话
                    result = {
                        "id": ex_idx + i,
                        "conversation": conversation
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    finally:
        if unwrapped:
            response_generator.model = orig_model


#### 用法示例（替换为你自己的路径 / model / ResponseGenerator） ####
if __name__ == "__main__":
    saved_dir = "/data/cs.aau.dk/zh45qz/router_data/output_policy/both"
    base_model = "RLHFlow/LLaMA3-SFT"
    device = "cuda:2"

    # model, tokenizer = load_policy_for_inference(saved_dir, base_model, device=device, dtype=torch.float16)

    model, tokenizer = load_policy_for_inference(
        saved_dir=saved_dir,  # 路径无所谓，不会用到
        base_model_name_or_path="RLHFlow/LLaMA3-SFT",
        device="cuda:2",
        dtype=torch.float16,
        use_adapter=True  # 🚨 关键：禁用 adapter
    )

    dataset_manager = DatasetManager("lmarena-ai/arena-hard-auto", split="train")

    # 你的 ResponseGenerator 类：构造时传模型和 tokenizer（保持与你现有实现一致）
    response_generator = ResponseGenerator(model, tokenizer)

    generate_response_test(response_generator, dataset_manager, out_path="output_full/sft_mtbench.jsonl")

