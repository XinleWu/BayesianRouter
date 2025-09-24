from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, inject_adapter_in_model, PeftModel
import torch
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, List


# lora_model
def setup_lora_model(model_name, device):
    # ——— 1) tokenizer + base model ———
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"           # decoder-only 必须左填充
    # base_model.config.max_length = 150       # 明确 max_length 避免 truncate 警告

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={  # 所有层都放到同一个 device
            "": device.index  # e.g. 1、2、3
        },
    )

    # ——— 2) 冻结所有原始权重 ———
    for p in base_model.parameters():
        p.requires_grad_(False)

    # ——— 3) 定义 LoRA 配置 ———
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none"
    )

    # ——— 4) 在同一个 base_model 上依次挂载 “policy” 和 “reference” 两个 adapter ———
    peft_model = get_peft_model(base_model, lora_cfg, adapter_name="policy")
    peft_model.add_adapter("reference", lora_cfg)

    # --- 使 reference 的 LoRA 参数显式为 0（等价于 base model） ---
    for n, p in peft_model.named_parameters():
        if ".reference." in n and ("lora_" in n or "loraA" in n or "loraB" in n):
            # 直接把权重置为 0
            p.data.zero_()

    # # ——— 5) 冻结 reference adapter 的参数（policy adapter 保持可训练） ———
    # peft_model.set_adapter("reference")
    # for n, p in peft_model.named_parameters():
    #     if "lora_" in n:
    #         p.requires_grad_(False)

    # # 5) **把 policy 的 LoRA 权重显式拷贝到 reference（保证 reference 是 policy 的 snapshot）**
    # #    这样 reference 就是 policy 在初始化时刻的拷贝
    # state = peft_model.state_dict()
    # # 找到所有 policy 参数名，并把值拷贝到对应的 reference 名
    # for name in list(state.keys()):
    #     if ".policy." in name:
    #         ref_name = name.replace(".policy.", ".reference.")
    #         if ref_name in state:
    #             # 直接用 state copy（不会影响计算图）
    #             state[ref_name] = state[name].clone()
    # peft_model.load_state_dict(state, strict=False)

    # 6) **显式冻结 reference adapter 的参数，确保不被优化器收录**
    #    并显式设置只有 policy 的 LoRA 参数 requires_grad=True
    for n, p in peft_model.named_parameters():
        # 默认全部先设为 False
        p.requires_grad = False
    for n, p in peft_model.named_parameters():
        # 唯一允许训练的：policy adapter 的 lora 权重
        if ".policy." in n and ("lora_" in n or "loraA" in n or "loraB" in n):
            p.requires_grad = True

    # 7) 切回 policy adapter并打印确认
    peft_model.set_adapter("policy")
    peft_model.print_trainable_parameters()  # 应只显示 policy adapter 上的参数

    return peft_model, tokenizer


# response_generator
class ResponseGenerator:
    def __init__(self, model, tokenizer, prompt_type="instruction", temperature=0.8):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_type = prompt_type
        self.temperature = temperature

        if isinstance(self.model, DistributedDataParallel):
            self.model.module.set_adapter("policy")  # 永久锁定policy适配器
        else:
            self.model.set_adapter("policy")


    def generate_prompt(self, query):
        """Generate prompt based on the task type."""
        if self.prompt_type == "reasoning":
            return (f"Your task is to answer the question below. "
                    f"Give step-by-step reasoning before you answer, "
                    f"and when you’re ready to answer, please use the format `Final answer:...`.\n"
                    f"Question: {query}\nSolution: ")
        elif self.prompt_type == "instruction":
            return query
        else:
            raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

    # def generate_responses(self, batch, n_responses=30, max_input_length=128, max_new_tokens=150, chunk_size=1):
    #     prompts = [self.generate_prompt(query) for query in batch]
    #     model_for_gen = getattr(self.model, "module", self.model)
    #
    #     # 记录模型原始 training 状态（True/False），以便生成后恢复
    #     was_training = model_for_gen.training
    #     model_for_gen.eval()
    #     gen_device = next(model_for_gen.parameters()).device
    #
    #     all_triples = []
    #     for i in range(0, len(prompts), chunk_size):
    #         chunk_prompts = prompts[i:i + chunk_size]
    #         chunk_queries = batch[i:i + chunk_size]
    #
    #         enc = self.tokenizer(
    #             chunk_prompts,
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True,
    #             max_length=max_input_length,
    #         )
    #
    #         input_ids = enc["input_ids"].to(gen_device)
    #         attention_mask = enc["attention_mask"].to(gen_device)
    #
    #         # 在 no_grad 下生成（已经处于 eval 模式），防止生成阶段占用梯度
    #         with torch.no_grad():
    #             output_ids = model_for_gen.generate(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 max_new_tokens=max_new_tokens,
    #                 temperature=self.temperature,
    #                 pad_token_id=self.tokenizer.eos_token_id,
    #                 do_sample=True,
    #                 num_return_sequences=n_responses,
    #             )
    #
    #         decoded = []
    #         for k in range(output_ids.size(0)):
    #             input_len = input_ids.size(1)
    #             gen_ids = output_ids[k, input_len:]
    #             text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
    #             decoded.append(text)
    #
    #         for j, query in enumerate(chunk_queries):
    #             start = j * n_responses
    #             group = decoded[start:start + n_responses]
    #             for resp in group:
    #                 all_triples.append((query, chunk_prompts[j], resp))
    #
    #         del input_ids, attention_mask, output_ids, decoded
    #         torch.cuda.empty_cache()
    #
    #     # 恢复成调用前的训练/评估模式（非常关键）
    #     if was_training:
    #         model_for_gen.train()
    #
    #     return all_triples

    def generate_responses(self, batch, n_responses=30, max_input_length=128, max_new_tokens=256,
                           chunk_size=1, prompts: Optional[List[str]] = None,
                           override_temperature: Optional[float] = None,
                           do_sample_override: Optional[bool] = None):
        """
        batch: list of queries (strings)
        prompts: optional list of prompts (strings) aligned to batch; if provided, will be used instead
                 of generate_prompt(query).
        override_temperature: if provided, temporarily use this temperature for generation.
        do_sample_override: if provided, uses this bool to force sampling/greedy; otherwise infer from temperature.
        """
        if prompts is None:
            prompts = [self.generate_prompt(q) for q in batch]
        if len(prompts) != len(batch):
            raise ValueError("prompts length must match batch length")

        model_for_gen = getattr(self.model, "module", self.model)
        print("gen_device:", next(model_for_gen.parameters()).device, flush=True)

        # 记录模型原始 training 状态（True/False），以便生成后恢复
        was_training = model_for_gen.training
        model_for_gen.eval()
        gen_device = next(model_for_gen.parameters()).device

        used_temp = self.temperature if override_temperature is None else float(override_temperature)
        # decide do_sample: explicit override has priority; else temperature>0 -> sample, else greedy
        if do_sample_override is not None:
            do_sample_flag = bool(do_sample_override)
        else:
            do_sample_flag = False if (used_temp is None or float(used_temp) == 0.0) else True

        all_triples = []
        for i in range(0, len(prompts), chunk_size):
            chunk_prompts = prompts[i:i + chunk_size]
            chunk_queries = batch[i:i + chunk_size]

            enc = self.tokenizer(
                chunk_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            )

            input_ids = enc["input_ids"].to(gen_device)
            attention_mask = enc["attention_mask"].to(gen_device)

            # 在 no_grad 下生成（已经处于 eval 模式），防止生成阶段占用梯度
            with torch.no_grad():
                output_ids = model_for_gen.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=used_temp,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=do_sample_flag,
                    num_return_sequences=n_responses,
                )

            # decode: HF returns (chunk_size * n_responses, seq_len) stacked by input
            decoded = []
            for k in range(output_ids.size(0)):
                input_len = input_ids.size(1)
                gen_ids = output_ids[k, input_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                decoded.append(text)

            for j, query in enumerate(chunk_queries):
                start = j * n_responses
                group = decoded[start:start + n_responses]
                for resp in group:
                    all_triples.append((query, chunk_prompts[j], resp))

            del input_ids, attention_mask, output_ids, decoded
            torch.cuda.empty_cache()

        # 恢复成调用前的训练/评估模式（非常关键）
        if was_training:
            model_for_gen.train()
        return all_triples


