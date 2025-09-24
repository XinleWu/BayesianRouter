import re
import os
import time
import json
import datetime
import torch
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from math_verify import parse, verify
from utils4 import load_config, DatasetManager, generate_preference_pairs_multi_rm, LLMTrainer, \
    load_reward_models, rpc_select_arms, rpc_update_arm, generate_offline_prior, rpc_offline_batch_encode, \
    DatasetManagerInstruct, rpc_init_globals, DatasetManagerMMLU
from policy_model import setup_lora_model, ResponseGenerator

save_dir = "/data/cs.aau.dk/zh45qz/router_data/output_3000/"


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12366'  # 12355任意未占用端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600))
    torch.cuda.set_device(rank)


def main(rank, world_size):
    setup_ddp(rank, world_size)

    # Initialize RPC service
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://127.0.0.1:29501", rpc_timeout=3600
        ),
    )
    config = load_config("config.yaml")
    # dataset_name = "strategyqa"
    # dataset_name = "iter3-20k"
    dataset_name = "gsm8k"
    # dataset_name = "mmlu"

    # Load reward_models on all ranks (rank0: real models; others: proxies)
    reward_device = torch.device("cuda:0") if rank == 0 else None
    reward_models = load_reward_models(
        config["reward_models"],
        device=reward_device,
        multi_gpu=False,
        rpc_info={"rank": rank, "world_size": world_size}
    )
    K = len(reward_models)

    # Rank0: initialize global LinUCB & embedding model, serve RPC
    if rank == 0:
        rpc_init_globals(
            num_reward_models=K,
            offline_router_path=config["offline_router_path"],
            sigma2=1.0,
            lambda0=1.0,
            device=torch.device("cuda:0")
        )
        print(f"[Rank0] Serving {K} reward models & offline router via RPC.")
    dist.barrier()  # 等待rpc_init_globals执行完

    if rank > 0:
        # Policy workers (rank > 0)
        policy_device = torch.device(f"cuda:{rank}")
        policy_model, tokenizer = setup_lora_model(config["model_name"], policy_device)
        policy_model.to(policy_device)

        # —— 为 policy workers (rank 1,2,3) 建 sub‐group ——
        policy_ranks = list(range(1, world_size))
        policy_group = dist.new_group(ranks=policy_ranks, backend="nccl")
        policy_model = DDP(policy_model,
                           device_ids=[policy_device],
                           output_device=policy_device,
                           find_unused_parameters=True,
                           process_group=policy_group)
        policy_model.train()

        # # Dataset manager
        # dataset_manager = DatasetManager({dataset_name: config["datasets"][dataset_name]},
        #                                  test_size=config["training"]["test_size"],
        #                                  dev_size=config["training"]["dev_size"])
        dataset_manager = DatasetManager({'gsm8k': 'hf:openai/gsm8k'}, dev_size=config["training"]["dev_size"])
        # dataset_manager = DatasetManagerMMLU({'mmlu': 'hf:cais/mmlu'}, dev_size=config["training"]["dev_size"])
        # dataset_manager = DatasetManagerInstruct(prompts_path='/data/cs.aau.dk/zh45qz/router_data/training_prompts/training_prompts_60.json',
        #                                          dev_size=config["training"]["dev_size"])

        # Response generation & trainer, pref generation
        trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
        trainer = LLMTrainer(policy_model,
                             Adam(trainable_params, lr=float(config["training"]["learning_rate"])),
                             tokenizer,
                             grad_accum_steps=64)  # 这个数字要小于真实batch size
        n_responses = config["training"]["n_responses"]
        response_generator = ResponseGenerator(policy_model, tokenizer,  # prompt_type="reasoning",
                                               temperature=config["training"]["temperature"])


        # # 训练前评估
        # def _set_adapter_on_response_generator(response_generator, adapter_name):
        #     # response_generator.model 可能是 DDP 包裹，也可能不是
        #     model_for_gen = getattr(response_generator.model, "module", response_generator.model)
        #     model_for_gen.set_adapter(adapter_name)
        # # === 在首次评估（希望得到基模型行为）前，切换到 reference ===
        # _set_adapter_on_response_generator(response_generator, "reference")
        # # dump_tora_jsonl(response_generator, dataset_manager, 'gsm8k')
        # compute_strategyqa_accuracy(response_generator, dataset_manager, "strategyqa", batch_size=8)
        # # 恢复为 policy，进入训练
        # _set_adapter_on_response_generator(response_generator, "policy")

        # compute_strategyqa_accuracy(response_generator, dataset_manager, "strategyqa", batch_size=8)
        # dump_tora_jsonl(response_generator, dataset_manager, 'gsm8k')
        # dump_tora_jsonl_xfinder(response_generator, dataset_manager, 'mmlu', out_json="log/mmlu_before_float32.json")
        # generate_response_test(response_generator, dataset_manager, out_path='output/iter3_outputs_before_300Online.json')

        # Training loop
        print(f"[RANK {rank}] Training on {dataset_name} dataset")
        M = config["training"]["iterations"]
        batch_size = config["training"]["batch_size"]
        eval_threshold = config["training"]["eval_threshold"]
        for iteration in range(M):
            total_loss = 0
            train_data = dataset_manager.get_train_data(dataset_name)
            print(f'len train: {len(train_data)}')
            for i in range(0, len(train_data), batch_size):
                start_time = time.time()
                print(f'{i}-th batch', flush=True)
                batch = train_data[i:i + batch_size]
                queries = [ex['question'] for ex in batch]

                # 1) Generate responses, triples: list of (query, prompt, response)
                triples = response_generator.generate_responses(queries, n_responses=n_responses,
                                                                max_input_length=64, max_new_tokens=256, chunk_size=4)
                t1 = time.time()
                print(f'generate_responses time: {t1-start_time}')

                query_response_pairs_for_rm = [(p, r) for (q, p, r) in triples]
                # 得到 candidate pairs 的所有 (q, a, b) 以及对应 embeddings/logits
                pairs_per_query = 4
                pair_queries, resp_as, resp_bs, pair_emb, pair_logits = generate_offline_prior(
                    query_response_pairs_for_rm)
                t11 = time.time()
                print(f'generate_offline_prior time: {t11-t1}')
                if len(pair_queries) == 0:
                    sel_prefs = []
                    sel_pair_indices = []
                    sel_pair_rm_indices = []
                else:
                    # # 向量化地为每个 pair 选 arm（一次 rpc 到 rank0）
                    # selected_rm_indices = rpc.rpc_sync(
                    #     to="worker0",
                    #     func=rpc_select_arms,
                    #     args=(pair_emb.detach(),)  # CPU tensor ok
                    # )  # list length N
                    selected_rm_indices = [2] * pair_emb.shape[0]  # 仅使用RM0
                    # selected_rm_indices = rpc.rpc_sync(  # 仅使用offline
                    #     to="worker0",
                    #     func=rpc_select_arms,
                    #     args=(pair_emb.detach(), pair_logits)  # pass priors (N, K)
                    # )
                    tt = time.time()
                    print(f'select arm time: {tt-t11}')

                    sel_rm_list = list(selected_rm_indices)
                    sel_rm_record = {
                        "selected_rm_indices": sel_rm_list
                    }
                    print(sel_rm_record)
                    # sel_rm_jsonl = os.path.join(save_dir, "onlyOffline.jsonl")
                    # with open(sel_rm_jsonl, "a", encoding="utf-8") as jf:
                    #     jf.write(json.dumps(sel_rm_record, ensure_ascii=False) + "\n")

                    # 使用每个 pair 分配到的 RM 来为这对的 a/b 打分，并按 query 采样得到最终用于训练的偏好对
                    sel_prefs, sel_pair_indices, sel_pair_rm_indices = generate_preference_pairs_multi_rm(
                        reward_models,
                        pair_queries,
                        resp_as,
                        resp_bs,
                        selected_rm_indices,
                        pairs_per_query=pairs_per_query,
                        tie_eps=1e-1,
                        rng_seed=None
                    )
                    t111 = time.time()
                    print(f'generate_preference_pairs_multi_rm time: {t111-tt}')
                print(
                    f"Generated {len(sel_prefs)} preference pairs from batch (expected up to {batch_size * pairs_per_query})")
                t2 = time.time()
                print(f'generate_preference_pairs_multi_rm time: {t2-t1}')

                # 3) train policy
                avg_loss, per_pair_losses = trainer.train_step(sel_prefs)
                total_loss += avg_loss
                t3 = time.time()
                print(f'train_step time: {t3-t2}')

                # 4) update MAB: 对每个被选的偏好对，用其对应的 RM 和该对的 context embedding 与 per-pair loss 做 update
                # sel_pair_indices: 原始 candidate pair idx (对应 pair_emb 的行)
                # sel_pair_rm_indices: 为每个 selected pair 分配的 RM idx
                for j, (pair_idx, rm_idx) in enumerate(zip(sel_pair_indices, sel_pair_rm_indices)):
                    try:
                        ctx = pair_emb[pair_idx]  # CPU tensor, shape (D,)
                    except Exception:
                        # defensive: if pair_emb unavailable, fallback to mean context
                        ctx = pair_emb.mean(dim=0)
                    # per_pair_losses 的顺序与 sel_prefs 对齐
                    per_pair_loss = float(per_pair_losses[j])
                    # RPC update：每对单独 update
                    rpc.rpc_sync(to="worker0", func=rpc_update_arm,
                                 args=(int(rm_idx), ctx.detach(), float(per_pair_loss)))
                print(
                    f"[Rank {rank} | Iter {iteration + 1} | Batch {i // batch_size + 1}]"
                    f"  | Loss={avg_loss:.4f}"
                )
                t4 = time.time()
                print(f'update MAB time: {t4-t3}')
            avg_loss = total_loss / (len(train_data) // batch_size)
            print(f"Iteration {iteration + 1}, Avg Loss: {avg_loss:.4f}")

            # model_to_save = policy_model.module if hasattr(policy_model, "module") else policy_model
            # model_save_dir = os.path.join(save_dir, "both")
            # # 优先使用 huggingface style 的 save_pretrained（如果该对象支持）
            # if hasattr(model_to_save, "save_pretrained"):
            #     model_to_save.save_pretrained(model_save_dir)
            #     # 同时保存 tokenizer
            #     try:
            #         tokenizer.save_pretrained(model_save_dir)
            #     except Exception as e:
            #         print(f"[Rank {rank}] Warning: tokenizer.save_pretrained failed: {e}")
            # else:
            #     # 否则保存 state_dict（通用方法）
            #     torch.save(model_to_save.state_dict(), os.path.join(model_save_dir, "pytorch_model.bin"))


            # compute_strategyqa_accuracy(response_generator, dataset_manager, "strategyqa", batch_size=8)
            dump_tora_jsonl(response_generator, dataset_manager, 'gsm8k')
            # dump_tora_jsonl_xfinder(response_generator, dataset_manager, 'mmlu', out_json="log/offline.json")
            # generate_response_test(response_generator, dataset_manager, out_path='output_3000/both.json')
            local_flag = torch.tensor(1 if avg_loss < eval_threshold else 0, device=policy_device)
            torch.distributed.all_reduce(local_flag, op=torch.distributed.ReduceOp.MIN, group=policy_group)
            if local_flag.item() == 1:  # 只有当所有 rank 的 local_flag 都是 1，才返回 1
                print("All ranks converged; breaking.")
                break
    # dist.barrier()
    rpc.shutdown()
    dist.destroy_process_group()


# 存储回答，用于比较性能
def generate_response_test(response_generator, dataset_manager, out_path,
                           dataset_name: str = 'iter3-20k',
                           batch_size: int = 16,
                           max_new_tokens: int = 256,
                           chunk_size: int = 4):
    # unwrap DDP if needed (so generate works)
    orig_model = getattr(response_generator, "model", None)
    unwrapped = False
    try:
        if isinstance(orig_model, DDP):
            response_generator.model = orig_model.module
            unwrapped = True

        test_set = dataset_manager.get_test_data(dataset_name)
        n = len(test_set)
        print(f'len test: {n}')

        with open(out_path, "w", encoding="utf-8") as fout:
            for i in range(0, n, batch_size):
                batch = test_set[i:i + batch_size]
                queries = []
                for ex in batch:
                    if "question" in ex:
                        queries.append(ex["question"])

                prompts = [response_generator.generate_prompt(q) for q in queries]
                outputs = response_generator.generate_responses(
                    batch=queries,
                    n_responses=1,
                    prompts=prompts,
                    override_temperature=0.0,
                    do_sample_override=False,
                    max_new_tokens=max_new_tokens,
                    chunk_size=chunk_size,
                )

                for (q_out, p_out, resp), ex in zip(outputs, batch):
                    result = {
                        "instruction": q_out,
                        "output": resp
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        if unwrapped:
            response_generator.model = orig_model


def dump_tora_jsonl(response_generator, dataset_manager, dataset_name: str,
                            batch_size: int = 16,
                            max_new_tokens: int = 256,
                            chunk_size: int = 4):
    # unwrap DDP if needed (so generate works)
    orig_model = getattr(response_generator, "model", None)
    unwrapped = False
    try:
        if isinstance(orig_model, DDP):
            response_generator.model = orig_model.module
            unwrapped = True

        test_set = dataset_manager.get_test_data(dataset_name)
        n = len(test_set)
        print(f'len dev: {n}')
        print(test_set[0])

        results = []
        # with open(out_path, "w", encoding="utf-8") as fout:
        for i in range(0, n, batch_size):
            batch = test_set[i:i+batch_size]
            # build queries -> use your existing generate_prompt (you said you have one)
            queries = []
            for ex in batch:
                # try common keys for question text
                if "question" in ex:
                    queries.append(ex["question"])
                elif "problem" in ex:
                    queries.append(ex["problem"])
                elif "input" in ex:
                    queries.append(ex["input"])
                else:
                    # fallback: stringify the whole example (shouldn't be necessary)
                    queries.append(str(ex))

            prompts = [response_generator.generate_prompt(q) for q in queries]

            outputs = response_generator.generate_responses(
                batch=queries,
                n_responses=1,
                prompts=prompts,
                override_temperature=0.0,
                do_sample_override=False,
                max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,
            )

            # outputs is list of (query, prompt, resp) with length == len(queries) when n_responses==1
            if len(outputs) != len(queries):
                # 一般不会发生，但打印警告以便排查
                print(f"[Warning] generate_responses returned {len(outputs)} outputs for {len(queries)} queries at batch start {i}")

            for (q_out, p_out, resp), ex in zip(outputs, batch):
                # fetch gold answer from common keys; ToRA expects key 'answer'
                if "answer" in ex:
                    gold = ex["answer"]
                elif "output" in ex:
                    gold = ex["output"]
                elif "label" in ex:
                    gold = ex["label"]
                else:
                    raise KeyError("dataset example missing gold field: expected one of ['answer','output','label']")

                gold = parse(gold)
                pred = parse(resp)
                correct = verify(gold, pred)
                print(f'gold: {gold}, pred: {pred}, correct: {correct}')

                results.append({
                    "answer": gold,
                    "pred": pred,
                    "correct": correct
                })
                # fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        cnt = 0
        for result in results:
            if result['correct']:
                cnt += 1
        total = len(results)
        print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")

    finally:
        if unwrapped:
            response_generator.model = orig_model


def parse_yes_no(answer: str):
    """
    Parse a model's free-form answer into 'yes' or 'no'.
    Returns 'yes', 'no', or None if cannot decide.
    """
    if not answer or not isinstance(answer, str):
        return None

    text = answer.strip().lower()

    # Direct yes/no
    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"

    # Synonyms / paraphrases
    yes_patterns = [r"\btrue\b", r"\bcorrect\b", r"\bindeed\b", r"\baffirmative\b", r"\babsolutely\b"]
    no_patterns = [r"\bfalse\b", r"\bincorrect\b", r"\bnegative\b", r"\bnot really\b", r"\bnah\b"]

    for pat in yes_patterns:
        if re.search(pat, text):
            return "yes"

    for pat in no_patterns:
        if re.search(pat, text):
            return "no"

    # Fallback: look at first word
    first_word = text.split()[0]
    if first_word in ["yes", "yeah", "yep", "yup"]:
        return "yes"
    if first_word in ["no", "nope", "nah"]:
        return "no"

    return None


def compute_strategyqa_accuracy(response_generator, dataset_manager, dataset_name: str, batch_size: int = 8):
    # —— 临时 unwrap DDP ——
    # 记录原模型
    orig_model = response_generator.model
    # 如果是 DDP，就拿到 module
    if isinstance(orig_model, DDP):
        response_generator.model = orig_model.module

    test_set = dataset_manager.get_dev_data(dataset_name)
    print(f'len dev: {len(test_set)}')
    print(test_set[0])

    correct = 0
    total = 0
    unknown = 0
    for i in range(0, len(test_set), batch_size):
        batch = test_set[i:i + batch_size]
        queries = [ex['question'] for ex in batch]
        gold_labels = ["yes" if ex["answer"] else "no" for ex in batch]

        prompts = [response_generator.generate_prompt(q) for q in queries]
        outputs = response_generator.generate_responses(
            batch=queries,
            n_responses=1,
            prompts=prompts,
            override_temperature=0.0,
            do_sample_override=False,
            max_new_tokens=256,
            chunk_size=4,
        )
        for (q, p, resp), gold in zip(outputs, gold_labels):
            print(p)
            print(resp)
            print('=' * 100)
            pred = parse_yes_no(resp)
            if not pred:
                unknown += 1
                continue
            if pred == gold:
                correct += 1
            total += 1

    # —— 恢复原模型 ——
    response_generator.model = orig_model

    accuracy = correct / total if total > 0 else 0.0
    acc = correct / len(test_set)
    print(f"StrategyQA Test Accuracy: {accuracy * 100:.2f}%, {acc * 100:.2f}% (unknown skipped: {unknown})", flush=True)
    return accuracy


# from xfinder import Evaluator
def dump_tora_jsonl_xfinder(response_generator, dataset_manager,
                            dataset_name="mmlu", batch_size=16,
                            max_new_tokens=256, chunk_size=4,
                            model_path_or_url="IAAR-Shanghai/xFinder-qwen1505",
                            inference_mode="local",
                            out_json="log/mmlu_RM3.json"):
    # unwrap DDP if present (minimal)
    orig_model = getattr(response_generator, "model", None)
    if hasattr(orig_model, "module"):
        response_generator.model = orig_model.module

    test_set = dataset_manager.get_test_data(dataset_name)
    print(f'len test: {len(test_set)}')
    all_records = []
    for i in range(0, len(test_set), batch_size):
        batch = test_set[i:i+batch_size]
        queries = [ex['question'] for ex in batch]
        prompts = [response_generator.generate_prompt(q) for q in queries]
        outputs = response_generator.generate_responses(
            batch=queries,
            n_responses=1,
            prompts=prompts,
            override_temperature=0.0,
            do_sample_override=False,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
        )
        for (q, p, resp), ex in zip(outputs, batch):
            opts = ex.get('choices')
            pairs = [[chr(65 + j), opts[j]] for j in range(len(opts))]
            # gold in your dataset is option-text (e.g. '4'); convert to letter
            ans_text = ex.get('answer', '')
            if ans_text in opts:
                idx = opts.index(ans_text)
            else:
                idx = 0
            gold_letter = chr(65 + idx)
            rec = {
                "question": ex['original_q'],
                "llm_output": resp,
                "key_answer_type": "alphabet_option",
                "standard_answer_range": json.dumps(pairs, ensure_ascii=False),
                "correct_answer": gold_letter
            }
            all_records.append(rec)

    # 一次性写入整个 JSON 文件
    with open(out_json, "w", encoding="utf-8") as fout:
        json.dump(all_records, fout, ensure_ascii=False, indent=2)

    # # run evaluator (will accept HF repo name or local path or api url)
    # ev = Evaluator(model_name="xFinder-qwen1505",
    #                inference_mode=inference_mode,
    #                model_path_or_url=model_path_or_url)
    # result = ev.evaluate(out_jsonl)
    # print(f'final result: {result}')

    # restore model if we unwrapped
    if hasattr(orig_model, "module"):
        response_generator.model = orig_model

    # return result



if __name__ == "__main__":
    world_size = 2
    from huggingface_hub import login
    login(token="hfxxx")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
