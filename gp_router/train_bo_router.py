import re
import gc
import os
import random
import time
import datetime
import torch
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from utils import load_config, DatasetManager, generate_preference_pairs, LLMTrainer, \
    load_reward_models, rpc_select_arm, rpc_update_arm, generate_offline_prior, rpc_offline_batch_encode
from policy_model import setup_lora_model, ResponseGenerator
from torch.nn.parallel import DistributedDataParallel


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12368'  # 12355任意未占用端口
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
            init_method="tcp://127.0.0.1:29503", rpc_timeout=3600
        ),
    )
    config = load_config("config.yaml")
    dataset_name = "strategyqa"

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
        from utils import rpc_init_globals
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

        # Dataset manager
        dataset_manager = DatasetManager({dataset_name: config["datasets"][dataset_name]},
                                         test_size=config["training"]["test_size"],
                                         dev_size=config["training"]["dev_size"],
                                         prompt_type="reasoning")

        # Response generation & trainer, pref generation
        # trainable_params = filter(lambda p: p.requires_grad, policy_model.parameters())
        trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
        trainer = LLMTrainer(policy_model,
                             Adam(trainable_params, lr=float(config["training"]["learning_rate"])),
                             tokenizer,
                             grad_accum_steps=4)
        n_responses = config["training"]["n_responses"]
        response_generator = ResponseGenerator(policy_model, tokenizer, prompt_type="reasoning",
                                               temperature=config["training"]["temperature"])

        # Training loop
        print(f"[RANK {rank}] Training on {dataset_name} dataset")
        M = config["training"]["iterations"]
        batch_size = config["training"]["batch_size"]
        eval_threshold = config["training"]["eval_threshold"]
        for iteration in range(M):
            total_loss = 0
            train_data = dataset_manager.get_train_data(dataset_name)
            for i in range(0, len(train_data), batch_size):
                start_time = time.time()
                print(f'{i}-th batch', flush=True)
                batch = train_data[i:i + batch_size]
                queries = [ex['question'] for ex in batch]

                # m = policy_model.module if isinstance(policy_model, DistributedDataParallel) else policy_model
                # state = {n: p.detach().cpu() for n, p in m.named_parameters()}
                # pairs = []
                # for n in state:
                #     if ".policy." in n:
                #         ref = n.replace(".policy.", ".reference.")
                #         if ref in state:
                #             maxdiff = (state[n] - state[ref]).abs().max().item()
                #             pairs.append((n, ref, maxdiff))
                # # 打印前 20 对
                # for a, b, d in pairs[:20]:
                #     print(f"{a} vs {b} -> max_abs_diff = {d:.6e}")
                # # 打印总体统计
                # diffs = [d for _, _, d in pairs]
                # print("pairs_count", len(diffs), "mean diff", sum(diffs) / len(diffs) if diffs else None, "max diff",
                #       max(diffs) if diffs else None)

                # # 1) Generate responses
                # query_response_pairs = response_generator.generate_responses(queries, n_responses=n_responses, chunk_size=4)
                # time0 = time.time()
                # # print(f'generate responses time: {time0-start_time}')
                # # print(query_response_pairs[0], flush=True)
                #
                # # 2) RPC to get embedding and arm selection from rank0
                # batch_context, batch_prior = generate_offline_prior(query_response_pairs)  # 这个似乎巨耗时？
                # time1 = time.time()
                # # print(f'generate prior time: {time1-time0}')
                #
                # selected_rm_idx = rpc.rpc_sync(to="worker0", func=rpc_select_arm,
                #                                args=(batch_context.detach(),
                #                                      batch_prior.detach(), 0.1))
                # # time2 = time.time()
                # # print(f'select rm time: {time2-time1}')
                # # selected_rm_idx = 1

                # # 3) Compute advantage: baseline minus selected loss
                # all_losses = []
                # all_prefs = []
                # for rm in reward_models:
                #     t0=time.time()
                #     prefs = generate_preference_pairs(rm, query_response_pairs)  # 这比较耗时？
                #     t1=time.time()
                #     # print(f'generate pref pairs time: {t1-t0}')
                #     loss_val = trainer.compute_preference_loss(prefs)  # 这也比较耗时。。
                #     # print(f'compute pref loss time: {time.time()-t1}')
                #     all_losses.append(loss_val)
                #     all_prefs.append(prefs)
                # baseline = sum(all_losses) / len(all_losses)
                # advantage = baseline - all_losses[selected_rm_idx]
                # time3 = time.time()
                # # print(f'advantage: {advantage}')
                # # print(f'compute advantage time: {time3 - time2}')
                #
                # print(f'===reward: {advantage}')
                # # 4) Update BayesianRouter with advantage
                # rpc.rpc_sync(to="worker0", func=rpc_update_arm,
                #              args=(selected_rm_idx, batch_context, advantage))
                # time4 = time.time()
                # # print(f'router update time: {time4-time3}')

                # # 6) Train policy on selected RM
                # sel_prefs = generate_preference_pairs(reward_models[selected_rm_idx], query_response_pairs)
                # loss = trainer.train_step(sel_prefs)
                # rpc.rpc_sync(to="worker0", func=rpc_update_arm,
                #              args=(selected_rm_idx, batch_context, loss))
                # total_loss += loss



                # 1) 生成回答
                query_response_pairs = response_generator.generate_responses(queries, n_responses=n_responses,
                                                                             chunk_size=4)
                # 2) 用 generate_preference_pairs 只负责“采样 pair”，先不贴标签
                # 这里采用某个 RM 来决定 top-mid 的采样分布（例如第 0 个；如果你更想用其它 RM，也只需改这行）
                sampled_pairs = generate_preference_pairs(  # 这里reward_models[0]要改改！
                    reward_models[0],  # 仅用于 top-mid 采样，不用于打标签
                    query_response_pairs,
                    pairs_per_query=4,  # 你的原配置
                    top_k=4, mid_start=8, mid_end=12,
                    return_pairs_only=True  # === 关键：只返回 (q, a, b)，不贴 winner/loser
                )
                if not sampled_pairs:
                    continue
                # 3) 为这些“已选中的 pairs”计算 embedding，并逐 pair 选臂
                queries_p = [q for (q, a, b) in sampled_pairs]
                resp_as = [a for (q, a, b) in sampled_pairs]
                resp_bs = [b for (q, a, b) in sampled_pairs]
                pair_ctx = rpc.rpc_sync(
                    to="worker0",
                    func=rpc_offline_batch_encode,
                    args=(queries_p, resp_as, resp_bs)
                )  # Tensor (N, D)
                # 如需用 prior，可再取一次 logits -> probs；当前 rpc_select_arm 忽略 prior，传 None 也可以
                # pair_logits = rpc.rpc_sync(to="worker0", func=rpc_offline_batch_logits, args=(pair_ctx,))
                # pair_prior  = torch.sigmoid(pair_logits)
                assigned_arms = []
                for i in range(pair_ctx.size(0)):
                    arm_i = rpc.rpc_sync(to="worker0", func=rpc_select_arm, args=(pair_ctx[i].detach(), None, 0.1))
                    assigned_arms.append(int(arm_i))
                # 4) 由“该 pair 被分配的臂（RM）”来贴标签（winner/loser）
                sel_prefs = []
                # assigned_arms = [0, 0, 0, 0]  # 仅选用RM0
                for (q, a, b), arm in zip(sampled_pairs, assigned_arms):
                    rm = reward_models[arm]
                    # 用该 RM 比较 (a, b)
                    scores = rm.batch_score([q, q], [a, b])
                    if scores[0] >= scores[1]:
                        sel_prefs.append((q, a, b))
                    else:
                        sel_prefs.append((q, b, a))
                # 5) 训练：保持 mini-batch/grad-accum 语义不变，同时拿到 per-pair loss 作为奖励
                loss, per_pair_losses = trainer.train_step(sel_prefs)
                # 6) 逐 pair 回传奖励，更新对应臂的后验（EMA baseline & z-score 在 rank0 内部完成）
                for arm, ctx, li in zip(assigned_arms, pair_ctx, per_pair_losses):
                    rpc.rpc_sync(to="worker0", func=rpc_update_arm,
                                 args=(int(arm), ctx.detach(), float(li)))
                total_loss += loss
                print(
                    f"[Rank {rank} | Iter {iteration + 1} | Batch {i // batch_size + 1}]"
                    f" RM={assigned_arms} | Loss={loss:.4f}"
                )
                time5 = time.time()
                # print(f'train step time: {time5-time4}')

            # 处理未完成的梯度累积
            if trainer.accum_step > 0:
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                trainer.accum_step = 0

            avg_loss = total_loss / (len(train_data) // batch_size)
            print(f"Iteration {iteration + 1}, Avg Loss: {avg_loss:.4f}")
            compute_strategyqa_accuracy(response_generator, dataset_manager, "strategyqa", batch_size=8)
            local_flag = torch.tensor(1 if avg_loss < eval_threshold else 0, device=policy_device)
            torch.distributed.all_reduce(local_flag, op=torch.distributed.ReduceOp.MIN, group=policy_group)
            if local_flag.item() == 1:  # 只有当所有 rank 的 local_flag 都是 1，才返回 1
                print("All ranks converged; breaking.")
                break
    # dist.barrier()
    rpc.shutdown()
    dist.destroy_process_group()


# Inference
def extract_yes_no(text):
    text = text.lower()
    # print(text)
    if re.search(r'\byes\b', text):
        return "Yes"
    elif re.search(r'\bno\b', text):
        return "No"
    elif "yes" in text:
        return "Yes"
    elif "no" in text:
        return "No"
    else:
        return "Unknown"


def compute_strategyqa_accuracy(response_generator, dataset_manager, dataset_name: str, batch_size: int = 8):
    # —— 临时 unwrap DDP ——
    # 记录原模型
    orig_model = response_generator.model
    # 如果是 DDP，就拿到 module
    if isinstance(orig_model, DDP):
        response_generator.model = orig_model.module

    test_set = dataset_manager.get_dev_data(dataset_name)
    correct = 0
    total = 0
    unknown = 0

    for i in range(0, len(test_set), batch_size):
        batch = test_set[i:i + batch_size]
        queries = [ex['question'] for ex in batch]
        gold_labels = ["Yes" if ex["answer"] else "No" for ex in batch]

        outputs = response_generator.generate_responses(queries, n_responses=1)

        for (query, response), gold in zip(outputs, gold_labels):
            print(response)
            print('=' * 200)
            pred = extract_yes_no(response)
            if pred == "Unknown":
                unknown += 1
                continue
            if pred == gold:
                correct += 1
            total += 1

    # —— 恢复原模型 ——
    response_generator.model = orig_model

    accuracy = correct / total if total > 0 else 0.0
    print(f"StrategyQA Test Accuracy: {accuracy * 100:.2f}% (unknown skipped: {unknown})", flush=True)
    return accuracy


if __name__ == "__main__":
    world_size = 2
    from huggingface_hub import login
    login(token="xxxx")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
