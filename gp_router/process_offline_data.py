import os
import json
import random
from collections import defaultdict, Counter, deque
from copy import deepcopy
from typing import List, Dict, Tuple, Set

Dir = "/data/cs.aau.dk/zh45qz/router_data/helpsteer3/"


# def balance_bt_pairs_globally(bt_pairs_all: List[Tuple[int, int, str]], rm_indices: List[int], seed=42):
#     random.seed(seed)
#
#     # 按 RM 统计所有赢和输的 BT pair 列表
#     rm_wins_list = defaultdict(list)
#     rm_losses_list = defaultdict(list)
#     for i, j, sid in bt_pairs_all:
#         rm_wins_list[i].append((i, j, sid))
#         rm_losses_list[j].append((i, j, sid))
#
#     # 计算每个 RM 赢输数量
#     rm_win_counts = {rm: len(rm_wins_list[rm]) for rm in rm_indices}
#     rm_loss_counts = {rm: len(rm_losses_list[rm]) for rm in rm_indices}
#     print("Original RM win/loss counts:")
#     for rm in rm_indices:
#         print(f"RM{rm}: wins={rm_win_counts[rm]}, losses={rm_loss_counts[rm]}")
#
#     # 取每个 RM 赢输数量的最小值，作为全局目标
#     win_loss_mins = [min(rm_win_counts[rm], rm_loss_counts[rm]) for rm in rm_indices]
#     global_min = min(win_loss_mins)
#     print(f"Global balanced count per RM win/loss: {global_min}")
#
#     # 随机打乱所有 BT pairs，统一采样
#     bt_pairs_shuffled = bt_pairs_all[:]
#     random.shuffle(bt_pairs_shuffled)
#
#     selected_pairs = set()
#     rm_win_selected_count = Counter()
#     rm_loss_selected_count = Counter()
#
#     for p in bt_pairs_shuffled:
#         winner, loser, sid = p
#         # 检查能否为双方计数加一
#         if rm_win_selected_count[winner] < global_min and rm_loss_selected_count[loser] < global_min:
#             selected_pairs.add(p)
#             rm_win_selected_count[winner] += 1
#             rm_loss_selected_count[loser] += 1
#
#         # 提前终止条件
#         if all(rm_win_selected_count[rm] >= global_min and rm_loss_selected_count[rm] >= global_min for rm in
#                rm_indices):
#             break
#
#     print("Balanced BT pairs - Per-RM win/loss stats after global balancing:")
#     for rm in rm_indices:
#         print(
#             f"  RM{rm}: wins={rm_win_selected_count[rm]}, losses={rm_loss_selected_count[rm]}, total={rm_win_selected_count[rm] + rm_loss_selected_count[rm]}")
#
#     return selected_pairs


def balance_bt_pairs_optimal(samples, num_models, seed=42):
    random.seed(seed)

    # 1) 构造 edges 列表与索引结构
    edges = []  # entries: (s_idx, p_idx, w, l)
    for s_idx, s in enumerate(samples):
        for p_idx, (w, l) in enumerate(s.get("bt_pairs", [])):
            edges.append((s_idx, p_idx, int(w), int(l)))
    E = len(edges)
    if E == 0:
        return samples

    # 辅助结构：winner -> list of edge indices
    winner_to_edges = [[] for _ in range(num_models)]
    # also keep mapping from edge_idx -> (s_idx,p_idx,w,l)
    for e_idx, (s_idx, p_idx, w, l) in enumerate(edges):
        winner_to_edges[w].append(e_idx)

    alive = [True] * E
    to_delete = set()

    # compute initial win/loss/diff from edges (更可靠)
    win_counts = [0] * num_models
    loss_counts = [0] * num_models
    for _, _, w, l in edges:
        win_counts[w] += 1
        loss_counts[l] += 1
    diff = [win_counts[i] - loss_counts[i] for i in range(num_models)]

    # helper to apply deletion of edge e_idx
    def delete_edge(e_idx):
        if not alive[e_idx]:
            return False
        s_idx, p_idx, w, l = edges[e_idx]
        alive[e_idx] = False
        to_delete.add((s_idx, p_idx))
        # update diffs
        diff[w] -= 1   # winner loses one win
        diff[l] += 1   # loser loses one loss -> its diff increases
        return True

    # Phase loop: try to improve until不再有改进
    changed = True
    iter_count = 0
    while changed:
        iter_count += 1
        changed = False

        # -------- Phase 1: 删除能同时改善两个节点的边 (diff[w]>0 and diff[l]<0) --------
        # 使用队列：所有当前 diff>0 的 winner
        q = deque([i for i in range(num_models) if diff[i] > 0])
        # pointer per winner to avoid repeatedly scanning same edges
        next_ptr = [0] * num_models

        while q:
            w = q.popleft()
            if diff[w] <= 0:
                continue
            ptr = next_ptr[w]
            edges_list = winner_to_edges[w]
            # scan outgoing edges for w
            while ptr < len(edges_list) and diff[w] > 0:
                e_idx = edges_list[ptr]
                ptr += 1
                if not alive[e_idx]:
                    continue
                _, _, _, l = edges[e_idx]
                if diff[l] < 0:
                    # good edge to delete
                    delete_edge(e_idx)
                    changed = True
                    # if loser now positive, add it to queue to process its outgoing edges
                    if diff[l] > 0:
                        q.append(l)
                # else skip, keep scanning
            next_ptr[w] = ptr
            # if w still has surplus and we exhausted its list, nothing more can be done in phase1
            # queue continues with other winners

        # if no change in phase1 and currently all diffs in [-1,1], we can stop
        if not changed:
            max_abs = max(abs(x) for x in diff)
            if max_abs <= 1:
                break

        # -------- Phase 2: 需要把剩余的 surplus (diff>0) 转移出去
        # 选择对 loser 影响最小的边（loser diff 最小）来删除
        winners = [i for i in range(num_models) if diff[i] > 0]
        # Prepare list of candidate alive edges for these winners
        # For each winner, collect (loser_diff, e_idx) 并按 loser_diff 升序
        for w in winners:
            if diff[w] <= 0:
                continue
            # collect alive outgoing edges
            cand = []
            for e_idx in winner_to_edges[w]:
                if not alive[e_idx]:
                    continue
                _, _, _, l = edges[e_idx]
                # we prefer smaller loser diff (minimize making someone too positive)
                cand.append((diff[l], e_idx))
            if not cand:
                continue
            # sort by loser diff ascending, tie broken random for variability
            cand.sort(key=lambda x: (x[0], random.random()))
            # delete up to diff[w] edges (or until cand exhausted)
            need = diff[w]
            for loser_diff, e_idx in cand:
                if need <= 0:
                    break
                if not alive[e_idx]:
                    continue
                delete_edge(e_idx)
                changed = True
                need -= 1
                # if the loser becomes positive, we will process it in next Phase1 loop
            # continue for next winner

        # after Phase2 we loop back: Phase1 can now find new useful deletions
        # safety: avoid infinite loop by limiting iterations (should converge quickly)
        if iter_count > 1000:
            # 万一非常罕见的情况没收敛，强制退出
            break

    # 应用删除：把 to_delete 中的 (s_idx,p_idx) 从 samples 的 bt_pairs 里删掉
    if to_delete:
        for s_idx, s in enumerate(samples):
            old_pairs = s.get("bt_pairs", [])
            if not old_pairs:
                continue
            new_pairs = []
            for p_idx, pair in enumerate(old_pairs):
                if (s_idx, p_idx) not in to_delete:
                    new_pairs.append(pair)
            s["bt_pairs"] = new_pairs

    # 最后统计并打印
    final_win = [0] * num_models
    final_loss = [0] * num_models
    for s in samples:
        for w, l in s.get("bt_pairs", []):
            final_win[int(w)] += 1
            final_loss[int(l)] += 1

    print("balance_bt_pairs_optimal 结果：")
    for i in range(num_models):
        print(f"RM{i}: win={final_win[i]}, loss={final_loss[i]}, diff={final_win[i]-final_loss[i]}")

    return samples


# Load original dataset
with open(os.path.join(Dir, "all_samples.json"), "r", encoding="utf-8") as f:
    full_samples = json.load(f)
print(f"full samples: {len(full_samples)}")

# Filter good samples
samples = [ex for ex in full_samples if ex['tag'] == 'good sample']
print(len(samples))

# # Deduplicate using (question, chosen, rejected)
# unique_samples_dict = {}
# for sample in samples:
#     key = (sample['question'], sample['chosen_answer'], sample['rejected_answer'])
#     if key not in unique_samples_dict:
#         unique_samples_dict[key] = sample
# unique_samples = list(unique_samples_dict.values())
# print(f"len unique: {len(unique_samples)}")

# # Filter to samples evaluated by 8B RMs
# unique_samples_8B = [s for s in unique_samples if any(s['int_labels'][i] == 1 for i in [-3, -2, -1])]
# print(f"unique_samples_8B: {len(unique_samples_8B)}")

# # De-duplicate by question 这里后面需要纠正一下，否则会删掉太多样本；
# query_dict = {}
# for sample in unique_samples:
#     q = sample['question']
#     if q not in query_dict:
#         query_dict[q] = sample
# unique_queries = list(query_dict.values())
# print(f"Unique filtered samples: {len(unique_queries)}")


rm_indices = [1, 3, 4, 7]  # RM0, RM1, RM2
# Step 1: Extract all samples with at least one RM pair disagreement
bt_candidate_samples = []
for sample in samples:
    labels = sample['int_labels']
    has_disagreement = any(
        labels[i] != labels[j]
        for i in rm_indices for j in rm_indices if i < j
    )
    if has_disagreement:
        bt_candidate_samples.append(sample)
print(f"Samples with RM disagreement (BT candidates): {len(bt_candidate_samples)}")

# Step 2: Generate all BT pairs and track RM wins/losses
bt_pairs_all = []  # (winner_rm, loser_rm, sample_id)
rm_wins = defaultdict(int)
rm_losses = defaultdict(int)
for sample in bt_candidate_samples:
    sid = sample['id']
    labels = sample['int_labels']
    for i in rm_indices:
        for j in rm_indices:
            if i >= j:
                continue
            if labels[i] == 1 and labels[j] == 0:
                bt_pairs_all.append((i, j, sid))
                rm_wins[i] += 1
                rm_losses[j] += 1
            elif labels[i] == 0 and labels[j] == 1:
                bt_pairs_all.append((j, i, sid))
                rm_wins[j] += 1
                rm_losses[i] += 1
print(f"Total BT pairs: {len(bt_pairs_all)}")


# 构造映射：真实 RM id -> 本地索引，以及反向映射
rm_to_local = {rm: idx for idx, rm in enumerate(rm_indices)}
local_to_rm = {v: k for k, v in rm_to_local.items()}
# 把 bt_pairs_all 按 sample id 聚合并映射 winner/loser 到本地索引
sid_to_pairs = defaultdict(list)
for w, l, sid in bt_pairs_all:
    # 如果某对包含不在 rm_indices 的 RM，可以选择跳过并打印警告
    if w not in rm_to_local or l not in rm_to_local:
        print(f"Warning: skipping pair with RM not in rm_indices: {(w,l,sid)}")
        continue
    sid_to_pairs[sid].append((rm_to_local[w], rm_to_local[l]))
# 构造 samples_for_balance（第二版函数需要的格式）
samples_for_balance = [{"id": sid, "bt_pairs": pairs} for sid, pairs in sid_to_pairs.items()]
# 调用第二版 balance 函数（num_models = 本地 RM 个数）
samples_for_balance = balance_bt_pairs_optimal(samples_for_balance, num_models=len(rm_indices), seed=42)
# 从返回的 samples_for_balance 里提取平衡后的 pairs，并映射回真实 RM id
bt_pairs_balanced = set()
for s in samples_for_balance:
    sid = s["id"]
    for w_loc, l_loc in s.get("bt_pairs", []):
        # 把本地索引映回到真实 RM id
        bt_pairs_balanced.add((local_to_rm[int(w_loc)], local_to_rm[int(l_loc)], sid))




# Step 3: Globally balance BT pairs so that each RM has ~equal wins and losses
# bt_pairs_balanced = balance_bt_pairs_optimal(bt_pairs_all, rm_indices)
# print(f"BT pairs after strict RM win/loss balancing: {len(bt_pairs_balanced)}")
bt_sample_ids = set(sid for _, _, sid in bt_pairs_balanced)
all_disagree_sample_ids = set(s['id'] for s in bt_candidate_samples)


# Step 4: Build sample id -> sample lookup
id_to_sample = {s['id']: deepcopy(s) for s in samples}
extra_clsreg_sample_ids = set(id_to_sample.keys()) - bt_sample_ids


# Step 5: Attach bt_pairs info to samples
sample_to_bt_pairs = defaultdict(list)
for i, j, sid in bt_pairs_balanced:
    sample_to_bt_pairs[sid].append([i, j])


# Step 6: Construct masked labels for cls/reg on all useful samples
def balance_clsreg_and_mask(sample_ids: Set[str], rm_indices: List[int], seed=42):
    random.seed(seed)
    selected_samples = [id_to_sample[sid] for sid in sample_ids]
    rm_to_pos_neg = {i: {'pos': [], 'neg': []} for i in rm_indices}

    for idx, s in enumerate(selected_samples):
        for i in rm_indices:
            label = s['int_labels'][i]
            if label == 1:
                rm_to_pos_neg[i]['pos'].append(idx)
            elif label == 0:
                rm_to_pos_neg[i]['neg'].append(idx)

    # 新结构：记录每个样本参与哪些 RM 的训练（不再是统一 index 集）
    sample_idx_to_kept_rms = defaultdict(set)
    for i in rm_indices:
        pos_list = rm_to_pos_neg[i]['pos']
        neg_list = rm_to_pos_neg[i]['neg']
        n = min(len(pos_list), len(neg_list))
        pos_sampled = random.sample(pos_list, n)
        neg_sampled = random.sample(neg_list, n)
        for idx in pos_sampled + neg_sampled:
            sample_idx_to_kept_rms[idx].add(i)

    print(f"Final samples selected for cls/reg: {len(sample_idx_to_kept_rms)}")
    result = []
    for idx, s in enumerate(selected_samples):
        new_item = deepcopy(s)
        sid = new_item['id']
        new_item['bt_pairs'] = sample_to_bt_pairs.get(sid, [])
        new_item['masked_int_labels'] = []
        new_item['masked_score_diffs'] = []
        for i in range(len(s['int_labels'])):
            if i in rm_indices and i in sample_idx_to_kept_rms[idx]:
                new_item['masked_int_labels'].append(s['int_labels'][i])
                new_item['masked_score_diffs'].append(s['score_diffs'][i])
            else:
                new_item['masked_int_labels'].append(-1000)
                new_item['masked_score_diffs'].append(-1000.0)
        result.append(new_item)

    # 统计每个 RM 的训练样本数量
    rm_sample_counts = {i: {'pos': 0, 'neg': 0} for i in rm_indices}
    for s in result:
        for i in rm_indices:
            label = s['masked_int_labels'][i]
            if label == -1000:
                continue  # 未参与训练，跳过
            if label == 1:
                rm_sample_counts[i]['pos'] += 1
            elif label == 0:
                rm_sample_counts[i]['neg'] += 1
            else:
                raise ValueError(f"Unexpected masked_int_label: {label}")
    print("Per-RM training samples (for cls/reg):")
    for rm in rm_indices:
        pos = rm_sample_counts[rm]['pos']
        neg = rm_sample_counts[rm]['neg']
        print(f"  RM{rm}: pos={pos}, neg={neg}, total={pos + neg}")

    return result


clsreg_sample_ids = bt_sample_ids.union(extra_clsreg_sample_ids)
final_samples = balance_clsreg_and_mask(clsreg_sample_ids, rm_indices)  # 这里改成unique_queries？

# # 未平衡版：直接取 id_to_sample 中对应的样本，并添加 bt_pairs
# final_samples = []
# for sid in clsreg_sample_ids:
#     s = deepcopy(id_to_sample[sid])
#     s['bt_pairs'] = sample_to_bt_pairs.get(sid, [])
#     # masked_int_labels 和 masked_score_diffs 全部保留原值（不mask）
#     s['masked_int_labels'] = s['int_labels'][:]
#     s['masked_score_diffs'] = s['score_diffs'][:]
#     final_samples.append(s)



random.shuffle(final_samples)
# for sample in final_samples:
#     sample['masked_int_labels'] = sample['int_labels']
#     sample['masked_score_diffs'] = sample['score_diffs']

# with open(Dir + "train_helpsteer_only1347.json", "w") as f:
#     json.dump(final_samples, f, indent=2)

