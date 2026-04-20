import sys
sys.path.append("/workspace/guest/cyh/workspace/PepCCD")

import argparse
import csv
import json
import os
import subprocess
import time
import traceback
from itertools import combinations

os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("DISABLE_JAX", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import AutoTokenizer

from train.align_train import ESMModel
from utils.script_utils import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
HOTSPOT_RESIDUES = {"W", "Y", "R"}
BIOACTIVE_RESIDUES = {"W", "Y", "F", "L", "I", "V", "M", "K", "R", "A"}
PKA_SIDE_CHAIN = {
    "C": 8.18,
    "D": 3.65,
    "E": 4.25,
    "H": 6.00,
    "K": 10.53,
    "R": 12.48,
    "Y": 10.07,
}
PKA_N_TERM = 8.0
PKA_C_TERM = 3.1


class UncertaintyHead(nn.Module):
    def __init__(self, input_dim=640):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class PretainedTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./checkpoints/ESM")
        self.vocab = self.tokenizer.get_vocab()
        self.rev_vocab = {idx: word for word, idx in self.vocab.items()}

    def batch_encode(self, seqs):
        return self.tokenizer(
            seqs,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )["input_ids"]

    def batch_decode(self, indices):
        decoded = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.detach().cpu().tolist()
            elif isinstance(idx, np.ndarray):
                idx = idx.tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(idx, skip_special_tokens=True)
            cleaned = "".join(token.replace(" ", "") for token in tokens if token and token not in {"<pad>", "<cls>", "<eos>", "<mask>"})
            decoded.append(cleaned)
        return "".join(decoded)

    def get_valid_length(self, indices):
        special_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.mask_token_id,
        }
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()

        valid_count = 0
        for token_id in indices:
            if token_id in special_tokens:
                continue
            token = self.rev_vocab.get(token_id)
            if token in AMINO_ACIDS:
                valid_count += 1
        return valid_count


def clean_peptide_sequence(seq):
    return "".join(aa for aa in seq.replace(" ", "") if aa in AMINO_ACIDS)


def estimate_net_charge(seq, ph=7.4):
    seq = clean_peptide_sequence(seq)
    if not seq:
        return 0.0

    positive = (
        1.0 / (1.0 + 10 ** (ph - PKA_N_TERM))
        + sum(1.0 / (1.0 + 10 ** (ph - PKA_SIDE_CHAIN[aa])) for aa in seq if aa in {"K", "R", "H"})
    )
    negative = (
        1.0 / (1.0 + 10 ** (PKA_C_TERM - ph))
        + sum(1.0 / (1.0 + 10 ** (PKA_SIDE_CHAIN[aa] - ph)) for aa in seq if aa in {"D", "E", "C", "Y"})
    )
    return float(positive - negative)


def hotspot_ratio(seq):
    seq = clean_peptide_sequence(seq)
    if not seq:
        return 0.0
    return float(sum(aa in HOTSPOT_RESIDUES for aa in seq) / len(seq))


def distinct_n(sequences, n):
    total = 0
    seen = set()
    for seq in sequences:
        seq = clean_peptide_sequence(seq)
        if len(seq) < n:
            continue
        grams = [seq[i:i + n] for i in range(len(seq) - n + 1)]
        total += len(grams)
        seen.update(grams)
    if total == 0:
        return 0.0
    return len(seen) / total


def normalized_edit_similarity(a, b):
    a = clean_peptide_sequence(a)
    b = clean_peptide_sequence(b)
    if not a and not b:
        return 1.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    dist = dp[n]
    return 1.0 - (dist / max(m, n))


def pairwise_diversity(sequences):
    cleaned = [clean_peptide_sequence(seq) for seq in sequences if clean_peptide_sequence(seq)]
    if len(cleaned) < 2:
        return 0.0
    scores = [1.0 - normalized_edit_similarity(a, b) for a, b in combinations(cleaned, 2)]
    return float(np.mean(scores)) if scores else 0.0


def pairwise_similarity(sequences):
    cleaned = [clean_peptide_sequence(seq) for seq in sequences if clean_peptide_sequence(seq)]
    if len(cleaned) < 2:
        return 0.0
    scores = [normalized_edit_similarity(a, b) for a, b in combinations(cleaned, 2)]
    return float(np.mean(scores)) if scores else 0.0


def sequence_similarity_to_native(generated, native):
    return normalized_edit_similarity(generated, native)


def instability_index(seq):
    seq = clean_peptide_sequence(seq)
    if len(seq) < 2:
        return 0.0
    try:
        return float(ProteinAnalysis(seq).instability_index())
    except Exception:
        return 0.0


def bioactivity_score(seq):
    seq = clean_peptide_sequence(seq)
    if not seq:
        return 0.0

    try:
        analysis = ProteinAnalysis(seq)
        aromaticity = float(analysis.aromaticity())
        gravy = float(analysis.gravy())
    except Exception:
        aromaticity = 0.0
        gravy = 0.0

    charge = estimate_net_charge(seq)
    instability = instability_index(seq)
    bioactive_fraction = sum(aa in BIOACTIVE_RESIDUES for aa in seq) / len(seq)

    charge_term = float(np.exp(-((charge - 2.5) ** 2) / (2 * 1.5 ** 2)))
    aromatic_term = min(1.0, aromaticity / 0.20) if aromaticity > 0 else 0.0
    hydrophobic_term = float(np.exp(-((gravy - 0.2) ** 2) / (2 * 0.8 ** 2)))
    stability_term = 1.0 / (1.0 + np.exp((instability - 45.0) / 8.0))

    score = (
        0.30 * charge_term
        + 0.20 * aromatic_term
        + 0.25 * hydrophobic_term
        + 0.15 * bioactive_fraction
        + 0.10 * stability_term
    )
    return float(np.clip(score, 0.0, 1.0))


def amino_acid_freq(seq):
    seq = clean_peptide_sequence(seq)
    if not seq:
        return {aa: 0.0 for aa in sorted(AMINO_ACIDS)}
    total = len(seq)
    return {aa: seq.count(aa) / total for aa in sorted(AMINO_ACIDS)}


def gacd(seq_a, seq_b):
    freq_a = amino_acid_freq(seq_a)
    freq_b = amino_acid_freq(seq_b)
    return float(np.mean([abs(freq_a[aa] - freq_b[aa]) for aa in sorted(AMINO_ACIDS)]))


def create_argparser():
    defaults = dict(
        test_csv="./dataset/PepFlow/pepflow_test.csv",
        output_dir="./evaluation/pepflow_eval",
        max_targets=24,
        num_samples=4,
        batch_size=4,
        seq_len=50,
        min_length=5,
        max_length=50,
        use_ddim=True,
        clip_denoised=True,
        use_fp16=False,
        baseline_model_path="./checkpoints/UR_PepCCD_MoE_pepflow/ema_0.9999_1040000.pt",
        baseline_uncertainty_head_path="./checkpoints/UR_PepCCD_MoE_pepflow/uncertainty_head.pt",
        rl_model_path="./checkpoints/UR_PepCCD_MoE/rl_finetuned_diffusion.pt",
        rl_uncertainty_head_path="./checkpoints/UR_PepCCD_MoE/rl_uncertainty_head.pt",
        baseline_tag="baseline_stage3",
        rl_tag="rl_final",
        seed=20260317,
        candidate_batches=1,
        rerank_top_k=0,
        rerank_bind_weight=0.85,
        rerank_charge_weight=0.15,
        rerank_margin_weight=0.0,
        rerank_charge_target=2.5,
        pep_encoder_path="./checkpoints/Align/best_pep.pth",
        prot_encoder_path="./checkpoints/Align/best_prot.pth",
        enable_structure_metrics=False,
        structure_eval_python="/home/gguser/xiehao/envs/structure_eval/bin/python",
        structure_eval_script="./evaluation/structure_eval.py",
        raw_pepflow_root="/workspace/guest/cyh/Dataset/PepFlow",
        structure_max_peptides_per_target=1,
        structure_force_rerun=False,
        vocab_size=30,
        class_cond=True,
        cls=2,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_generation_bundle(model_path, uncertainty_head_path, args, device):
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    if args.use_fp16:
        model.convert_to_fp16()

    uncertainty_head = UncertaintyHead(input_dim=args.n_embd).to(device)
    if uncertainty_head_path and os.path.exists(uncertainty_head_path):
        uncertainty_head.load_state_dict(
            torch.load(uncertainty_head_path, map_location=device, weights_only=True)
        )
    uncertainty_head.eval()
    return model, diffusion, uncertainty_head


def sample_for_target(model, diffusion, uncertainty_head, prot_encoder, pep_encoder, tokenizer, prot_seq, args, device, target_pool=None):
    test_tokens = tokenizer.batch_encode([prot_seq]).to(device)
    prot_z = prot_encoder([prot_seq]).to(device)
    u_score = uncertainty_head(prot_z)
    collected = []

    def model_fn(inputs_embeds, timesteps, reference=None, self_condition=None, u_score=None):
        if self_condition is None and reference is not None:
            self_condition = reference[0]
        return model(
            inputs_embeds=inputs_embeds,
            timesteps=timesteps,
            self_condition=self_condition,
            u_score=u_score,
        )

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
    for _ in range(max(1, args.candidate_batches)):
        u_score_batch = u_score.repeat(args.batch_size, 1)
        ref_prot = prot_z.repeat(args.batch_size, 1)
        ref_prot = ref_prot / ref_prot.norm(dim=-1, keepdim=True)
        condition = ref_prot.unsqueeze(1).repeat(1, args.seq_len, 1)
        model_kwargs = {
            "reference": [condition, test_tokens.repeat(args.batch_size, 1)],
            "self_condition": condition,
            "u_score": u_score_batch,
        }
        samples = sample_fn(
            model_fn,
            shape=(args.batch_size, args.seq_len, args.n_embd),
            noise=None,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=False,
        )
        logits = model.get_logits(samples[-1])
        indices = torch.topk(logits, k=1, dim=-1).indices.squeeze(-1)
        for seq in indices:
            valid_length = tokenizer.get_valid_length(seq)
            if args.min_length <= valid_length <= args.max_length:
                decoded = clean_peptide_sequence(tokenizer.batch_decode([seq.detach().cpu().numpy()]))
                if decoded:
                    collected.append(decoded)

    if not collected:
        return [], float(u_score.item())

    unique_candidates = list(dict.fromkeys(collected))
    if args.rerank_top_k > 0:
        with torch.no_grad():
            prot_feat = prot_encoder([prot_seq])
            pep_feat = pep_encoder(unique_candidates)
            bind_scores = F.cosine_similarity(pep_feat, prot_feat, dim=-1)
            margin_scores = torch.zeros_like(bind_scores)
            if args.rerank_margin_weight > 0 and target_pool is not None:
                negative_targets = [seq for seq in target_pool if seq != prot_seq]
                if negative_targets:
                    neg_features = prot_encoder(negative_targets).to(device)
                    neg_features = F.normalize(neg_features, dim=-1)
                    pep_norm = F.normalize(pep_feat, dim=-1)
                    margin_scores = bind_scores - (pep_norm @ neg_features.T).max(dim=1).values
        scored = []
        for seq, bind_score, margin_score in zip(unique_candidates, bind_scores.tolist(), margin_scores.tolist()):
            charge_err = abs(estimate_net_charge(seq) - args.rerank_charge_target)
            charge_score = float(np.exp(-(charge_err ** 2) / 2.0))
            rerank_score = (
                args.rerank_bind_weight * bind_score
                + args.rerank_charge_weight * charge_score
                + args.rerank_margin_weight * margin_score
            )
            scored.append((rerank_score, seq))
        scored.sort(key=lambda item: item[0], reverse=True)
        peptides = [seq for _, seq in scored[:args.rerank_top_k]]
    else:
        peptides = unique_candidates[:args.num_samples]
    return peptides, float(u_score.item())


def evaluate_generated_set(prot_seq, native_pep, generated_peps, pep_encoder, prot_encoder, train_peptides):
    valid_peps = [clean_peptide_sequence(seq) for seq in generated_peps if clean_peptide_sequence(seq)]
    metrics = {
        "num_generated": len(generated_peps),
        "num_valid": len(valid_peps),
        "validity": (len(valid_peps) / len(generated_peps)) if generated_peps else 0.0,
        "uniqueness": 0.0,
        "novelty": 0.0,
        "diversity_edit": 0.0,
        "distinct_1": 0.0,
        "distinct_2": 0.0,
        "bind_mean": 0.0,
        "bind_best": 0.0,
        "native_bind": 0.0,
        "charge_mean": 0.0,
        "charge_in_range_rate": 0.0,
        "charge_abs_error_mean": 0.0,
        "hotspot_mean": 0.0,
        "avg_length": 0.0,
        "seq_similarity_mean": 0.0,
        "seq_similarity_best": 0.0,
        "instability_mean": 0.0,
        "gacd_mean": 0.0,
        "intra_sim": 0.0,
        "inter_sim": 0.0,
        "superior_ratio": 0.0,
        "bioactivity": 0.0,
    }
    if not valid_peps:
        return metrics

    unique_peps = list(dict.fromkeys(valid_peps))
    novelty_count = sum(seq not in train_peptides for seq in unique_peps)
    charges = [estimate_net_charge(seq) for seq in valid_peps]
    hotspots = [hotspot_ratio(seq) for seq in valid_peps]
    lengths = [len(seq) for seq in valid_peps]
    seq_sims = [sequence_similarity_to_native(seq, native_pep) for seq in valid_peps]
    instabilities = [instability_index(seq) for seq in valid_peps]
    gacd_scores = [gacd(seq, native_pep) for seq in valid_peps]
    bioactivities = [bioactivity_score(seq) for seq in valid_peps]

    with torch.no_grad():
        prot_feat = prot_encoder([prot_seq])
        gen_feat = pep_encoder(valid_peps)
        native_feat = pep_encoder([clean_peptide_sequence(native_pep)])
        bind_scores = F.cosine_similarity(gen_feat, prot_feat, dim=-1)
        native_bind = F.cosine_similarity(native_feat, prot_feat, dim=-1).item()

    metrics.update({
        "uniqueness": len(unique_peps) / len(valid_peps),
        "novelty": novelty_count / len(unique_peps),
        "diversity_edit": pairwise_diversity(valid_peps),
        "distinct_1": distinct_n(valid_peps, 1),
        "distinct_2": distinct_n(valid_peps, 2),
        "bind_mean": bind_scores.mean().item(),
        "bind_best": bind_scores.max().item(),
        "native_bind": native_bind,
        "charge_mean": float(np.mean(charges)),
        "charge_in_range_rate": float(np.mean([1.0 if 1.0 <= c <= 4.0 else 0.0 for c in charges])),
        "charge_abs_error_mean": float(np.mean([abs(c - 2.5) for c in charges])),
        "hotspot_mean": float(np.mean(hotspots)),
        "avg_length": float(np.mean(lengths)),
        "seq_similarity_mean": float(np.mean(seq_sims)),
        "seq_similarity_best": float(np.max(seq_sims)),
        "instability_mean": float(np.mean(instabilities)),
        "gacd_mean": float(np.mean(gacd_scores)),
        "intra_sim": pairwise_similarity(valid_peps),
        "bioactivity": float(np.mean(bioactivities)),
    })
    return metrics


def compute_inter_similarity(generated_rows):
    by_target = {}
    for row in generated_rows:
        by_target.setdefault(row["target_id"], []).append(row["generated_peptide"])
    target_ids = sorted(by_target)
    scores = []
    for i, src_id in enumerate(target_ids):
        for dst_id in target_ids[i + 1:]:
            for src_seq in by_target[src_id]:
                for dst_seq in by_target[dst_id]:
                    scores.append(normalized_edit_similarity(src_seq, dst_seq))
    return float(np.mean(scores)) if scores else 0.0


def compute_superior_ratio_by_target(generated_rows, dataset_df, prot_encoder, pep_encoder, device):
    if not generated_rows:
        return {}

    target_to_prot = {row.id: row.prot_seq for row in dataset_df.itertuples(index=False)}
    target_ids = list(target_to_prot.keys())

    with torch.no_grad():
        prot_features = prot_encoder([target_to_prot[target_id] for target_id in target_ids]).to(device)
        prot_features = F.normalize(prot_features, dim=-1)
    target_index = {target_id: idx for idx, target_id in enumerate(target_ids)}

    rows_by_target = {}
    for row in generated_rows:
        rows_by_target.setdefault(row["target_id"], []).append(row)

    superior = {}
    peptide_batch_size = 8
    for target_id, rows in rows_by_target.items():
        peptides = [clean_peptide_sequence(row["generated_peptide"]) for row in rows if clean_peptide_sequence(row["generated_peptide"])]
        if not peptides or target_id not in target_index:
            superior[target_id] = 0.0
            continue

        sim_parts = []
        with torch.no_grad():
            for start in range(0, len(peptides), peptide_batch_size):
                batch_peptides = peptides[start:start + peptide_batch_size]
                pep_features = pep_encoder(batch_peptides).to(device)
                pep_features = F.normalize(pep_features, dim=-1)
                sim_parts.append(pep_features @ prot_features.T)
                del pep_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        sim_matrix = torch.cat(sim_parts, dim=0)

        pos_idx = target_index[target_id]
        pos_scores = sim_matrix[:, pos_idx]
        if sim_matrix.shape[1] == 1:
            superior[target_id] = 1.0
            continue

        neg_mask = torch.ones(sim_matrix.shape[1], dtype=torch.bool, device=sim_matrix.device)
        neg_mask[pos_idx] = False
        neg_scores = sim_matrix[:, neg_mask]
        superior[target_id] = float((pos_scores > neg_scores.max(dim=1).values).float().mean().item())
        del sim_parts, sim_matrix, pos_scores, neg_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return superior


def aggregate_records(records, generated_rows):
    if not records:
        return {}
    numeric_keys = [k for k, v in records[0].items() if isinstance(v, (int, float))]
    summary = {}
    for key in numeric_keys:
        summary[key] = float(np.mean([row[key] for row in records]))
    summary["inter_sim"] = compute_inter_similarity(generated_rows)
    summary["bioactivity_note"] = "heuristic_sequence_level_proxy"
    return summary


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flush_outputs(summary_path, per_target_path, generated_path, summaries, per_target_rows, generated_rows):
    write_json(summary_path, summaries)
    write_csv(per_target_path, per_target_rows)
    write_csv(generated_path, generated_rows)


def append_progress(log_path, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def run_model_eval(tag, model_path, uncertainty_head_path, dataset, train_peptides, args, device, progress_callback=None):
    tokenizer = PretainedTokenizer()
    prot_encoder = ESMModel("./checkpoints/ESM", device)
    prot_encoder.load_state_dict(
        torch.load(args.prot_encoder_path, map_location=device, weights_only=False),
        strict=False,
    )
    prot_encoder.eval()

    pep_encoder = ESMModel("./checkpoints/ESM", device)
    pep_encoder.load_state_dict(
        torch.load(args.pep_encoder_path, map_location=device, weights_only=False),
        strict=False,
    )
    pep_encoder.eval()

    model, diffusion, uncertainty_head = load_generation_bundle(
        model_path, uncertainty_head_path, args, device
    )

    records = []
    generated_rows = []
    total_start = time.perf_counter()
    total_generated = 0
    target_pool = list(dict.fromkeys(dataset["prot_seq"].tolist()))
    for row in dataset.itertuples(index=False):
        target_start = time.perf_counter()
        peptides, u_score = sample_for_target(
            model, diffusion, uncertainty_head, prot_encoder, pep_encoder, tokenizer, row.prot_seq, args, device, target_pool=target_pool
        )
        eval_metrics = evaluate_generated_set(
            row.prot_seq, row.pep_seq, peptides, pep_encoder, prot_encoder, train_peptides
        )
        elapsed = time.perf_counter() - target_start
        eval_metrics.update({
            "u_score": u_score,
            "seconds_per_target": elapsed,
            "id": row.id,
        })
        total_generated += len(peptides)
        records.append(eval_metrics)
        for seq in peptides:
            generated_rows.append({
                "model_tag": tag,
                "target_id": row.id,
                "generated_peptide": seq,
                "u_score": u_score,
            })
        if progress_callback is not None:
            progress_callback(
                tag=tag,
                records=records,
                generated_rows_local=generated_rows,
                total_generated=total_generated,
                elapsed=time.perf_counter() - total_start,
                last_target_id=row.id,
            )

    superior_by_target = compute_superior_ratio_by_target(
        generated_rows, dataset, prot_encoder, pep_encoder, device
    )
    for row in records:
        row["superior_ratio"] = float(superior_by_target.get(row["id"], 0.0))

    summary = aggregate_records(records, generated_rows)
    summary["model_tag"] = tag
    summary["num_targets"] = len(records)
    summary["num_generated_total"] = total_generated
    summary["total_seconds"] = time.perf_counter() - total_start
    summary["seconds_per_target"] = summary["total_seconds"] / max(1, len(records))
    summary["seconds_per_peptide"] = summary["total_seconds"] / max(1, total_generated)
    return summary, records, generated_rows


def main():
    args = create_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    per_target_path = os.path.join(args.output_dir, "per_target_metrics.csv")
    generated_path = os.path.join(args.output_dir, "generated_peptides.csv")
    progress_log_path = os.path.join(args.output_dir, "eval_progress.log")
    error_log_path = os.path.join(args.output_dir, "eval_error.log")
    device = os.environ.get("PEPCCD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    test_df = pd.read_csv(args.test_csv)
    train_df = pd.read_csv("./dataset/PepFlow/pepflow_train.csv")
    if args.max_targets > 0:
        test_df = test_df.head(args.max_targets)
    train_peptides = {clean_peptide_sequence(seq) for seq in train_df["pep_seq"].tolist()}

    summaries = []
    per_target_rows = []
    generated_rows = []

    model_specs = [
        (args.baseline_tag, args.baseline_model_path, args.baseline_uncertainty_head_path),
        (args.rl_tag, args.rl_model_path, args.rl_uncertainty_head_path),
    ]

    append_progress(progress_log_path, f"evaluation started on device={device}")

    def progress_callback(tag, records, generated_rows_local, total_generated, elapsed, last_target_id):
        partial_records = []
        for row in records:
            row_copy = dict(row)
            row_copy["model_tag"] = tag
            partial_records.append(row_copy)
        partial_generated = list(generated_rows_local)

        merged_per_target = [row for row in per_target_rows if row.get("model_tag") != tag] + partial_records
        merged_generated = [row for row in generated_rows if row.get("model_tag") != tag] + partial_generated
        partial_summary = aggregate_records(records, generated_rows_local)
        partial_summary.update({
            "model_tag": tag,
            "num_targets": len(records),
            "num_generated_total": total_generated,
            "total_seconds": elapsed,
            "seconds_per_target": elapsed / max(1, len(records)),
            "seconds_per_peptide": elapsed / max(1, total_generated),
            "partial": True,
        })
        merged_summaries = [row for row in summaries if row.get("model_tag") != tag] + [partial_summary]
        flush_outputs(summary_path, per_target_path, generated_path, merged_summaries, merged_per_target, merged_generated)
        append_progress(
            progress_log_path,
            f"{tag}: completed target {last_target_id} ({len(records)}/{len(test_df)}), generated={total_generated}",
        )

    try:
        for tag, model_path, uncertainty_path in model_specs:
            append_progress(progress_log_path, f"starting model={tag}")
            summary, records, generated = run_model_eval(
                tag, model_path, uncertainty_path, test_df, train_peptides, args, device, progress_callback=progress_callback
            )
            summaries = [row for row in summaries if row.get("model_tag") != tag] + [summary]
            per_target_rows = [row for row in per_target_rows if row.get("model_tag") != tag]
            generated_rows = [row for row in generated_rows if row.get("model_tag") != tag]
            for row in records:
                row["model_tag"] = tag
                per_target_rows.append(row)
            generated_rows.extend(generated)
            flush_outputs(summary_path, per_target_path, generated_path, summaries, per_target_rows, generated_rows)
            append_progress(progress_log_path, f"finished model={tag}")
    except Exception:
        with open(error_log_path, "w") as f:
            f.write(traceback.format_exc())
        append_progress(progress_log_path, f"evaluation failed; traceback saved to {error_log_path}")
        raise

    if args.enable_structure_metrics:
        structure_script = os.path.abspath(args.structure_eval_script)
        append_progress(progress_log_path, "starting structure evaluation")
        subprocess.run(
            [
                args.structure_eval_python,
                structure_script,
                "--summary_json",
                summary_path,
                "--per_target_csv",
                per_target_path,
                "--generated_csv",
                generated_path,
                "--test_csv",
                args.test_csv,
                "--raw_pepflow_root",
                args.raw_pepflow_root,
                "--output_dir",
                args.output_dir,
                "--max_peptides_per_target",
                str(args.structure_max_peptides_per_target),
                "--device",
                device,
            ]
            + (["--force_rerun"] if args.structure_force_rerun else []),
            check=True,
        )
        append_progress(progress_log_path, "finished structure evaluation")

    print(json.dumps(summaries, indent=2))
    print(f"summary saved to {summary_path}")
    print(f"per-target metrics saved to {per_target_path}")
    print(f"generated peptides saved to {generated_path}")


if __name__ == "__main__":
    main()
