import sys
import os
import time
sys.path.append(".")  # Ensure project root is importable.
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm  # Progress bar.

from model.resample import create_named_schedule_sampler
from utils.script_utils import model_and_diffusion_defaults, create_model_and_diffusion
from train.align_train import ESMModel
import argparse
from utils.script_utils import add_dict_to_argparser

# --- 1. Uncertainty head ---
import torch.nn as nn
class UncertaintyHead(nn.Module):
    def __init__(self, input_dim=640):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


def append_progress(log_path, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def initialize_robust_mlp_from_spec(model):
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "h"):
        return
    for block in transformer.h:
        if hasattr(block, "mlp") and hasattr(block, "mlp_robust"):
            block.mlp_robust.load_state_dict(block.mlp.state_dict())


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
            cleaned = "".join(
                token.replace(" ", "")
                for token in tokens
                if token and token not in {"<pad>", "<cls>", "<eos>", "<mask>"}
            )
            decoded.append(cleaned)
        return "".join(decoded)

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
HOTSPOT_RESIDUES = {"W", "Y", "R"}
DEFAULT_U_SCORE_TARGET = 0.25
DEFAULT_U_SCORE_REG_WEIGHT = 0.05
DEFAULT_UNCERTAINTY_HEAD_LR = 1e-6
DEFAULT_ADV_TEMPERATURE = 0.35
DEFAULT_PPO_CLIP_EPS = 0.2
DEFAULT_PPO_UPDATES = 3
DEFAULT_REWARD_MODEL_LR = 1e-5
DEFAULT_REWARD_UPDATES = 2
DEFAULT_MARGIN_FLOOR = 0.0
DEFAULT_MARGIN_REWARD_WEIGHT = 0.7
DEFAULT_BIND_REWARD_WEIGHT = 0.3
DEFAULT_MARGIN_PENALTY_WEIGHT = 0.5
DEFAULT_ELBO_AUX_WEIGHT = 0.25


class ImplicitRewardModel(nn.Module):
    def __init__(self, feat_dim=640, hidden_dim=512):
        super().__init__()
        input_dim = feat_dim * 4 + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, prot_feat, pep_feat, elbo_gain, u_score):
        prot_feat = F.normalize(prot_feat, dim=-1)
        pep_feat = F.normalize(pep_feat, dim=-1)
        interaction = prot_feat * pep_feat
        distance = torch.abs(prot_feat - pep_feat)
        if elbo_gain.dim() == 1:
            elbo_gain = elbo_gain.unsqueeze(-1)
        if u_score.dim() == 1:
            u_score = u_score.unsqueeze(-1)
        features = torch.cat([prot_feat, pep_feat, interaction, distance, elbo_gain, u_score], dim=-1)
        return self.net(features).squeeze(-1)


def clean_peptide_sequence(seq):
    return "".join([aa for aa in seq.replace(" ", "") if aa.isalpha() and aa.isupper()])


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
    return positive - negative


def charge_reward(seq, target_charge=2.5, sigma=1.2):
    charge = estimate_net_charge(seq)
    reward = np.exp(-((charge - target_charge) ** 2) / (2 * sigma ** 2))
    return float(reward), float(charge)


def hotspot_reward(seq):
    seq = clean_peptide_sequence(seq)
    if not seq:
        return 0.0, 0.0
    hotspot_ratio = sum(aa in HOTSPOT_RESIDUES for aa in seq) / len(seq)
    # ratio is already in [0, 1], apply a mild amplification.
    reward = min(1.0, hotspot_ratio * 2.5)
    return float(reward), float(hotspot_ratio)


def encode_text_batch(encoder, texts, batch_size=8):
    parts = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            if not chunk:
                continue
            parts.append(encoder(chunk))
    return torch.cat(parts, dim=0) if parts else torch.empty(0, device=encoder.device)


def build_negative_protein_bank(protein_list, prot_encoder, device, batch_size=8):
    unique_proteins = list(dict.fromkeys(protein_list))
    prot_features = encode_text_batch(prot_encoder, unique_proteins, batch_size=batch_size).to(device)
    prot_features = F.normalize(prot_features, dim=-1)
    protein_index = {prot: idx for idx, prot in enumerate(unique_proteins)}
    return unique_proteins, protein_index, prot_features


# --- 2. Reward computation ---
def compute_reward(pep_seqs, prot_seq, pep_encoder, prot_encoder, prot_bank_features, prot_bank_index, u_score, device, args):
    clean_peptides = [clean_peptide_sequence(seq) for seq in pep_seqs]
    with torch.no_grad():
        prot_features = prot_encoder([prot_seq])
        pep_features = encode_text_batch(pep_encoder, clean_peptides, batch_size=8).to(device)
        prot_features = F.normalize(prot_features, dim=-1)
        pep_features = F.normalize(pep_features, dim=-1)
        cos_sim = F.cosine_similarity(pep_features, prot_features, dim=-1)

        if prot_seq in prot_bank_index:
            pos_idx = prot_bank_index[prot_seq]
            sim_matrix = pep_features @ prot_bank_features.T
            pos_scores = sim_matrix[:, pos_idx]
            if sim_matrix.shape[1] > 1:
                neg_mask = torch.ones(sim_matrix.shape[1], dtype=torch.bool, device=device)
                neg_mask[pos_idx] = False
                hardest_negative = sim_matrix[:, neg_mask].max(dim=1).values
            else:
                hardest_negative = torch.zeros_like(pos_scores)
            spec_margin = pos_scores - hardest_negative
        else:
            pos_scores = cos_sim
            hardest_negative = torch.zeros_like(pos_scores)
            spec_margin = pos_scores

    charge_rewards = []
    hotspot_rewards = []
    charges = []
    hotspot_ratios = []
    for seq in clean_peptides:
        c_reward, net_charge = charge_reward(seq)
        h_reward, h_ratio = hotspot_reward(seq)
        charge_rewards.append(c_reward)
        hotspot_rewards.append(h_reward)
        charges.append(net_charge)
        hotspot_ratios.append(h_ratio)

    r_charge = torch.tensor(charge_rewards, device=device, dtype=cos_sim.dtype)
    r_hotspot = torch.tensor(hotspot_rewards, device=device, dtype=cos_sim.dtype)
    r_robust = args.charge_weight * r_charge + args.hotspot_weight * r_hotspot

    # Keep u in a moderate interval to avoid robust-only hard switching.
    u_val = u_score.view(-1)
    u_mix = args.u_min + (args.u_max - args.u_min) * u_val

    spec_reward = args.margin_reward_weight * spec_margin + args.bind_reward_weight * pos_scores
    affinity_gate = torch.sigmoid(12.0 * (spec_margin - args.margin_floor))
    effective_robust = affinity_gate * r_robust + (1.0 - affinity_gate) * torch.clamp(spec_reward, min=0.0)
    affinity_penalty = F.relu(args.bind_floor - cos_sim)
    margin_penalty = F.relu(args.margin_floor - spec_margin)

    total_reward = (
        (1.0 - u_mix) * spec_reward
        + u_mix * effective_robust
        - args.bind_penalty_weight * affinity_penalty
        - args.margin_penalty_weight * margin_penalty
    )
    aux_metrics = {
        "cos_sim": cos_sim.detach(),
        "pos_score": pos_scores.detach(),
        "neg_score": hardest_negative.detach(),
        "spec_margin": spec_margin.detach(),
        "charge_reward": r_charge.detach(),
        "hotspot_reward": r_hotspot.detach(),
        "robust_reward": r_robust.detach(),
        "affinity_gate": affinity_gate.detach(),
        "affinity_penalty": affinity_penalty.detach(),
        "margin_penalty": margin_penalty.detach(),
        "u_mix": u_mix.detach(),
        "net_charge": charges,
        "hotspot_ratio": hotspot_ratios,
        "pep_features": pep_features.detach(),
        "prot_features": prot_features.detach(),
    }
    return total_reward, aux_metrics


def build_condition_from_protein(prot_z, repeat_size, seq_len, device):
    prot_z_batch = prot_z.repeat(repeat_size, 1).to(device)
    ref_prot = prot_z_batch / prot_z_batch.norm(dim=-1, keepdim=True)
    return ref_prot.unsqueeze(1).repeat(1, seq_len, 1).to(device)


def compute_diffusion_candidate_losses(model, diffusion, tokenizer, peptide_texts, prot_z, u_score, schedule_sampler, device, seq_len, n_embd):
    if not peptide_texts:
        return torch.empty(0, device=device)

    input_ids = tokenizer.batch_encode(peptide_texts).to(device)[:, :seq_len].contiguous()
    batch_size = input_ids.shape[0]
    condition = build_condition_from_protein(prot_z, batch_size, seq_len, device)
    model_kwargs = {
        "self_condition": condition,
        "u_score": u_score.repeat(batch_size, 1).to(device),
        "input_ids": input_ids,
    }
    t, _ = schedule_sampler.sample(batch_size, device)
    dummy_x_start = torch.zeros(batch_size, seq_len, n_embd, device=device)
    losses = diffusion.training_losses(model, x_start=dummy_x_start, t=t, model_kwargs=model_kwargs.copy())
    return losses["loss"]


def compute_group_relative_advantages(rewards):
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    leave_one_out = (rewards.sum() - rewards) / (rewards.numel() - 1)
    advantages = rewards - leave_one_out
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    return advantages


def compute_ppo_loss(new_losses, old_losses, advantages, clip_eps):
    # Diffusion surrogate log-prob: log pi(y|x) ~= -L_diffusion(x, y)
    old_logprob = -old_losses.detach()
    new_logprob = -new_losses
    ratios = torch.exp(new_logprob - old_logprob)
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
    surr1 = ratios * advantages
    surr2 = clipped_ratios * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))
    approx_kl = torch.mean(old_logprob - new_logprob).detach()
    clip_frac = torch.mean((torch.abs(ratios - 1.0) > clip_eps).float()).detach()
    return policy_loss, approx_kl, clip_frac


def build_reward_labels(spec_margin, outcome_rewards):
    labels = (spec_margin > 0).float()
    if labels.sum() == 0:
        labels[outcome_rewards.argmax()] = 1.0
    if labels.sum() == labels.numel():
        labels[outcome_rewards.argmin()] = 0.0
    return labels


def train_implicit_reward_model(reward_model, reward_optimizer, prot_feat, pep_feat, elbo_gain, u_value, outcome_labels, args):
    norm_elbo = (elbo_gain - elbo_gain.mean()) / (elbo_gain.std(unbiased=False) + 1e-6)
    elbo_target = torch.sigmoid(norm_elbo)
    last_loss = None
    for _ in range(args.reward_updates):
        reward_optimizer.zero_grad()
        logits = reward_model(
            prot_feat,
            pep_feat,
            norm_elbo.detach(),
            u_value.repeat(pep_feat.shape[0], 1).detach(),
        )
        bce_loss = F.binary_cross_entropy_with_logits(logits, outcome_labels)
        aux_loss = F.mse_loss(torch.sigmoid(logits), elbo_target.detach())
        loss = bce_loss + args.elbo_aux_weight * aux_loss
        loss.backward()
        reward_optimizer.step()
        last_loss = loss.detach()
    return last_loss

def main():
    device = os.environ.get("PEPCCD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting UR-PepCCD PPO fine-tuning on device: {device}")

    # ===== Load arguments and progress logging =====
    from utils.script_utils import args_to_dict
    
    args = create_argparser().parse_args()
    progress_log_path = args.progress_log_path or os.path.splitext(args.save_model_path)[0] + "_progress.log"
    os.makedirs(os.path.dirname(progress_log_path), exist_ok=True)
    append_progress(progress_log_path, f"training started on device={device}")
    
    # Build diffusion model.
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Load Stage-3 EMA checkpoint.
    model.load_state_dict(torch.load(args.stage3_model_path, map_location=device), strict=False)
    initialize_robust_mlp_from_spec(model)
    model.to(device)

    ref_model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    ref_model.load_state_dict(torch.load(args.ref_model_path, map_location=device), strict=False)
    initialize_robust_mlp_from_spec(ref_model)
    ref_model.to(device)
    ref_model.eval()
    
    uncertainty_head = UncertaintyHead(input_dim=640).to(device)
    if os.path.exists(args.stage3_uncertainty_head_path):
        uncertainty_head.load_state_dict(
            torch.load(args.stage3_uncertainty_head_path, map_location=device, weights_only=True)
        )
        print(f">> Loaded Stage-3 uncertainty head: {args.stage3_uncertainty_head_path}")
    else:
        print(f">> Stage-3 uncertainty head not found, training from scratch: {args.stage3_uncertainty_head_path}")
    
    prot_encoder = ESMModel("./checkpoints/ESM", device)
    prot_encoder.load_state_dict(
        torch.load("./checkpoints/Align/best_prot.pth", map_location=device),
        strict=False,
    )
    
    pep_encoder = ESMModel("./checkpoints/ESM", device)
    pep_encoder.load_state_dict(
        torch.load("./checkpoints/Align/best_pep.pth", map_location=device),
        strict=False,
    )

    reward_model = ImplicitRewardModel(feat_dim=prot_encoder.hidden_size).to(device)
    
    myTokenizer = PretainedTokenizer()
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    # ===== Freeze most weights and train selected modules only =====
    for param in model.parameters(): param.requires_grad = False
    for param in prot_encoder.parameters(): param.requires_grad = False
    for param in pep_encoder.parameters(): param.requires_grad = False
    for param in ref_model.parameters(): param.requires_grad = False
        
    robust_params = []
    for i in range(len(model.transformer.h)):
        if hasattr(model.transformer.h[i], 'mlp_robust'):
            for param in model.transformer.h[i].mlp_robust.parameters():
                param.requires_grad = True
                robust_params.append(param)
                
    head_params = []
    for param in uncertainty_head.parameters():
        param.requires_grad = True
        head_params.append(param)

    reward_params = []
    for param in reward_model.parameters():
        param.requires_grad = True
        reward_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": robust_params, "lr": args.robust_lr},
            {"params": head_params, "lr": args.uncertainty_head_lr},
        ]
    )
    reward_optimizer = torch.optim.AdamW(
        [{"params": reward_params, "lr": args.reward_model_lr}]
    )

    # ===== Load training targets =====
    csv_path = "./dataset/PepFlow/pepflow_train.csv"  # Must include column `prot_seq`.
    print(f">> Loading target list: {csv_path}")
    df = pd.read_csv(csv_path)
    protein_list = df['prot_seq'].tolist()
    protein_list = protein_list[:100]
    _, protein_bank_index, protein_bank_features = build_negative_protein_bank(protein_list, prot_encoder, device)
    append_progress(progress_log_path, f"protein bank built with {len(protein_bank_index)} targets")

    epochs = args.epochs
    batch_size = args.batch_size  # Number of sampled candidates per target.
    def model_fn_sample(inputs_embeds, timesteps, self_condition=None, u_score=None):
        return model(inputs_embeds=inputs_embeds, timesteps=timesteps, self_condition=self_condition, u_score=u_score)

    print("\nStarting PPO-style RL training...")
    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch+1}/{epochs} ==========")
        append_progress(progress_log_path, f"epoch {epoch+1}/{epochs} started")
        
        # Iterate over targets with a progress bar.
        pbar = tqdm(protein_list, desc="Training", unit="target")
        
        for step, target_prot in enumerate(pbar):
            target_id = df.iloc[step]["id"] if "id" in df.columns and step < len(df) else f"step_{step}"
            append_progress(progress_log_path, f"epoch {epoch+1}: start target {target_id} ({step+1}/{len(protein_list)})")
            if len(target_prot) > 1024: 
                append_progress(progress_log_path, f"epoch {epoch+1}: skip target {target_id} due to length={len(target_prot)}")
                continue  # Skip overlong proteins to avoid ESM OOM.

            model.eval()
            uncertainty_head.eval()
            
            # [A. Sampling]
            with torch.no_grad():
                test_tokens = myTokenizer.batch_encode([target_prot]).to(device)
                prot_z = prot_encoder([target_prot]).to(device)
                u_score = uncertainty_head(prot_z)
                
                u_score_batch = u_score.repeat(batch_size, 1).to(device)
                condition = build_condition_from_protein(prot_z, batch_size, 50, device)
                
                model_kwargs = {"self_condition": condition, "u_score": u_score_batch}
                
                # Keep sampling silent to avoid noisy logs.
                samples = diffusion.ddim_sample_loop(
                    model_fn_sample, shape=(batch_size, 50, args.n_embd),
                    model_kwargs=model_kwargs, device=device, clip_denoised=True, progress=False
                )
                sample_emb = samples[-1]
                
                logits = model.get_logits(sample_emb)
                _, indices = torch.topk(logits, k=1, dim=-1)
                indices = indices.squeeze(-1)
                
                gen_peptides = [myTokenizer.batch_decode([seq.cpu().numpy()]) for seq in indices]
                gen_peptides = [clean_peptide_sequence(seq) for seq in gen_peptides if clean_peptide_sequence(seq)]
                if not gen_peptides:
                    append_progress(progress_log_path, f"epoch {epoch+1}: target {target_id} generated no valid peptides")
                    continue
            append_progress(progress_log_path, f"epoch {epoch+1}: target {target_id} sampled {len(gen_peptides)} peptides")
            
            # [B. Reward and candidate scoring]
            outcome_rewards, reward_metrics = compute_reward(
                gen_peptides, target_prot, pep_encoder, prot_encoder,
                protein_bank_features, protein_bank_index, u_score, device, args
            )
            with torch.no_grad():
                current_losses = compute_diffusion_candidate_losses(
                    model, diffusion, myTokenizer, gen_peptides, prot_z, u_score,
                    schedule_sampler, device, 50, args.n_embd
                )
                ref_losses = compute_diffusion_candidate_losses(
                    ref_model, diffusion, myTokenizer, gen_peptides, prot_z, u_score,
                    schedule_sampler, device, 50, args.n_embd
                )
                old_losses = current_losses.detach()

            reward_labels = build_reward_labels(reward_metrics["spec_margin"], outcome_rewards.detach())
            prot_feat_batch = reward_metrics["prot_features"].repeat(len(gen_peptides), 1).detach()
            reward_model.train()
            reward_model_loss = train_implicit_reward_model(
                reward_model,
                reward_optimizer,
                prot_feat_batch,
                reward_metrics["pep_features"].detach(),
                (ref_losses - current_losses).detach(),
                u_score.detach(),
                reward_labels,
                args,
            )
            reward_model.eval()
            with torch.no_grad():
                norm_elbo = (ref_losses - current_losses)
                norm_elbo = (norm_elbo - norm_elbo.mean()) / (norm_elbo.std(unbiased=False) + 1e-6)
                implicit_logits = reward_model(
                    prot_feat_batch,
                    reward_metrics["pep_features"].detach(),
                    norm_elbo,
                    u_score.repeat(len(gen_peptides), 1).detach(),
                )
                implicit_rewards = torch.tanh(implicit_logits)
                total_rewards = args.outcome_reward_weight * outcome_rewards + args.implicit_reward_weight * implicit_rewards
                advantages = compute_group_relative_advantages(total_rewards)
            
            # [C. PPO update]
            model.train()
            uncertainty_head.train()
            prot_z_train = prot_encoder([target_prot]).to(device)
            all_input_ids = myTokenizer.batch_encode(gen_peptides).to(device)[:, :50].contiguous()

            last_loss = None
            last_kl = None
            last_clip_frac = None
            for _ in range(args.ppo_updates):
                optimizer.zero_grad()

                u_score_train = uncertainty_head(prot_z_train.detach())
                condition_train = build_condition_from_protein(prot_z_train, all_input_ids.shape[0], 50, device)
                train_kwargs = {
                    "self_condition": condition_train,
                    "u_score": u_score_train.repeat(all_input_ids.shape[0], 1).to(device),
                    "input_ids": all_input_ids
                }

                t, _ = schedule_sampler.sample(all_input_ids.shape[0], device)
                dummy_x_start = torch.zeros(all_input_ids.shape[0], 50, args.n_embd, device=device)
                losses = diffusion.training_losses(model, x_start=dummy_x_start, t=t, model_kwargs=train_kwargs.copy())
                ppo_loss, approx_kl, clip_frac = compute_ppo_loss(
                    losses["loss"], old_losses, advantages.detach(), args.ppo_clip_eps
                )
                u_reg = args.u_score_reg_weight * (u_score_train.mean() - args.u_score_target) ** 2
                loss = ppo_loss + u_reg

                loss.backward()
                optimizer.step()

                last_loss = loss
                last_kl = approx_kl
                last_clip_frac = clip_frac
            append_progress(
                progress_log_path,
                f"epoch {epoch+1}: target {target_id} done "
                f"R={total_rewards.mean().item():.3f} "
                f"margin={reward_metrics['spec_margin'].mean().item():.3f} "
                f"bind={reward_metrics['cos_sim'].mean().item():.3f} "
                f"u={u_score.item():.3f} loss={last_loss.item():.4f}"
            )
            
            pbar.set_postfix({
                "u": f"{u_score.item():.3f}", 
                "R": f"{total_rewards.mean().item():.3f}",
                "Out": f"{outcome_rewards.mean().item():.3f}",
                "Imp": f"{implicit_rewards.mean().item():.3f}",
                "Adv": f"{advantages.max().item():.3f}",
                "KL": f"{last_kl.item():.3f}",
                "Clip": f"{last_clip_frac.item():.2f}",
                "RM": f"{reward_model_loss.item():.3f}",
                "Bind": f"{reward_metrics['cos_sim'].mean().item():.3f}",
                "Margin": f"{reward_metrics['spec_margin'].mean().item():.3f}",
                "Rob": f"{reward_metrics['robust_reward'].mean().item():.3f}",
                "Charge": f"{reward_metrics['charge_reward'].mean().item():.3f}",
                "Hot": f"{reward_metrics['hotspot_reward'].mean().item():.3f}",
                "Pen": f"{reward_metrics['affinity_penalty'].mean().item():.3f}",
                "Loss": f"{last_loss.item():.4f}"
            })

    # Save final checkpoints.
    print("\n>> Saving PPO fine-tuned checkpoints...")
    torch.save(model.state_dict(), args.save_model_path)
    torch.save(uncertainty_head.state_dict(), args.save_uncertainty_head_path)
    if args.save_reward_model_path:
        torch.save(reward_model.state_dict(), args.save_reward_model_path)
    append_progress(progress_log_path, "training finished and checkpoints saved")
    print("PPO fine-tuning completed.")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=8,
        use_ddim=False,
        min_length=0,    
        max_length=50,   
        use_fp16=False,
        seq_len=[50],
        vocab_size=30,
        class_cond=True,
        n_embd=512,
        stage3_model_path="./checkpoints/UR_PepCCD_MoE_pepflow/ema_0.9999_1000000.pt",
        ref_model_path="./checkpoints/UR_PepCCD_MoE_pepflow/ema_0.9999_1000000.pt",
        stage3_uncertainty_head_path="./checkpoints/UR_PepCCD_MoE_pepflow/uncertainty_head.pt",
        save_model_path="./checkpoints/UR_PepCCD_MoE/rl_elbo_diffusion.pt",
        save_uncertainty_head_path="./checkpoints/UR_PepCCD_MoE/rl_elbo_uncertainty_head.pt",
        save_reward_model_path="./checkpoints/UR_PepCCD_MoE/rl_elbo_reward_model.pt",
        progress_log_path="",
        charge_weight=0.65,
        hotspot_weight=0.35,
        bind_floor=0.10,
        bind_penalty_weight=0.40,
        margin_floor=DEFAULT_MARGIN_FLOOR,
        margin_reward_weight=DEFAULT_MARGIN_REWARD_WEIGHT,
        bind_reward_weight=DEFAULT_BIND_REWARD_WEIGHT,
        margin_penalty_weight=DEFAULT_MARGIN_PENALTY_WEIGHT,
        u_min=0.10,
        u_max=0.45,
        outcome_reward_weight=0.75,
        implicit_reward_weight=0.25,
        reward_model_lr=DEFAULT_REWARD_MODEL_LR,
        reward_updates=DEFAULT_REWARD_UPDATES,
        elbo_aux_weight=DEFAULT_ELBO_AUX_WEIGHT,
        epochs=10,
        robust_lr=1e-5,
        uncertainty_head_lr=DEFAULT_UNCERTAINTY_HEAD_LR,
        u_score_target=DEFAULT_U_SCORE_TARGET,
        u_score_reg_weight=DEFAULT_U_SCORE_REG_WEIGHT,
        adv_temperature=DEFAULT_ADV_TEMPERATURE,
        ppo_clip_eps=DEFAULT_PPO_CLIP_EPS,
        ppo_updates=DEFAULT_PPO_UPDATES,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == '__main__':
    main()
