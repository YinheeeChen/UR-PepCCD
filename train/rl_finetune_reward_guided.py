import sys
import os
sys.path.append(".")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

from model.resample import create_named_schedule_sampler
from train.align_train import ESMModel
from utils.script_utils import add_dict_to_argparser, args_to_dict, create_model_and_diffusion, model_and_diffusion_defaults


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
U_SCORE_TARGET = 0.25
U_SCORE_REG_WEIGHT = 0.05
UNCERTAINTY_HEAD_LR = 1e-6
ROBUST_CHARGE_WEIGHT = 0.65
ROBUST_HOTSPOT_WEIGHT = 0.35
BIND_FLOOR = 0.10
BIND_PENALTY_WEIGHT = 0.40
U_MIN = 0.10
U_MAX = 0.45


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
    ratio = sum(aa in HOTSPOT_RESIDUES for aa in seq) / len(seq)
    reward = min(1.0, ratio * 2.5)
    return float(reward), float(ratio)


def compute_reward(pep_seqs, prot_seq, pep_encoder, prot_encoder, u_score, device):
    clean_peptides = [clean_peptide_sequence(seq) for seq in pep_seqs]
    with torch.no_grad():
        prot_features = prot_encoder([prot_seq])
        pep_features = pep_encoder(clean_peptides)
        cos_sim = F.cosine_similarity(pep_features, prot_features, dim=-1)

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
    r_robust = ROBUST_CHARGE_WEIGHT * r_charge + ROBUST_HOTSPOT_WEIGHT * r_hotspot
    u_val = u_score.view(-1)
    u_mix = U_MIN + (U_MAX - U_MIN) * u_val
    affinity_gate = torch.sigmoid(12.0 * (cos_sim - BIND_FLOOR))
    effective_robust = affinity_gate * r_robust + (1.0 - affinity_gate) * torch.clamp(cos_sim, min=0.0)
    affinity_penalty = F.relu(BIND_FLOOR - cos_sim)
    total_reward = (1.0 - u_mix) * cos_sim + u_mix * effective_robust - BIND_PENALTY_WEIGHT * affinity_penalty
    aux_metrics = {
        "cos_sim": cos_sim.detach(),
        "charge_reward": r_charge.detach(),
        "hotspot_reward": r_hotspot.detach(),
        "robust_reward": r_robust.detach(),
        "affinity_penalty": affinity_penalty.detach(),
        "net_charge": charges,
        "hotspot_ratio": hotspot_ratios,
    }
    return total_reward, aux_metrics


def build_condition_from_protein(prot_z, repeat_size, seq_len, device):
    prot_z_batch = prot_z.repeat(repeat_size, 1).to(device)
    ref_prot = prot_z_batch / prot_z_batch.norm(dim=-1, keepdim=True)
    return ref_prot.unsqueeze(1).repeat(1, seq_len, 1).to(device)


def main():
    device = os.environ.get("PEPCCD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    args = create_argparser().parse_args()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.stage3_model_path, map_location=device), strict=False)
    initialize_robust_mlp_from_spec(model)
    model.to(device)

    uncertainty_head = UncertaintyHead(input_dim=640).to(device)
    if os.path.exists(args.stage3_uncertainty_head_path):
        uncertainty_head.load_state_dict(
            torch.load(args.stage3_uncertainty_head_path, map_location=device, weights_only=True)
        )

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

    my_tokenizer = PretainedTokenizer()
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    for param in model.parameters():
        param.requires_grad = False
    for param in prot_encoder.parameters():
        param.requires_grad = False
    for param in pep_encoder.parameters():
        param.requires_grad = False

    robust_params = []
    for block in model.transformer.h:
        if hasattr(block, "mlp_robust"):
            for param in block.mlp_robust.parameters():
                param.requires_grad = True
                robust_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": robust_params, "lr": 1e-5},
            {"params": uncertainty_head.parameters(), "lr": UNCERTAINTY_HEAD_LR},
        ]
    )

    df = pd.read_csv("./dataset/PepFlow/pepflow_train.csv")
    protein_list = df["prot_seq"].tolist()[:100]

    epochs = 10
    batch_size = 8
    top_k = 2

    def model_fn_sample(inputs_embeds, timesteps, self_condition=None, u_score=None):
        return model(inputs_embeds=inputs_embeds, timesteps=timesteps, self_condition=self_condition, u_score=u_score)

    for epoch in range(epochs):
        pbar = tqdm(protein_list, desc=f"Training", unit="")
        for target_prot in pbar:
            if len(target_prot) > 1024:
                continue

            model.eval()
            uncertainty_head.eval()
            with torch.no_grad():
                prot_z = prot_encoder([target_prot]).to(device)
                u_score = uncertainty_head(prot_z)
                condition = build_condition_from_protein(prot_z, batch_size, 50, device)
                samples = diffusion.ddim_sample_loop(
                    model_fn_sample,
                    shape=(batch_size, 50, args.n_embd),
                    model_kwargs={"self_condition": condition, "u_score": u_score.repeat(batch_size, 1)},
                    device=device,
                    clip_denoised=True,
                    progress=False,
                )
                logits = model.get_logits(samples[-1])
                indices = torch.topk(logits, k=1, dim=-1).indices.squeeze(-1)
                gen_peptides = [my_tokenizer.batch_decode([seq.cpu().numpy()]) for seq in indices]

            rewards, reward_metrics = compute_reward(gen_peptides, target_prot, pep_encoder, prot_encoder, u_score, device)
            topk_indices = torch.topk(rewards, k=top_k).indices
            best_peptides = [gen_peptides[idx] for idx in topk_indices]

            model.train()
            uncertainty_head.train()
            optimizer.zero_grad()
            prot_z_train = prot_encoder([target_prot]).to(device)
            u_score_train = uncertainty_head(prot_z_train.detach())
            best_input_ids = my_tokenizer.batch_encode(best_peptides).to(device)[:, :50].contiguous()
            condition_train = build_condition_from_protein(prot_z_train, top_k, 50, device)
            train_kwargs = {
                "self_condition": condition_train,
                "u_score": u_score_train.repeat(top_k, 1).to(device),
                "input_ids": best_input_ids,
            }
            t, _ = schedule_sampler.sample(top_k, device)
            dummy_x_start = torch.zeros(top_k, 50, args.n_embd, device=device)
            losses = diffusion.training_losses(model, x_start=dummy_x_start, t=t, model_kwargs=train_kwargs.copy())
            u_reg = U_SCORE_REG_WEIGHT * (u_score_train.mean() - U_SCORE_TARGET) ** 2
            loss = losses["loss"].mean() + u_reg
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "u": f"{u_score.item():.3f}",
                "R": f"{rewards.mean().item():.3f}",
                "Bind": f"{reward_metrics['cos_sim'].mean().item():.3f}",
                "Rob": f"{reward_metrics['robust_reward'].mean().item():.3f}",
                "Charge": f"{reward_metrics['charge_reward'].mean().item():.3f}",
                "Hot": f"{reward_metrics['hotspot_reward'].mean().item():.3f}",
                "Pen": f"{reward_metrics['affinity_penalty'].mean().item():.3f}",
                "Loss": f"{loss.item():.4f}",
            })

    torch.save(model.state_dict(), "./checkpoints/UR_PepCCD_MoE/rl_reward_guided_diffusion.pt")
    torch.save(uncertainty_head.state_dict(), "./checkpoints/UR_PepCCD_MoE/rl_reward_guided_uncertainty_head.pt")


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
        stage3_uncertainty_head_path="./checkpoints/UR_PepCCD_MoE_pepflow/uncertainty_head.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
