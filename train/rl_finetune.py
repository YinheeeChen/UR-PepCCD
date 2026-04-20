import sys
import os
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
U_SCORE_TARGET = 0.25
U_SCORE_REG_WEIGHT = 0.05
UNCERTAINTY_HEAD_LR = 1e-6
ROBUST_CHARGE_WEIGHT = 0.65
ROBUST_HOTSPOT_WEIGHT = 0.35
BIND_FLOOR = 0.10
BIND_PENALTY_WEIGHT = 0.40
U_MIN = 0.10
U_MAX = 0.45
OUTCOME_REWARD_WEIGHT = 0.75
IMPLICIT_REWARD_WEIGHT = 0.25
ADV_TEMPERATURE = 0.35


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


# --- 2. Reward computation ---
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

    # Keep u in a moderate interval to avoid robust-only hard switching.
    u_val = u_score.view(-1)
    u_mix = U_MIN + (U_MAX - U_MIN) * u_val

    # Suppress robust bonus when affinity is low.
    affinity_gate = torch.sigmoid(12.0 * (cos_sim - BIND_FLOOR))
    effective_robust = affinity_gate * r_robust + (1.0 - affinity_gate) * torch.clamp(cos_sim, min=0.0)
    affinity_penalty = F.relu(BIND_FLOOR - cos_sim)

    # Keep UR core fusion while adding affinity guardrails.
    total_reward = (1.0 - u_mix) * cos_sim + u_mix * effective_robust - BIND_PENALTY_WEIGHT * affinity_penalty
    aux_metrics = {
        "cos_sim": cos_sim.detach(),
        "charge_reward": r_charge.detach(),
        "hotspot_reward": r_hotspot.detach(),
        "robust_reward": r_robust.detach(),
        "affinity_gate": affinity_gate.detach(),
        "affinity_penalty": affinity_penalty.detach(),
        "u_mix": u_mix.detach(),
        "net_charge": charges,
        "hotspot_ratio": hotspot_ratios,
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

def main():
    device = os.environ.get("PEPCCD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  UR-PepCCD  (RLHF): {device}")

    # =====  =====
    from utils.script_utils import args_to_dict
    
    args = create_argparser().parse_args()
    
    #  diffusion 
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    #   EMA 
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
        print(f">>  Stage 3 uncertainty head: {args.stage3_uncertainty_head_path}")
    else:
        print(f">>  Stage 3 uncertainty head: {args.stage3_uncertainty_head_path}")
    
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
    
    myTokenizer = PretainedTokenizer()
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    # =====  () =====
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

    optimizer = torch.optim.AdamW(
        [
            {"params": robust_params, "lr": 1e-5},
            {"params": head_params, "lr": UNCERTAINTY_HEAD_LR},
        ]
    )

    # =====  () =====
    csv_path = "./dataset/PepFlow/pepflow_train.csv" #  train.csv  protein_sequence 
    print(f">> : {csv_path}")
    df = pd.read_csv(csv_path)
    protein_list = df['prot_seq'].tolist()
    protein_list = protein_list[:100]

    epochs = 10
    batch_size = 8 #  8 
    def model_fn_sample(inputs_embeds, timesteps, self_condition=None, u_score=None):
        return model(inputs_embeds=inputs_embeds, timesteps=timesteps, self_condition=self_condition, u_score=u_score)

    print("\n  RL ...")
    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch+1}/{epochs} ==========")
        
        #   tqdm  protein_list
        pbar = tqdm(protein_list, desc=f"Training", unit="")
        
        for step, target_prot in enumerate(pbar):
            if len(target_prot) > 1024: 
                continue #  ESM 

            model.eval()
            uncertainty_head.eval()
            
            # [A.  (Sample)]
            with torch.no_grad():
                test_tokens = myTokenizer.batch_encode([target_prot]).to(device)
                prot_z = prot_encoder([target_prot]).to(device)
                u_score = uncertainty_head(prot_z)
                
                u_score_batch = u_score.repeat(batch_size, 1).to(device)
                condition = build_condition_from_protein(prot_z, batch_size, 50, device)
                
                model_kwargs = {"self_condition": condition, "u_score": u_score_batch}
                
                # progress=False 
                samples = diffusion.ddim_sample_loop(
                    model_fn_sample, shape=(batch_size, 50, args.n_embd),
                    model_kwargs=model_kwargs, device=device, clip_denoised=True, progress=False
                )
                sample_emb = samples[-1]
                
                logits = model.get_logits(sample_emb)
                _, indices = torch.topk(logits, k=1, dim=-1)
                indices = indices.squeeze(-1)
                
                gen_peptides = [myTokenizer.batch_decode([seq.cpu().numpy()]) for seq in indices]
            
            # [B.  (Reward & Filter)]
            outcome_rewards, reward_metrics = compute_reward(gen_peptides, target_prot, pep_encoder, prot_encoder, u_score, device)
            with torch.no_grad():
                current_losses = compute_diffusion_candidate_losses(
                    model, diffusion, myTokenizer, gen_peptides, prot_z, u_score,
                    schedule_sampler, device, 50, args.n_embd
                )
                ref_losses = compute_diffusion_candidate_losses(
                    ref_model, diffusion, myTokenizer, gen_peptides, prot_z, u_score,
                    schedule_sampler, device, 50, args.n_embd
                )
                implicit_rewards = ref_losses - current_losses
                total_rewards = OUTCOME_REWARD_WEIGHT * outcome_rewards + IMPLICIT_REWARD_WEIGHT * implicit_rewards
                advantages = compute_group_relative_advantages(total_rewards)
                sample_weights = torch.softmax(advantages / ADV_TEMPERATURE, dim=0).detach()
            
            # [C.  (Update)]
            model.train()
            uncertainty_head.train()
            optimizer.zero_grad()
            
            prot_z_train = prot_encoder([target_prot]).to(device)
            u_score_train = uncertainty_head(prot_z_train.detach())
            
            all_input_ids = myTokenizer.batch_encode(gen_peptides).to(device)[:, :50].contiguous()
            
            condition_train = build_condition_from_protein(prot_z_train, all_input_ids.shape[0], 50, device)
            train_kwargs = {
                "self_condition": condition_train,
                "u_score": u_score_train.repeat(all_input_ids.shape[0], 1).to(device),
                "input_ids": all_input_ids
            }
            
            t, _ = schedule_sampler.sample(all_input_ids.shape[0], device)
            dummy_x_start = torch.zeros(all_input_ids.shape[0], 50, args.n_embd, device=device) 
            
            losses = diffusion.training_losses(model, x_start=dummy_x_start, t=t, model_kwargs=train_kwargs.copy())
            u_reg = U_SCORE_REG_WEIGHT * (u_score_train.mean() - U_SCORE_TARGET) ** 2
            loss = (losses["loss"] * sample_weights).sum() + u_reg
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                "u": f"{u_score.item():.3f}", 
                "R": f"{total_rewards.mean().item():.3f}",
                "Out": f"{outcome_rewards.mean().item():.3f}",
                "Imp": f"{implicit_rewards.mean().item():.3f}",
                "Adv": f"{advantages.max().item():.3f}",
                "Bind": f"{reward_metrics['cos_sim'].mean().item():.3f}",
                "Rob": f"{reward_metrics['robust_reward'].mean().item():.3f}",
                "Charge": f"{reward_metrics['charge_reward'].mean().item():.3f}",
                "Hot": f"{reward_metrics['hotspot_reward'].mean().item():.3f}",
                "Pen": f"{reward_metrics['affinity_penalty'].mean().item():.3f}",
                "Loss": f"{loss.item():.4f}"
            })

    # 
    print("\n>>  RL ...")
    torch.save(model.state_dict(), "./checkpoints/UR_PepCCD_MoE/rl_elbo_diffusion.pt")
    torch.save(uncertainty_head.state_dict(), "./checkpoints/UR_PepCCD_MoE/rl_elbo_uncertainty_head.pt")
    print("  UR-PepCCD ")

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
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == '__main__':
    main()
