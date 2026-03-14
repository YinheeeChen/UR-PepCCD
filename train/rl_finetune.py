import sys
import os
sys.path.append(".") # 确保能读到根目录

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm  # 🌟 引入进度条神器

from model.resample import create_named_schedule_sampler
from utils.script_utils import model_and_diffusion_defaults, create_model_and_diffusion
from train.align_train import ESMModel
from samples.sampling import PretainedTokenizer
import argparse
from utils.script_utils import add_dict_to_argparser

# --- 1. 定义不确定性探头 ---
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
    # ratio 已经是 0~1，再做一个轻微放大，突出热点残基的贡献
    reward = min(1.0, hotspot_ratio * 2.5)
    return float(reward), float(hotspot_ratio)


# --- 2. 核心 Reward 计算 ---
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
    r_robust = 0.7 * r_charge + 0.3 * r_hotspot
    u_val = u_score.view(-1)
    # UR-PepCCD 核心公式：Reward = (1 - u)*亲和力 + u*鲁棒性
    total_reward = (1.0 - u_val) * cos_sim + u_val * r_robust
    aux_metrics = {
        "cos_sim": cos_sim.detach(),
        "charge_reward": r_charge.detach(),
        "hotspot_reward": r_hotspot.detach(),
        "net_charge": charges,
        "hotspot_ratio": hotspot_ratios,
    }
    return total_reward, aux_metrics

def main():
    device = os.environ.get("PEPCCD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 UR-PepCCD 强化学习微调 (RLHF)！使用设备: {device}")

    # ===== 加载模型 =====
    from utils.script_utils import args_to_dict
    
    args = create_argparser().parse_args()
    
    # 这里用原作者自带的函数，精准过滤掉 diffusion 不需要的多余参数！
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # ⚠️ 请确保这个文件名和你刚才跑出来的 EMA 权重名一致！
    model.load_state_dict(torch.load(args.stage3_model_path, map_location=device), strict=False)
    model.to(device)
    
    uncertainty_head = UncertaintyHead(input_dim=640).to(device)
    if os.path.exists(args.stage3_uncertainty_head_path):
        uncertainty_head.load_state_dict(
            torch.load(args.stage3_uncertainty_head_path, map_location=device, weights_only=True)
        )
        print(f">> 已加载 Stage 3 uncertainty head: {args.stage3_uncertainty_head_path}")
    else:
        print(f">> 未找到 Stage 3 uncertainty head，将从随机初始化开始训练: {args.stage3_uncertainty_head_path}")
    
    prot_encoder = ESMModel("./checkpoints/ESM", device)
    prot_encoder.load_state_dict(torch.load("./checkpoints/Align/best_prot.pth", map_location=device))
    
    pep_encoder = ESMModel("./checkpoints/ESM", device)
    pep_encoder.load_state_dict(torch.load("./checkpoints/Align/best_pep.pth", map_location=device))
    
    myTokenizer = PretainedTokenizer()
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    # ===== 精准锁血 (只更新我们需要的部分) =====
    for param in model.parameters(): param.requires_grad = False
    for param in prot_encoder.parameters(): param.requires_grad = False
    for param in pep_encoder.parameters(): param.requires_grad = False
        
    trainable_params = []
    for i in range(len(model.transformer.h)):
        if hasattr(model.transformer.h[i], 'mlp_robust'):
            for param in model.transformer.h[i].mlp_robust.parameters():
                param.requires_grad = True
                trainable_params.append(param)
                
    for param in uncertainty_head.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

    # ===== 读取数据集 (靶点题库) =====
    csv_path = "./dataset/PepFlow/pepflow_train.csv" # 可以换成 train.csv 只要里面有 protein_sequence 这一列就行
    print(f">> 正在读取靶点题库: {csv_path}")
    df = pd.read_csv(csv_path)
    protein_list = df['prot_seq'].tolist()
    protein_list = protein_list[:100]

    epochs = 10
    batch_size = 8 # 每次给一个靶点生成 8 条候选答案
    top_k = 2      # 挑最好的 2 条更新
    
    def model_fn_sample(inputs_embeds, timesteps, self_condition=None, u_score=None):
        return model(inputs_embeds=inputs_embeds, timesteps=timesteps, self_condition=self_condition, u_score=u_score)

    print("\n🔥 开始激烈的 RL 训练...")
    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch+1}/{epochs} ==========")
        
        # 🌟 使用 tqdm 包装 protein_list，创建动态进度条
        pbar = tqdm(protein_list, desc=f"Training", unit="靶点")
        
        for step, target_prot in enumerate(pbar):
            if len(target_prot) > 1024: 
                continue # 跳过超长序列防止 ESM 爆显存

            model.eval()
            uncertainty_head.eval()
            
            # [A. 采样 (Sample)]
            with torch.no_grad():
                test_tokens = myTokenizer.batch_encode([target_prot]).to(device)
                prot_z = prot_encoder([target_prot]).to(device)
                u_score = uncertainty_head(prot_z)
                
                u_score_batch = u_score.repeat(batch_size, 1).to(device)
                prot_z_batch = prot_z.repeat(batch_size, 1).to(device)
                ref_prot = prot_z_batch / prot_z_batch.norm(dim=-1, keepdim=True)
                condition = ref_prot.unsqueeze(1).repeat(1, 50, 1).to(device)
                
                model_kwargs = {"self_condition": condition, "u_score": u_score_batch}
                
                # progress=False 取消每次生成的长条刷屏
                samples = diffusion.ddim_sample_loop(
                    model_fn_sample, shape=(batch_size, 50, args.n_embd),
                    model_kwargs=model_kwargs, device=device, clip_denoised=True, progress=False
                )
                sample_emb = samples[-1]
                
                logits = model.get_logits(sample_emb)
                _, indices = torch.topk(logits, k=1, dim=-1)
                indices = indices.squeeze(-1)
                
                gen_peptides = [myTokenizer.batch_decode([seq.cpu().numpy()]) for seq in indices]
            
            # [B. 阅卷与过滤 (Reward & Filter)]
            rewards, reward_metrics = compute_reward(gen_peptides, target_prot, pep_encoder, prot_encoder, u_score, device)
            
            topk_indices = torch.topk(rewards, k=top_k).indices
            best_peptides_text = [gen_peptides[idx] for idx in topk_indices]
            
            # [C. 更新进化 (Update)]
            model.train()
            uncertainty_head.train()
            optimizer.zero_grad()
            
            prot_z_train = prot_encoder([target_prot]).to(device)
            u_score_train = uncertainty_head(prot_z_train.detach())
            
            best_input_ids = myTokenizer.batch_encode(best_peptides_text).to(device)[:, :50].contiguous()
            
            condition_train = (prot_z_train / prot_z_train.norm(dim=-1, keepdim=True)).unsqueeze(1).repeat(top_k, 50, 1).to(device)
            train_kwargs = {
                "self_condition": condition_train,
                "u_score": u_score_train.repeat(top_k, 1).to(device),
                "input_ids": best_input_ids # 核心：喂入优胜者的 input_ids
            }
            
            t, _ = schedule_sampler.sample(top_k, device)
            dummy_x_start = torch.zeros(top_k, 50, args.n_embd, device=device) 
            
            # 🌟 加上 .copy() 防坑机制，防止底层代码弹出键！
            losses = diffusion.training_losses(model, x_start=dummy_x_start, t=t, model_kwargs=train_kwargs.copy())
            loss = losses["loss"].mean()
            
            loss.backward()
            optimizer.step()
            
            # 🌟 动态更新进度条右侧的面板数据
            pbar.set_postfix({
                "u": f"{u_score.item():.3f}", 
                "R": f"{rewards.mean().item():.3f}", 
                "Bind": f"{reward_metrics['cos_sim'].mean().item():.3f}",
                "Charge": f"{reward_metrics['charge_reward'].mean().item():.3f}",
                "Hot": f"{reward_metrics['hotspot_reward'].mean().item():.3f}",
                "Loss": f"{loss.item():.4f}"
            })

    # 保存最终神装
    print("\n>> 正在保存 RL 进化后的权重...")
    torch.save(model.state_dict(), "./checkpoints/UR_PepCCD_MoE/rl_finetuned_diffusion.pt")
    torch.save(uncertainty_head.state_dict(), "./checkpoints/UR_PepCCD_MoE/rl_uncertainty_head.pt")
    print("🎉 恭喜！强化学习微调全部完成，你的 UR-PepCCD 已经满级！")

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

if __name__ == '__main__':
    main()
