import copy
import functools
import os
import blobfile as bf

import torch as th
import torch.cuda
from torch.optim import AdamW

from config.backen_config import logger
from model.fp16_utils import MixedPrecisionTrainer
from model.nn import update_ema
from model.resample import LossAwareSampler, UniformSampler
import matplotlib.pyplot as plt
import torch.nn as nn

INITIAL_LOG_LOSS_SCALE = 20.0
U_SCORE_TARGET = 0.25
U_SCORE_REG_WEIGHT = 0.05
UNCERTAINTY_HEAD_LR = 1e-5

class UncertaintyHead(nn.Module):
    def __init__(self, input_dim=640):  # ESM hidden size is usually 320 or 640.
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1] for routing weight u.
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


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        device,
        diffusion,
        pep_encoder,
        prot_encoder,
        train_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        features_encoder=None,
        save_dir=None
    ):
        self.model = model
        self.device = device
        self.diffusion = diffusion
        self.train_data = train_data
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0


        self.sync_cuda = th.cuda.is_available()


        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.ema_params = [
            copy.deepcopy(self.mp_trainer.master_params)
            for _ in range(len(self.ema_rate))
        ]
        
        self.use_ddp = False
        self.ddp_model = self.model
        if features_encoder is not None:
            self.features_encoder = features_encoder
        if pep_encoder is not None:
            self.pep_encoder = pep_encoder
            self.pep_encoder.eval()
        if prot_encoder is not None:
            self.prot_encoder = prot_encoder
            self.prot_encoder.eval()

        self.uncertainty_head = UncertaintyHead(input_dim=640).to(self.device)
        self.opt.add_param_group(
            {
                "params": self.uncertainty_head.parameters(),
                "lr": min(self.lr, UNCERTAINTY_HEAD_LR),
                "weight_decay": self.weight_decay,
            }
        )

       

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            train_batch, train_cond = next(self.train_data)
            self.train_run_step(train_batch, train_cond)
            
            if self.step % self.save_interval == 0:
                self.save()
                
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def train_run_step(self, batch, cond):
        self.train_forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()


    def train_forward_backward(self, batch, cond):

        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].clone().to(self.device)
            micro_cond = {}

            current_prot_features = None

            for k, v in cond.items():
                if k == 'input_ids':
                    micro_cond[k] = v[i: i + self.microbatch].to(self.device)
            
                else:
                    with torch.no_grad():
                        
                        protein_sequences = v[i: i + self.microbatch]
                        decoded_sequences = [
                            self.prot_encoder.tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "")
                            for seq in protein_sequences
                        ]
                        prot_features = self.prot_encoder(decoded_sequences) 
                        current_prot_features = prot_features
                        prot_features_norm = prot_features / prot_features.norm(dim=-1,keepdim=True) 
                        prot_features_norm = prot_features_norm.unsqueeze(1).repeat(1, 256, 1)  
                        micro_cond['self_condition'] = prot_features_norm

            u_reg = 0.0
            if current_prot_features is not None:
                # Detach to avoid backpropagating into the frozen ESM encoder.
                u_score = self.uncertainty_head(current_prot_features.detach())
                micro_cond['u_score'] = u_score
                u_reg = U_SCORE_REG_WEIGHT * (u_score.mean() - U_SCORE_TARGET) ** 2
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean() + u_reg

            print(f"train_loss: {loss}")
            self.mp_trainer.backward(loss)
            
    
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            print(f"saving model {rate}...")
            if not rate:
                filename = f"diffusion_model_1{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_1{(self.step+self.resume_step):06d}.pt"
            save_path = bf.join(self.save_dir, filename)
            with bf.BlobFile(save_path, "wb") as f:
                th.save(state_dict, f)
        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        uncertainty_path = bf.join(self.save_dir, "uncertainty_head.pt")
        with bf.BlobFile(uncertainty_path, "wb") as f:
            th.save(self.uncertainty_head.state_dict(), f)



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
