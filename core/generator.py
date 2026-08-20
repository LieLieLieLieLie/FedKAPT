"""
core/generator.py — Module 3: Transport-Conditioned CVAE Generation.
Loss = Reconstruction (Gaussian NLL) + β·KL(warm-up) + λ·Sinkhorn OT regularization
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from config import Config
from models.networks import CVAE
from evaluation.utils import AverageMeter


def _sinkhorn_loss(x_gen, cond, reg):
    if x_gen.size(0) <= 1:
        return x_gen.new_tensor(0.0)
    M = torch.cdist(x_gen, cond, p=2).pow(2)
    M = M / (M.detach().max().clamp_min(1e-8))
    log_K = -M / max(reg, 1e-6)
    log_u = torch.zeros(x_gen.size(0), device=x_gen.device, dtype=x_gen.dtype)
    log_v = torch.zeros(cond.size(0),  device=x_gen.device, dtype=x_gen.dtype)
    log_a = torch.full_like(log_u, -np.log(x_gen.size(0)))
    log_b = torch.full_like(log_v, -np.log(cond.size(0)))
    for _ in range(30):
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_K.T + log_u[None, :], dim=1)
    T = torch.exp(log_u[:, None] + log_K + log_v[None, :])
    return (T * M).sum()


def _cvae_loss(x_mu, x_lv, x_tgt, z_mu, z_lv, z_cycle, cond,
               beta, ot_lam, ot_reg, cycle_lam):
    var   = x_lv.exp().clamp(1e-4)
    recon = (0.5 * (x_lv + (x_tgt - x_mu).pow(2) / var)).mean()
    kl    = (-0.5 * (1 + z_lv - z_mu.pow(2) - z_lv.exp())).mean()
    # OT regularisation aligns generated release-space features with the
    # transmitted prototype conditions, encouraging condition-aligned outputs.
    ot_l  = _sinkhorn_loss(x_mu, cond, ot_reg)
    cycle = (z_cycle - z_mu.detach()).pow(2).mean()
    total = recon + beta * kl + ot_lam * ot_l + cycle_lam * cycle
    return {"total": total, "recon": recon, "kl": kl, "ot": ot_l,
            "cycle": cycle}


class CVAEGenerator:
    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.device = cfg.device
        self.cvae   = None

    def train(self, t_features, conditions, logger=None):
        cfg_c    = self.cfg.cvae
        x_dim    = t_features.shape[1]
        cond_dim = conditions.shape[1]
        out_dim  = cond_dim

        # The decoder produces features in the r-dimensional public projection /
        # source-prototype release space.  Its target is the
        # transported source condition; target-specific information is carried
        # through z and protected by the latent-cycle term below.
        self.cvae = CVAE(x_dim, cond_dim, out_dim,
                         cfg_c.hidden_dims, cfg_c.latent_dim).to(self.device)
        opt   = torch.optim.Adam(self.cvae.parameters(),
                                 lr=cfg_c.lr, weight_decay=cfg_c.weight_decay)
        sched = StepLR(opt, cfg_c.lr_scheduler_step, cfg_c.lr_scheduler_gamma)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(t_features).float(),
                          torch.from_numpy(conditions).float()),
            batch_size=cfg_c.batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator().manual_seed(self.cfg.seed))
        meters = {k: AverageMeter(k) for k in
                  ("total", "recon", "kl", "ot", "cycle")}

        warmup = getattr(cfg_c, "kl_warmup_epochs", 0)

        self.cvae.train()
        for epoch in range(1, cfg_c.epochs + 1):
            # KL warm-up: ramp beta from 0→cfg_c.beta over the first
            # kl_warmup_epochs to prevent posterior collapse early in training.
            if warmup > 0 and epoch <= warmup:
                beta_eff = cfg_c.beta * (epoch / warmup)
            else:
                beta_eff = cfg_c.beta

            for m in meters.values(): m.reset()
            for xb, cb in loader:
                xb, cb = xb.to(self.device), cb.to(self.device)
                x_mu, x_lv, z_mu, z_lv, z_cycle = self.cvae(xb, cb)
                # No element-wise cross-space interpolation is used. The
                # release-space decoder is anchored to the transported condition,
                # while latent cycle consistency preserves sample-specific
                # information encoded from the target space.
                x_tgt = cb
                losses = _cvae_loss(
                    x_mu, x_lv, x_tgt, z_mu, z_lv, z_cycle, cb,
                    beta_eff, cfg_c.ot_lambda, cfg_c.ot_reg,
                    getattr(cfg_c, "latent_cycle_weight", 0.5))
                opt.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(self.cvae.parameters(), cfg_c.grad_clip)
                opt.step()
                n = xb.size(0)
                for k, v in losses.items(): meters[k].update(v.item(), n)
            sched.step()
            if logger and epoch % 20 == 0:
                logger.info(f"  [CVAE] {epoch:>3d}/{cfg_c.epochs}  "
                            f"total={meters['total'].avg:.4f}  "
                            f"recon={meters['recon'].avg:.4f}  "
                            f"kl={meters['kl'].avg:.4f}  "
                            f"ot={meters['ot'].avg:.4f}  "
                            f"cycle={meters['cycle'].avg:.4f}  "
                            f"β={beta_eff:.3f}")

    @torch.no_grad()
    def generate(self, t_features, conditions):
        self.cvae.eval()
        all_gen, all_var = [], []
        loader = DataLoader(
            TensorDataset(torch.from_numpy(t_features).float(),
                          torch.from_numpy(conditions).float()),
            batch_size=256, shuffle=False)
        for xb, cb in loader:
            xb, cb = xb.to(self.device), cb.to(self.device)
            # Use posterior mean z_mu (not a prior sample) for deterministic
            # generation at inference time.
            z_mu, _z_lv = self.cvae.encoder(xb)
            x_gen, x_lv = self.cvae.decoder(z_mu, cb)
            recon_var = x_lv.exp().mean(dim=-1)
            all_gen.append(x_gen.cpu().numpy())
            all_var.append(recon_var.cpu().numpy())
        return (np.concatenate(all_gen).astype(np.float32),
                np.concatenate(all_var).astype(np.float32))

    @torch.no_grad()
    def latent_cycle_mse(self, t_features, conditions):
        """Measure posterior-latent recovery on deterministic generated means."""
        self.cvae.eval()
        total, count = 0.0, 0
        loader = DataLoader(
            TensorDataset(torch.from_numpy(t_features).float(),
                          torch.from_numpy(conditions).float()),
            batch_size=256, shuffle=False)
        for xb, cb in loader:
            xb, cb = xb.to(self.device), cb.to(self.device)
            z_mu, _ = self.cvae.encoder(xb)
            x_mu, _ = self.cvae.decoder(z_mu, cb)
            z_cycle = self.cvae.cycle_encoder(x_mu)
            total += float((z_cycle - z_mu).pow(2).sum().item())
            count += int(z_mu.numel())
        return total / max(count, 1)
