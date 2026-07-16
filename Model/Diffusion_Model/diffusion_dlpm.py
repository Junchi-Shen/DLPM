import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .Utils import default, identity, normalize_to_neg_one_to_one, exists
from .DLPM.generative_levy_process import GenerativeLevyProcess
from .DLPM.dlpm_core import ModelMeanType, ModelVarType


class DLPMDiffusion1D(nn.Module):
    """Denoising Lévy Probabilistic Model (DLPM) for 1D sequences.

    Follows "Heavy-Tailed Diffusion with Denoising Lévy Probabilistic Models"
    (Shariatian, Simsekli, Durmus; arXiv:2407.18609):

      forward:  x_t = bargamma_t * x_0 + barsigma_t * eps,   eps ~ SαS (isotropic)
      training: standard DLPM loss. Using the variance-mixing identity
                eps = sqrt(a) * z with a ~ positive (α/2)-stable and z ~ N(0, I),
                the network predicts the chain noise eps_t conditioned on a_t,
                with an Lp objective (p <= α required for finite loss) and
                optional median-of-means Monte Carlo over a_t.
      sampling: stochastic DLPM ancestral sampling (posterior mean/variance
                conditioned on the sampled A chain), or deterministic DLIM
                with skip steps (the DDIM analogue).

    alpha is a fixed model hyperparameter in (1, 2]; alpha = 2 recovers DDPM.
    """

    def __init__(
        self,
        *,
        seq_length,
        timesteps=500,
        sampling_timesteps=None,
        alpha=1.8,
        auto_normalize=False,
        model,
        condition_network: nn.Module = None,
        **kwargs
    ):
        super().__init__()
        self.model = model

        alpha = float(alpha)
        assert 1.0 < alpha <= 2.0, f'alpha must be in (1, 2], got {alpha}'
        self.alpha = alpha

        channels_arg = kwargs.get('channels', None)
        self.channels = channels_arg if channels_arg is not None else self.model.channels
        self.channel_first = kwargs.get('channel_first', True)
        self.seq_length = seq_length
        self.objective = 'pred_noise'  # DLPM predicts the chain noise eps

        self.condition_network = condition_network
        self.has_condition_network = condition_network is not None

        self.generative_process = GenerativeLevyProcess(
            alpha=alpha,
            device=next(model.parameters()).device,
            reverse_steps=timesteps,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED,
            scale=kwargs.get('dlpm_scale', 'scale_preserving'),
        )

        # optional clamping of the auxiliary variable a_t (paper: numerical stability)
        clamp_a = kwargs.get('dlpm_clamp_a', None)
        clamp_eps = kwargs.get('dlpm_clamp_eps', None)
        if clamp_eps is None:
            # U-Net 输出经 tanh*output_scaling 封顶(±10)，网络无法预测超出该幅度的 eps；
            # 采样初噪必须落在可表示范围内，否则极端元素会在反向链中被 1/gamma 放大
            clamp_eps = float(getattr(model, 'output_scaling', 10.0))
        self.generative_process.dlpm.gen_a.setParams(clamp_a=clamp_a)
        self.generative_process.dlpm.gen_eps.setParams(clamp_eps=clamp_eps)

        # ---- standard DLPM loss settings (paper defaults) ----
        # Lp exponent: must satisfy p <= alpha so that E||eps||_p^.. is finite.
        # lploss = 2.0 -> per-sample L2 norm (not squared), 1.0 -> smooth L1.
        self.lploss = float(kwargs.get('dlpm_lploss', 2.0))
        self.monte_carlo_outer = int(kwargs.get('dlpm_monte_carlo_outer', 1))
        self.monte_carlo_inner = int(kwargs.get('dlpm_monte_carlo_inner', 1))
        self.loss_monte_carlo = kwargs.get('dlpm_loss_monte_carlo', 'mean')  # 'mean' | 'median'

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else timesteps
        self.ddim_sampling_eta = kwargs.get('ddim_sampling_eta', 0.0)

        self.auto_normalize = auto_normalize
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = identity

    # ------------------------------------------------------------------ utils

    def _expand_mask(self, mask, ref):
        if mask is None:
            return None
        mask = mask.float()
        while mask.dim() < ref.dim():
            mask = mask.unsqueeze(1)
        return mask.expand_as(ref) if mask.shape != ref.shape else mask

    def _get_condition(self, cond_input):
        if cond_input is None:
            return None
        c = torch.nan_to_num(cond_input, nan=0.0, posinf=10.0, neginf=-10.0)
        if not self.has_condition_network:
            return c
        return self.condition_network(c)

    # ------------------------------------------------------------------ loss

    def _masked_lp(self, pred, target, m_exp, p):
        """Per-sample Lp discrepancy restricted to valid (masked) positions."""
        diff = pred - target
        if p == 1.0:
            el = F.smooth_l1_loss(pred, target, beta=1.0, reduction='none')
        else:
            el = diff.abs().pow(2.0 if p == 2.0 else p)
        if m_exp is not None:
            count = m_exp.sum(dim=list(range(1, el.dim()))).clamp(min=1.0)
            el = (el * m_exp).sum(dim=list(range(1, el.dim()))) / count
        else:
            el = el.mean(dim=list(range(1, el.dim())))
        if p == 2.0:
            el = torch.sqrt(el + 1e-12)   # L2 norm, not squared (finite for alpha > 1)
        elif p != 1.0:
            el = el.pow(1.0 / p)
        return el  # shape (B,)

    def p_losses(self, x_start, t, cond_input=None, mask=None, global_step=0, **kwargs):
        """Standard DLPM training loss (arXiv:2407.18609)."""
        if not torch.isfinite(x_start).all():
            raise FloatingPointError(f'x_start contains non-finite values at step {global_step}')

        dlpm = self.generative_process.dlpm
        p_cond = self._get_condition(cond_input)

        if mask is not None:
            mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)

        mco, mci = self.monte_carlo_outer, self.monte_carlo_inner
        total_mc = mco * mci

        x0 = x_start if total_mc == 1 else x_start.repeat(total_mc, *([1] * (x_start.dim() - 1)))
        t_ext = t if total_mc == 1 else t.repeat(total_mc)

        # a_t ~ positive (alpha/2)-stable, one per sample (isotropic); shared across inner MC
        outer_shape = list(x_start.shape)
        outer_shape[0] *= mco
        a_t = dlpm.get_one_rv_faster_sampling(outer_shape)
        if mci > 1:
            a_t = a_t.repeat(mci, *([1] * (a_t.dim() - 1)))
        z_t = torch.randn_like(x0)

        # x_t = bg*x0 + sqrt(a_t)*bs*z ;  eps_t = (x_t - bg*x0)/bs = sqrt(a_t)*z
        x_t, eps_t = dlpm.get_one_rv_loss_elements(t_ext, x0, a_t, z_t)

        p_cond_ext = None
        if p_cond is not None:
            p_cond_ext = p_cond if total_mc == 1 else p_cond.repeat(total_mc, *([1] * (p_cond.dim() - 1)))

        model_out = self.model(x_t, time=t_ext, cond_input=p_cond_ext)
        if not torch.isfinite(model_out).all():
            print(f'⚠️ [DLPM] non-finite model output at step {global_step}, skipping batch')
            return None

        m_exp = self._expand_mask(mask, x0[:x_start.shape[0]])
        if m_exp is not None and total_mc > 1:
            m_exp = m_exp.repeat(total_mc, *([1] * (m_exp.dim() - 1)))

        losses = self._masked_lp(model_out, eps_t, m_exp, self.lploss)

        if self.loss_monte_carlo == 'median' and mco > 1:
            losses = losses.reshape(mci, mco, x_start.shape[0]).mean(dim=0)
            losses, _ = losses.median(dim=0)
            loss = losses.mean()
        else:
            loss = losses.mean()

        return loss if torch.isfinite(loss) else None

    def forward(self, img, cond_input=None, mask=None, global_step=0, **kwargs):
        img = self.normalize(img)
        # t in [1, T): t=0 has bs=0 (zero noise), nothing to learn there
        t = torch.randint(1, self.num_timesteps, (img.shape[0],), device=img.device).long()
        return self.p_losses(img, t, cond_input=cond_input, mask=mask, global_step=global_step, **kwargs)

    # ------------------------------------------------------------------ sampling

    @torch.no_grad()
    def p_sample_loop(self, shape, return_noise=False, model_forward_kwargs: dict = dict()):
        was_training = self.model.training
        self.model.eval()
        dlpm = self.generative_process.dlpm

        mask = model_forward_kwargs.get('mask')
        if mask is not None:
            mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)

        # fresh Lévy chain for every sampling call (A / Sigmas must match shape)
        dlpm.sample_A(shape, self.num_timesteps)
        dlpm.compute_Sigmas()

        pc = self._get_condition(model_forward_kwargs.get('cond_input'))
        noise0 = dlpm.barsigmas[-1] * dlpm.gen_eps.generate(size=shape)
        img = noise0

        for t in tqdm(reversed(range(1, self.num_timesteps)), desc='DLPM sampling',
                      total=self.num_timesteps - 1):
            m = self._expand_mask(mask, img)
            if m is not None:
                img = img * m + noise0 * (1 - m)
            out = self.generative_process.p_sample(
                self.model, img,
                torch.full((shape[0],), t, device=img.device, dtype=torch.long),
                model_kwargs={'cond_input': pc},
            )
            img = out['sample']

        if was_training:
            self.model.train()
        img = self.unnormalize(img)
        mf = self._expand_mask(mask, img)
        if mf is not None:
            img = img * mf + noise0 * (1 - mf)
        return (img, noise0) if return_noise else img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=False, model_forward_kwargs: dict = dict(),
                    return_noise=False, sampling_timesteps=None):
        """Deterministic DLIM sampling with skip steps (paper's DDIM analogue)."""
        was_training = self.model.training
        self.model.eval()
        steps = default(sampling_timesteps, self.sampling_timesteps)
        dlpm = self.generative_process.dlpm

        noise0 = dlpm.barsigmas[-1] * dlpm.gen_eps.generate(size=shape)

        processed_cond = self._get_condition(model_forward_kwargs.get('cond_input'))
        ddim_model_kwargs = {'cond_input': processed_cond}

        img = self.generative_process.ddim_sample_loop(
            self.model,
            shape=shape,
            noise=noise0,
            clip_denoised=clip_denoised,
            model_kwargs=ddim_model_kwargs,
            eta=self.ddim_sampling_eta,
            sampling_timesteps=steps,
            progress=True,
        )

        if was_training:
            self.model.train()
        img = self.unnormalize(img)
        mask = model_forward_kwargs.get('mask')
        if exists(mask):
            mask = self._expand_mask(mask, img)
            img = img * mask + noise0 * (1 - mask)
        return (img, noise0) if return_noise else img

    @torch.no_grad()
    def sample(self, batch_size=16, cond_input=None, mask=None, return_noise=False, sampling_timesteps=None):
        shape = (batch_size, self.channels, self.seq_length) if self.channel_first \
            else (batch_size, self.seq_length, self.channels)

        steps = default(sampling_timesteps, self.sampling_timesteps)
        if steps < self.num_timesteps:
            print(f'🚀 DLIM accelerated sampling: {steps} steps (of {self.num_timesteps})')
            return self.ddim_sample(
                shape,
                sampling_timesteps=steps,
                model_forward_kwargs={'cond_input': cond_input, 'mask': mask},
                return_noise=return_noise,
            )
        return self.p_sample_loop(
            shape,
            model_forward_kwargs={'cond_input': cond_input, 'mask': mask},
            return_noise=return_noise,
        )
