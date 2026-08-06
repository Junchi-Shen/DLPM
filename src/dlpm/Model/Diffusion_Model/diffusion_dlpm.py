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

      forward:  x_t = bargamma_t * x_0 + barsigma_t * eps, with the deployed
                financial configuration using coordinate-wise SαS scale draws
                (``dlpm_isotropic=False``).
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
            # For financial return coordinates, a separate positive-stable
            # scale per time coordinate prevents one path-level scale draw
            # from masquerading as volatility clustering.
            isotropic=bool(kwargs.get('dlpm_isotropic', False)),
            scale=kwargs.get('dlpm_scale', 'scale_preserving'),
        )
        self.sample_x0_clip = kwargs.get('sample_x0_clip', None)
        self.generative_process.dlpm.sample_x0_clip = self.sample_x0_clip

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
        self.lploss = float(kwargs.get('dlpm_lploss', 1.0))
        if self.lploss > self.alpha + 1e-8:
            raise ValueError(
                f'dlpm_lploss={self.lploss} exceeds alpha={self.alpha}; '
                'choose p <= alpha for a finite-moment DLPM objective.'
            )
        self.monte_carlo_outer = int(kwargs.get('dlpm_monte_carlo_outer', 1))
        self.monte_carlo_inner = int(kwargs.get('dlpm_monte_carlo_inner', 1))
        self.loss_monte_carlo = kwargs.get('dlpm_loss_monte_carlo', 'mean')  # 'mean' | 'median'
        self.use_financial_regularizers = bool(kwargs.get('use_financial_regularizers', False))
        self.financial_loss_weights = kwargs.get('financial_loss_weights', {}) or {}
        self.financial_loss_scale = float(kwargs.get('financial_loss_scale', 1.0))
        self.financial_loss_warmup_steps = int(kwargs.get('financial_loss_warmup_steps', 0))
        self.loss_diagnostics_every = int(kwargs.get('loss_diagnostics_every', 0))
        self.last_loss_components = {}
        self.financial_regularizer_clip = float(kwargs.get('financial_regularizer_clip', 8.0))
        self.financial_return_clip = float(kwargs.get('financial_return_clip', 3.0))
        self.config_volatility_scale = float(kwargs.get('volatility_scale', 0.09))

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else timesteps
        self.ddim_sampling_eta = kwargs.get('ddim_sampling_eta', 0.0)

        self.auto_normalize = auto_normalize
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = identity

    def _financial_regularizer_loss(self, pred_x0, x_start, mask=None):
        if not self.use_financial_regularizers:
            return pred_x0.new_tensor(0.0)

        weights = self.financial_loss_weights
        if not weights:
            return pred_x0.new_tensor(0.0)

        m = self._expand_mask(mask, x_start)
        if m is None:
            m = torch.ones_like(x_start)
        m = m.clone()
        if m.shape[-1] > 0:
            m[..., 0] = 0.0
        count = m.sum(dim=-1, keepdim=True).clamp(min=1.0)

        clip = max(float(self.financial_regularizer_clip), 1.0)
        pred = torch.nan_to_num(pred_x0, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip) * m
        targ = torch.nan_to_num(x_start, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip) * m

        total = pred.new_tensor(0.0)

        def add(name, value):
            nonlocal total
            w = float(weights.get(name, 0.0))
            if w > 0 and torch.isfinite(value).all():
                total = total + w * value

        add('pointwise_x0', F.smooth_l1_loss(pred * m, targ.detach() * m))
        ret_limit = max(float(self.financial_return_clip), 0.5)
        return_excess = F.relu(pred.abs() - ret_limit) * m
        add('return_range', return_excess.pow(2).sum() / m.sum().clamp(min=1.0))

        pred_cum = pred.cumsum(dim=-1)
        targ_cum = targ.cumsum(dim=-1)
        # Position 0 is the deterministic anchor and is excluded from m.
        # Therefore N valid returns occupy positions 1..N, so the final
        # valid cumulative-return position is N rather than N-1.
        last_idx = count.squeeze(-1).long().clamp(min=1, max=pred_cum.shape[-1] - 1)
        gather_idx = last_idx.unsqueeze(-1)
        pred_terminal = pred_cum.gather(-1, gather_idx).squeeze(-1)
        targ_terminal = targ_cum.gather(-1, gather_idx).squeeze(-1)

        add('terminal_drift', F.smooth_l1_loss(pred_terminal, targ_terminal.detach()))
        add('cumulative_return', F.smooth_l1_loss(pred_cum * m, targ_cum.detach() * m))
        if pred_terminal.numel() >= 8:
            term_levels = torch.tensor([0.05, 0.50, 0.85, 0.95], device=pred.device)
            pred_term_q = torch.quantile(pred_terminal.float(), term_levels)
            targ_term_q = torch.quantile(targ_terminal.float(), term_levels)
            add('terminal_quantile', F.smooth_l1_loss(pred_term_q, targ_term_q.detach()))
        pred_cum_flat = pred_cum[m > 0.5]
        targ_cum_flat = targ_cum[m > 0.5]
        if pred_cum_flat.numel() >= 32 and targ_cum_flat.numel() >= 32:
            cum_levels = torch.tensor([0.50, 0.85, 0.95, 0.99], device=pred.device)
            pred_cum_abs_q = torch.quantile(pred_cum_flat.abs().float(), cum_levels)
            targ_cum_abs_q = torch.quantile(targ_cum_flat.abs().float(), cum_levels)
            add('cumulative_abs_quantile', F.smooth_l1_loss(pred_cum_abs_q, targ_cum_abs_q.detach()))

        pred_mean = pred.sum(dim=-1, keepdim=True) / count
        targ_mean = targ.sum(dim=-1, keepdim=True) / count
        pred_var = (((pred - pred_mean) ** 2) * m).sum(dim=-1, keepdim=True) / count
        targ_var = (((targ - targ_mean) ** 2) * m).sum(dim=-1, keepdim=True) / count
        pred_std = torch.sqrt(pred_var + 1e-6)
        targ_std = torch.sqrt(targ_var + 1e-6)
        add('global_vol', F.smooth_l1_loss(pred_std, targ_std.detach()))

        pred_mean_abs = (pred.abs() * m).sum(dim=-1, keepdim=True) / count
        targ_mean_abs = (targ.abs() * m).sum(dim=-1, keepdim=True) / count
        add('mean_abs_return', F.smooth_l1_loss(pred_mean_abs, targ_mean_abs.detach()))

        win = max(8, min(32, pred.shape[-1] // 4))
        stride = max(4, win // 2)
        if pred.shape[-1] >= win:
            pw = pred.unfold(-1, win, stride)
            tw = targ.unfold(-1, win, stride)
            mw = m.unfold(-1, win, stride)
            wc = mw.sum(dim=-1, keepdim=True).clamp(min=1.0)
            pm = (pw * mw).sum(dim=-1, keepdim=True) / wc
            tm = (tw * mw).sum(dim=-1, keepdim=True) / wc
            pv = torch.sqrt((((pw - pm) ** 2) * mw).sum(dim=-1, keepdim=True) / wc + 1e-6)
            tv = torch.sqrt((((tw - tm) ** 2) * mw).sum(dim=-1, keepdim=True) / wc + 1e-6)
            add('vol_clustering', F.smooth_l1_loss(pv, tv.detach()))

        pred_flat = pred[m > 0.5]
        targ_flat = targ[m > 0.5]
        if pred_flat.numel() >= 32 and targ_flat.numel() >= 32:
            pq = torch.quantile(pred_flat.float(), torch.tensor([0.01, 0.99], device=pred.device))
            tq = torch.quantile(targ_flat.float(), torch.tensor([0.01, 0.99], device=targ.device))
            add('tail_quantile', F.smooth_l1_loss(pq, tq.detach()))
            abs_levels = torch.tensor([0.50, 0.75, 0.90, 0.95, 0.99], device=pred.device)
            pred_abs_q = torch.quantile(pred_flat.abs().float(), abs_levels)
            targ_abs_q = torch.quantile(targ_flat.abs().float(), abs_levels)
            add('abs_tail_quantile', F.smooth_l1_loss(pred_abs_q, targ_abs_q.detach()))

            qp = torch.quantile(pred_flat.float(), torch.tensor([0.05, 0.25, 0.75, 0.95], device=pred.device))
            qt = torch.quantile(targ_flat.float(), torch.tensor([0.05, 0.25, 0.75, 0.95], device=targ.device))
            pred_iqr = (qp[2] - qp[1]).clamp(min=1e-6)
            targ_iqr = (qt[2] - qt[1]).clamp(min=1e-6)
            pred_tail_width = qp[3] - qp[0]
            targ_tail_width = qt[3] - qt[0]
            add('tail_iqr_ratio', F.smooth_l1_loss(
                pred_tail_width / pred_iqr,
                (targ_tail_width / targ_iqr).detach()
            ))
            for threshold in (0.5, 1.0, 2.0):
                pred_rate = torch.sigmoid(8.0 * (pred_flat.abs() - threshold)).mean()
                targ_rate = torch.sigmoid(8.0 * (targ_flat.abs() - threshold)).mean()
                add(f'tail_exceedance_{threshold:g}', F.smooth_l1_loss(pred_rate, targ_rate.detach()))

        # Volatility clustering: match lag-1 autocorrelation of absolute
        # returns, rather than only matching rolling volatility levels. This
        # constrains persistence of turbulent periods without imposing a
        # directional drift on the generated path.
        if pred.shape[-1] >= 4:
            ret_pred = pred[..., 1:]
            ret_targ = targ[..., 1:]
            ret_mask = m[..., 1:]

            def pair_corr(first, second, pair_mask):
                pair_count = pair_mask.sum(dim=-1, keepdim=True).clamp(min=2.0)
                mean_first = (first * pair_mask).sum(dim=-1, keepdim=True) / pair_count
                mean_second = (second * pair_mask).sum(dim=-1, keepdim=True) / pair_count
                df = first - mean_first
                ds = second - mean_second
                cov = (df * ds * pair_mask).sum(dim=-1, keepdim=True) / pair_count
                var_first = ((df * df) * pair_mask).sum(dim=-1, keepdim=True) / pair_count
                var_second = ((ds * ds) * pair_mask).sum(dim=-1, keepdim=True) / pair_count
                return cov / torch.sqrt(var_first * var_second + 1e-6)

            def lag_corr(current, lag):
                pair_mask = ret_mask[..., lag:] * ret_mask[..., :-lag]
                return pair_corr(current[..., lag:], current[..., :-lag], pair_mask)

            # Signed-return dependence: lag 1 is a microstructure-inspired
            # diagnostic (e.g. bid-ask bounce), while lag 5 captures short
            # horizon persistence without claiming intraday order-flow data.
            for lag, name in ((1, 'return_autocorr_1'), (5, 'return_autocorr_5')):
                if ret_pred.shape[-1] > lag + 2:
                    pred_acf = lag_corr(ret_pred, lag)
                    targ_acf = lag_corr(ret_targ, lag)
                    add(name, F.smooth_l1_loss(pred_acf, targ_acf.detach()))

            # Volatility clustering: autocorrelation of absolute returns.
            p_abs = ret_pred.abs()
            t_abs = ret_targ.abs()
            pair_mask = ret_mask[..., 1:] * ret_mask[..., :-1]
            p_abs_next, p_abs_prev = p_abs[..., 1:], p_abs[..., :-1]
            t_abs_next, t_abs_prev = t_abs[..., 1:], t_abs[..., :-1]
            pred_abs_acf = pair_corr(p_abs_next, p_abs_prev, pair_mask)
            targ_abs_acf = pair_corr(t_abs_next, t_abs_prev, pair_mask)
            add('volatility_clustering_acf', F.smooth_l1_loss(
                pred_abs_acf, targ_abs_acf.detach()
            ))

            # Leverage effect: negative returns followed by larger absolute
            # returns, a compact daily-frequency proxy for asymmetric risk.
            pred_leverage = pair_corr(ret_pred[..., :-1], p_abs[..., 1:], pair_mask)
            targ_leverage = pair_corr(ret_targ[..., :-1], t_abs[..., 1:], pair_mask)
            add('leverage_effect', F.smooth_l1_loss(
                pred_leverage, targ_leverage.detach()
            ))

        vol_scale = float(getattr(self, 'config_volatility_scale', 0.09))
        pred_curve = torch.exp(torch.clamp(pred_cum * vol_scale, -5.0, 5.0))
        targ_curve = torch.exp(torch.clamp(targ_cum * vol_scale, -5.0, 5.0))
        pred_peak = torch.cummax(pred_curve, dim=-1).values.clamp(min=1e-6)
        targ_peak = torch.cummax(targ_curve, dim=-1).values.clamp(min=1e-6)
        pred_dd_path = pred_curve / pred_peak - 1.0
        targ_dd_path = targ_curve / targ_peak - 1.0
        # Padding must not be interpreted as a zero drawdown. Use a neutral
        # positive value at invalid positions before taking the minimum.
        pred_dd = torch.where(m > 0.5, pred_dd_path, torch.ones_like(pred_dd_path)).amin(dim=-1)
        targ_dd = torch.where(m > 0.5, targ_dd_path, torch.ones_like(targ_dd_path)).amin(dim=-1)
        add('drawdown', F.smooth_l1_loss(pred_dd, targ_dd.detach()))

        # Path-level calibration in cumulative log-price space. The model
        # predicts scaled log returns, while the downstream financial object
        # is S_t = S_0 * exp(sum return_t). Matching local returns alone can
        # still leave a small persistent drift that explodes after exponentiation.
        pred_log_path = torch.cumsum(pred * vol_scale, dim=-1)
        targ_log_path = torch.cumsum(targ * vol_scale, dim=-1)
        path_mask = m > 0.5
        pred_path_flat = pred_log_path[path_mask]
        targ_path_flat = targ_log_path[path_mask]
        if pred_path_flat.numel() >= 32 and targ_path_flat.numel() >= 32:
            path_levels = torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], device=pred.device)
            pred_path_q = torch.quantile(pred_path_flat.float(), path_levels)
            targ_path_q = torch.quantile(targ_path_flat.float(), path_levels)
            add('path_log_quantile', F.smooth_l1_loss(pred_path_q, targ_path_q.detach()))

            abs_path_levels = torch.tensor([0.50, 0.90, 0.95, 0.99], device=pred.device)
            pred_abs_path_q = torch.quantile(pred_path_flat.abs().float(), abs_path_levels)
            targ_abs_path_q = torch.quantile(targ_path_flat.abs().float(), abs_path_levels)
            add('path_abs_quantile', F.smooth_l1_loss(pred_abs_path_q, targ_abs_path_q.detach()))

        if pred_terminal.numel() >= 8:
            # This is deliberately in log-price units, not price units. It
            # directly penalizes terminal fan expansion before exp() magnifies it.
            terminal_levels = torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], device=pred.device)
            pred_log_terminal = pred_log_path.gather(-1, gather_idx).squeeze(-1)
            targ_log_terminal = targ_log_path.gather(-1, gather_idx).squeeze(-1)
            pred_log_q = torch.quantile(pred_log_terminal.float(), terminal_levels)
            targ_log_q = torch.quantile(targ_log_terminal.float(), terminal_levels)
            add('path_terminal_quantile', F.smooth_l1_loss(pred_log_q, targ_log_q.detach()))

            # A separate slope term discourages a systematic forecast drift
            # without forcing the conditional median to zero.
            horizon = count.squeeze(-1).float().clamp(min=1.0)
            pred_slope = pred_log_terminal / horizon
            targ_slope = targ_log_terminal / horizon
            add('path_drift', F.smooth_l1_loss(pred_slope, targ_slope.detach()))

        return total * max(float(self.financial_loss_scale), 0.0)

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

    def _weighted_mean(self, losses, sample_weight):
        if sample_weight is None:
            return losses.mean()
        w = sample_weight.float().reshape(-1).to(losses.device)
        if w.numel() != losses.numel():
            repeat = max(losses.numel() // max(w.numel(), 1), 1)
            w = w.repeat(repeat)
        w = torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=0.0)
        return (losses * w).sum() / w.sum().clamp(min=1e-8)

    def p_losses(self, x_start, t, cond_input=None, mask=None, global_step=0, sample_weight=None, **kwargs):
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

        # Final configuration draws coordinate-wise positive stable scales.
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
            loss = self._weighted_mean(losses, sample_weight)
        else:
            loss = self._weighted_mean(losses, sample_weight)

        denoising_loss = loss
        applied_financial_loss = loss.new_tensor(0.0)
        financial_warmup = 0.0
        if self.use_financial_regularizers:
            pred_x0 = dlpm.predict_xstart(x_t, t_ext, model_out)
            fin_loss = self._financial_regularizer_loss(pred_x0, x0, m_exp)
            if torch.isfinite(fin_loss).all():
                financial_warmup = 1.0
                if self.financial_loss_warmup_steps > 0:
                    financial_warmup = min(
                        1.0,
                        max(float(global_step), 0.0) / float(self.financial_loss_warmup_steps)
                    )
                    fin_loss = fin_loss * financial_warmup
                applied_financial_loss = fin_loss
                loss = loss + fin_loss

        if (
            self.loss_diagnostics_every > 0
            and int(global_step) % self.loss_diagnostics_every == 0
        ):
            self.last_loss_components = {
                'step': int(global_step),
                'denoising': float(denoising_loss.detach().item()),
                'financial': float(applied_financial_loss.detach().item()),
                'financial_warmup': float(financial_warmup),
                'total': float(loss.detach().item()),
            }

        return loss if torch.isfinite(loss) else None

    def forward(self, img, cond_input=None, mask=None, global_step=0, sample_weight=None, **kwargs):
        img = self.normalize(img)
        # t in [1, T): t=0 has bs=0 (zero noise), nothing to learn there
        t = torch.randint(1, self.num_timesteps, (img.shape[0],), device=img.device).long()
        return self.p_losses(
            img, t,
            cond_input=cond_input,
            mask=mask,
            global_step=global_step,
            sample_weight=sample_weight,
            **kwargs
        )

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
        if self.sample_x0_clip is not None:
            img = img.clamp(-float(self.sample_x0_clip), float(self.sample_x0_clip))
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
        if self.sample_x0_clip is not None:
            img = img.clamp(-float(self.sample_x0_clip), float(self.sample_x0_clip))
        mask = model_forward_kwargs.get('mask')
        if exists(mask):
            mask = self._expand_mask(mask, img)
            img = img * mask + noise0 * (1 - mask)
        return (img, noise0) if return_noise else img

    @torch.no_grad()
    def native_skip_sample(self, shape, clip_denoised=False,
                           model_forward_kwargs: dict = dict(),
                           return_noise=False, sampling_timesteps=None):
        """Stochastic skip sampling with the native DLPM posterior."""
        was_training = self.model.training
        self.model.eval()
        steps = default(sampling_timesteps, self.sampling_timesteps)
        dlpm = self.generative_process.dlpm
        noise0 = dlpm.barsigmas[-1] * dlpm.gen_eps.generate(size=shape)
        processed_cond = self._get_condition(model_forward_kwargs.get('cond_input'))
        model_kwargs = {'cond_input': processed_cond}
        img = self.generative_process.native_skip_sample_loop(
            self.model,
            shape=shape,
            noise=noise0,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            sampling_timesteps=steps,
            progress=True,
        )
        if was_training:
            self.model.train()
        img = self.unnormalize(img)
        if self.sample_x0_clip is not None:
            img = img.clamp(-float(self.sample_x0_clip), float(self.sample_x0_clip))
        mask = model_forward_kwargs.get('mask')
        if exists(mask):
            mask = self._expand_mask(mask, img)
            img = img * mask + noise0 * (1 - mask)
        return (img, noise0) if return_noise else img

    @torch.no_grad()
    def sample(self, batch_size=16, cond_input=None, mask=None, return_noise=False,
               sampling_timesteps=None, sampler='dlim'):
        shape = (batch_size, self.channels, self.seq_length) if self.channel_first \
            else (batch_size, self.seq_length, self.channels)

        steps = default(sampling_timesteps, self.sampling_timesteps)
        if sampler == 'native_skip':
            print(f'Native DLPM skip sampling: {steps} steps (of {self.num_timesteps})')
            return self.native_skip_sample(
                shape,
                sampling_timesteps=steps,
                model_forward_kwargs={'cond_input': cond_input, 'mask': mask},
                return_noise=return_noise,
            )
        if steps < self.num_timesteps:
            print(f'DLIM accelerated sampling: {steps} steps (of {self.num_timesteps})')
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
