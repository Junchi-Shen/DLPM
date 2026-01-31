import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm.auto import tqdm
from functools import partial
from random import random

from .Utils import default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, exists
from .DLPM.generative_levy_process import GenerativeLevyProcess
from .DLPM.dlpm_core import ModelMeanType, ModelVarType

class DLPMDiffusion1D(nn.Module):
    def __init__(
        self,
        *,
        seq_length,
        timesteps=1000,
        sampling_timesteps=None,
        alpha=1.75,
        objective='pred_noise',
        auto_normalize=True,
        model,
        condition_network: nn.Module = None,
        **kwargs
    ):
        super().__init__()
        self.model = model
        
        # 1. 属性配置与显式设置
        channels_arg = kwargs.get('channels', None)
        self.channels = channels_arg if channels_arg is not None else self.model.channels
        sc = kwargs.get('self_condition', None)
        self.self_condition = (sc if sc is not None else getattr(self.model, 'self_condition', False))
        self.channel_first = kwargs.get('channel_first', True)
        self.seq_length = seq_length
        self.objective = 'pred_noise' # 锁定物理层模式

        # 2. Alpha 参数与条件编码
        self.learnable_alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.condition_network = condition_network
        self.has_condition_network = condition_network is not None
        self.cond_out_dim = getattr(condition_network, 'output_dim', None) if self.has_condition_network else None

        # 3. 物理引擎 (锁定 EPSILON 以适配底层断言)
        self.generative_process = GenerativeLevyProcess(
            alpha=alpha,
            device=next(model.parameters()).device,
            reverse_steps=timesteps,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED,
            scale=kwargs.get('dlpm_scale', 'scale_preserving')
        )

        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else timesteps
        self.ema_beta = kwargs.get('ema_beta', 0.99)
        
        # 4. 指标注册
        self.metrics = ['global_vol', 'heavy_tail', 'vol_clustering', 'spectral', 'drift', 'relative_jump', 'quantile', 'skewness']
        for name in self.metrics:
            self.register_buffer(f'ema_{name}', torch.tensor(0.0))

        self.warmup_steps = int(kwargs.get('train_num_steps', 20000) * kwargs.get('warmup_ratio', 0.15))
        self.auto_normalize = auto_normalize
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = identity 
        self.ddim_sampling_eta = kwargs.get('ddim_sampling_eta', 0.0)
        self.debug_check = kwargs.get('debug_check', False)          # 前向finite检查
        self.debug_grad_hook = kwargs.get('debug_grad_hook', False)  # 分支项梯度hook定位
        self.debug_raise = kwargs.get('debug_raise', False)          # 发现坏值是否直接raise（默认False保持原行为：return None）

    # --- [修正 P0-1] 工业级鲁棒辅助工具 ---

    def _get_is_rank0(self):
        dist = getattr(torch, "distributed", None)
        if dist is None: return True
        return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    def _expand_mask(self, mask, ref):
        """[修正 P0-1] 递归补维广播，支持任意形状 Mask 输入"""
        if mask is None: return None
        mask = mask.float()
        while mask.dim() < ref.dim():
            mask = mask.unsqueeze(1) # 逐级补齐维度 (B,T) -> (B,1,T) -> (B,1,1,T)
        return mask.expand_as(ref) if mask.shape != ref.shape else mask

    def _get_condition(self, cond_input):
        if cond_input is None: return None
        c = cond_input.clone()
        c = torch.nan_to_num(cond_input, nan=0.0, posinf=10.0, neginf=-10.0)
        c[:, :5] = torch.clamp(c[:, :5], min=-10.0, max=10.0)
        if not self.has_condition_network: return c
        if self.cond_out_dim is None:
            return self.condition_network(c) if (cond_input.dim() == 2 and cond_input.shape[-1] == 7) else c
        p_cond = self.condition_network(c)
        p_cond = torch.clamp(p_cond, min=-15.0, max=15.0) # 约束 Embedding 空间量级
        return p_cond
       
    def _probe_branch_grads(self, term_cache: dict, param: torch.Tensor, global_step: int, every: int = 100):
        """
        term_cache: 你循环里保存的 {name: term}，term 是标量Tensor
        param: 你要定位的关键参数，例如 self.model.init_conv.weight
        """
        if (global_step % every) != 0:
            return
        if not self._get_is_rank0():
            return
        if param is None or (not param.requires_grad):
            return

        print(f"\n🔎 [BranchGrad Probe] step={global_step} param=init_conv.weight")
        for name, term in term_cache.items():
            if term is None or (not term.requires_grad):
                continue
            try:
                g = torch.autograd.grad(
                    term, param,
                    retain_graph=True,  # 不影响后续总loss backward
                    allow_unused=True
                )[0]
            except Exception as e:
                print(f"   - {name}: grad error -> {repr(e)}")
                continue

            if g is None:
                print(f"   - {name}: grad=None (unused)")
                continue

            finite = torch.isfinite(g)
            nan = torch.isnan(g).sum().item()
            inf = torch.isinf(g).sum().item()
            mx = g[finite].abs().max().item() if finite.any() else float("nan")
            print(f"   - {name:<15} | nan={nan:<6} inf={inf:<6} max_abs_finite={mx:.3e}")
    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, t, eps=None):
        return self.generative_process.q_sample(x_start=x_start, t=t, eps=eps)

    def _power_spectrum(self, x, mask=None, eps=1e-8):
        x = torch.nan_to_num(x, nan=0.0)
        m = self._expand_mask(mask, x)
        if m is not None:
            mean = (x * m).sum(dim=-1, keepdim=True) / m.sum(dim=-1, keepdim=True).clamp(min=1.0)
            x = torch.where(m > 0.5, x, mean)
        Xf = torch.fft.rfft(x, dim=-1)
        P = (Xf.real**2 + Xf.imag**2 + eps).sqrt()
        max_p = P.amax(dim=-1, keepdim=True).clamp(min=1e-8)
        return P / max_p

    def _masked_statistics(self, x, m_exp, eps=1e-6):
        if m_exp is None:
            return x.mean(dim=-1, keepdim=True), torch.sqrt(x.var(dim=-1, keepdim=True) + eps).clamp(min=1e-5)
        count = m_exp.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (x * m_exp).sum(dim=-1, keepdim=True) / count
        var = (((x - mean)**2) * m_exp).sum(dim=-1, keepdim=True) / count
        std = torch.sqrt(var + eps).clamp(min=1e-5)
        return mean, std

    def _masked_quantile(self, x, mask, q: float):
        B, C, T = x.shape
        x_flat = x.view(B * C, T).float() # 分位数强制 FP32
        m_flat = self._expand_mask(mask, x).view(B * C, T) if mask is not None else None
        if m_flat is None:
            return x_flat.quantile(q, dim=-1).view(B, C, 1), torch.ones(B, C, 1, device=x.device)
        sample_min = x_flat.amin(dim=-1, keepdim=True).detach() - 1.0
        x_filled = torch.where(m_flat > 0.5, x_flat, sample_min)
        valid_gate = ((m_flat > 0.5).float().mean(dim=-1, keepdim=True) > 0.2).float()
        return x_filled.quantile(q, dim=-1).view(B, C, 1), valid_gate.view(B, C, 1)

    # --- [修正 P1-4] 黑盒拦截与异常诊断 ---

    def _report_error(self, stage, x_start, pred_x0=None, model_out=None):
        if not self._get_is_rank0(): return
        print(f"\n🚨 [拦截报告] {stage}")
        with torch.no_grad():
            t_min = x_start.amin().detach().float().item(); t_max = x_start.amax().detach().float().item()
            print(f"   - Target Range: [{t_min:.4f}, {t_max:.4f}]")
            if pred_x0 is not None and torch.isfinite(pred_x0).all():
                p_min = pred_x0.amin().detach().float().item(); p_max = pred_x0.amax().detach().float().item()
                print(f"   - PredX0 Range: [{p_min:.4f}, {p_max:.4f}]")
            if model_out is not None:
                print(f"   - ModelOut Finite: {torch.isfinite(model_out).all().item()}")
        print("-" * 45)

    def _stats(self, x: torch.Tensor):
        if x is None:
            return None
        with torch.no_grad():
            finite = torch.isfinite(x)
            nan_cnt = int(torch.isnan(x).sum().item())
            inf_cnt = int(torch.isinf(x).sum().item())
            max_abs = float(x[finite].abs().max().item()) if finite.any() else float("nan")
            mean = float(x[finite].mean().item()) if finite.any() else float("nan")
            std  = float(x[finite].std().item()) if finite.any() else float("nan")
            return {
                "shape": tuple(x.shape),
                "dtype": str(x.dtype),
                "nan": nan_cnt,
                "inf": inf_cnt,
                "max_abs_finite": max_abs,
                "mean_finite": mean,
                "std_finite": std,}

    def _check_finite(self, name: str, x: torch.Tensor, stage: str, global_step: int, hard: bool=False):
        """旁路检查，不改变x；hard=True时可选raise"""
        if not self.debug_check:
            return True
        if x is None:
            return True
        ok = torch.isfinite(x).all().item()
        if ok:
            return True

        if self._get_is_rank0():
            print(f"\n🚨 [FiniteCheck Fail] step={global_step} stage={stage} name={name}")
            print("   ", self._stats(x))
            print("-" * 80)

        if hard or self.debug_raise:
            raise FloatingPointError(f"Non-finite detected at {stage}:{name} step={global_step}")
        return False

    def _make_grad_hook(self, name: str, stage: str, global_step: int):
        """给每个分支term挂梯度hook：哪一项的梯度先炸，就打印哪一项"""
        def _hook(grad):
            if grad is None:
                return grad
            if not torch.isfinite(grad).all():
                if self._get_is_rank0():
                    print(f"\n🚨 [GradHook Fail] step={global_step} stage={stage} term={name}")
                    print("   grad:", self._stats(grad))
                    print("-" * 80)
                if self.debug_raise:
                    raise FloatingPointError(f"Non-finite grad at {stage}:{name} step={global_step}")
            return grad
        return _hook

    # --- 训练核心：全链路防爆 ---

    def p_losses(self, x_start, t, cond_input=None, noise=None, mask=None, global_step=0, **kwargs):
        p_cond = self._get_condition(cond_input)
        if p_cond is not None and not torch.isfinite(p_cond).all():
            self._report_error("Condition NaN after cleaning", x_start)
            return None
        if not torch.isfinite(x_start).all():
            print(f"❌ [数据源异常] Step {global_step} 传入的 x_start 包含 NaN")
            return None
        
        
        # [修正 P1] 入口规范化 Mask
        if mask is not None:
            mask = mask.float()
            if mask.dim() == 2: mask = mask.unsqueeze(1)
        
        if not torch.isfinite(self.learnable_alpha).all():
            self._report_error("Alpha NaN Reset", x_start)
            with torch.no_grad(): self.learnable_alpha.copy_(torch.tensor(1.75).to(self.learnable_alpha.device))
        with torch.amp.autocast("cuda",enabled=False): # 强制关闭此段的自动混合精度
            current_alpha = torch.clamp(self.learnable_alpha, 1.5, 2.0).float()
        self.generative_process.dlpm.alpha = current_alpha
        eps_common = 1e-6

        # [拦截 1] 噪声检查
        noise = noise if noise is not None else self.generative_process.dlpm.gen_eps.generate(size=x_start.shape)
        noise = torch.clamp(noise, -8.0, 8.0)
        if not torch.isfinite(noise).all(): return None
        
        x_t, _ = self.q_sample(x_start=x_start, t=t, eps=noise)
        m_exp = self._expand_mask(mask, x_start)
        
        if x_t.abs().max() > 100:
            print(f"⚠️ [扩散爆炸预警] Step {global_step} | t={t.min().item()}~{t.max().item()} | x_t Max={x_t.abs().max().item():.2f}")
            x_t = torch.clamp(x_t, -20.0, 20.0)

        # 1. 模型预测与自条件
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                out = self.model(x_t, time=t, cond_input=p_cond)
                x_self_cond = self.generative_process.dlpm.predict_xstart(x_t, t, out).detach()

        model_out = self.model(x_t, time=t, cond_input=p_cond, y_self_cond=x_self_cond)
        
        if not torch.isfinite(model_out).all():
        # 抓取崩溃现场的关键上下文
            print(f"\n🚨 [崩溃现场采样] Step: {global_step}")
            print(f"   - 时间步 t 范围: {t.float().mean().item():.1f}")
            print(f"   - x_t 统计: Mean={x_t.mean().item():.4f}, Std={x_t.std().item():.4f}")
            print(f"   - 条件输入检查: Finite={torch.isfinite(p_cond).all().item() if p_cond is not None else 'N/A'}")
            return None
        
        # [拦截 2] 输出网关
        if not torch.isfinite(model_out).all():
            self._report_error("Model Out NaN", x_start, model_out=model_out); return None

        pred_x0_raw = self.generative_process.dlpm.predict_xstart(x_t, t, model_out)
        pred_x0 = torch.clamp(pred_x0_raw, -2.5, 2.5)

        # [拦截 3] PredX0 网关
        if not torch.isfinite(pred_x0).all():
            self._report_error("Pred_X0 NaN", x_start, pred_x0=pred_x0); return None

        mse_un = F.smooth_l1_loss(model_out, noise, reduction='none')
        base_loss = mse_un.mean(dim=(1, 2)).mean()
        if not torch.isfinite(base_loss): return None

        # --- 8维对齐 (带 FP32 精度隔离) ---
        c_losses = {}
        p_mn, p_sd = self._masked_statistics(pred_x0, m_exp, eps_common)
        t_mn, t_sd = self._masked_statistics(x_start, m_exp, eps_common)
        c_losses['global_vol'] = F.l1_loss(p_sd, t_sd).mean()

        # [修正 P0-2] 强制 FP32 与合理 Clip，防止 Lévy 跳跃引发的数值爆炸
        def _get_kurt(x, m, mn, sd):
            z = ((x - mn) / sd).clamp(-10, 10)
            z4 = torch.pow(z, 4)
            if m is not None:
                return ((z4 * m).sum(-1) / m.sum(-1).clamp(min=1.0)) - 3.0
            return z4.mean(-1) - 3.0
        
        def _get_skew(x, m, mn, sd):
            z = ((x - mn) / sd).clamp(-10, 10)
            z3 = torch.pow(z, 3)
            if m is not None:
                return (z3 * m).sum(-1) / m.sum(-1).clamp(min=1.0)
            return z3.mean(-1)

        p_kurt, t_kurt = _get_kurt(pred_x0, m_exp, p_mn, p_sd), _get_kurt(x_start, m_exp, t_mn, t_sd)
        c_losses['heavy_tail'] = F.l1_loss(p_kurt, t_kurt).mean()
        c_losses['skewness'] = F.l1_loss(_get_skew(pred_x0, m_exp, p_mn, p_sd), _get_skew(x_start, m_exp, t_mn, t_sd)).mean()

        win, stride = max(8, self.seq_length//8), max(4, self.seq_length//32)
        p_win, t_win = pred_x0.unfold(-1, win, stride), x_start.unfold(-1, win, stride)
        m_win_exp = self._expand_mask(mask.unfold(-1, win, stride) if mask is not None else None, p_win)
        _, pv = self._masked_statistics(p_win, m_win_exp, eps_common); _, tv = self._masked_statistics(t_win, m_win_exp, eps_common)
        c_losses['vol_clustering'] = F.smooth_l1_loss(pv, tv).mean()

        # [修正 P0-2] Drift 标度校准 (均值漂移模式)
        m_drift = m_exp if m_exp is not None else torch.ones_like(x_start)
        den_d = m_drift.sum(-1, keepdim=True).clamp(min=1.0)
        c_losses['drift'] = F.smooth_l1_loss((pred_x0 * m_drift).sum(-1, keepdim=True)/den_d, (x_start * m_drift).sum(-1, keepdim=True).detach()/den_d, beta=1.0).mean()

        p_diff, t_diff = pred_x0[..., 1:] - pred_x0[..., :-1], (x_start[..., 1:] - x_start[..., :-1]).detach()
        m_j = self._expand_mask((mask[..., 1:] * mask[..., :-1]) if mask is not None else None, p_diff)
        den_j = m_j.sum() if m_j is not None else torch.tensor(float(p_diff.numel()), device=p_diff.device, dtype=p_diff.dtype)
        c_losses['relative_jump'] = (F.l1_loss(p_diff, t_diff, reduction='none') * (m_j.float() if m_j is not None else 1.0)).sum() / den_j.clamp(min=1.0)

        # Spectral 跳算
        if global_step < 1000 or global_step % 4 == 0:
            c_losses['spectral'] = F.mse_loss(self._power_spectrum(pred_x0, mask), self._power_spectrum(x_start, mask)).mean()
        else:
            c_losses['spectral'] = torch.tensor(0.0, device=x_start.device)

        pq, v_gate = self._masked_quantile(pred_x0, mask, 0.99)
        tq, _ = self._masked_quantile(x_start, mask, 0.99)
        c_losses['quantile'] = (F.mse_loss(pq, tq.detach(), reduction='none') * v_gate).mean()

        # 权重聚合
        snr = (self.generative_process.dlpm.bargammas[t]**2) / (self.generative_process.dlpm.barsigmas[t]**2 + 1e-5)
        sw = torch.sigmoid((snr - 0.05) / 0.05).view(-1, 1, 1)
        g_v_gate = (m_exp.mean(dim=(1, 2), keepdim=True) > 0.1).float() if m_exp is not None else torch.ones((x_start.shape[0], 1, 1), device=x_start.device)

        weights = self._get_annealed_weights(global_step)
        total_loss = base_loss
        
        do_log = global_step % 100 == 0 and self._get_is_rank0()
        if do_log:
            with torch.no_grad(): av, blv = current_alpha.cpu().item(), base_loss.cpu().item()
            print(f"\n📡 [驾驶舱] Step: {global_step} | Alpha: {av:.4f} | BaseLoss: {blv:.4f}\n" + "-"*85)
        term_cache = {}
        for name in self.metrics:
            val = c_losses[name]

            # ✅ 分支 forward 值检查（不改变功能：不ok时保持你原来逻辑：continue）
            if not self._check_finite(name=f"c_losses[{name}]", x=val, stage="p_losses/c_losses", global_step=global_step):
                continue

            ema_buf = getattr(self, f'ema_{name}')
            with torch.no_grad():
                ema_buf.copy_(ema_buf * self.ema_beta + val.detach() * (1 - self.ema_beta))
            w = weights.get(name, 0.0)

            if w <= 0:
                continue

            term = (w * (val / (ema_buf + 1e-6)) * sw * g_v_gate).mean()

            # ✅ term forward 检查（term是标量，但也可能变成nan）
            if not self._check_finite(name=f"term[{name}]", x=term, stage="p_losses/term", global_step=global_step):
                continue

            # ✅ 关键：分支梯度hook（哪一项的梯度炸了会直接报出term名）
            if self.debug_grad_hook and term.requires_grad:
                term.register_hook(self._make_grad_hook(name=name, stage="p_losses/term_grad", global_step=global_step))

            total_loss = total_loss + term
            term_cache[name] = term
            try:
                p = getattr(self.model, "init_conv", None)
                target_p = getattr(p, "weight", None) if p is not None else None
                self._probe_branch_grads(term_cache, target_p, global_step, every=100)
            except Exception as _:
                pass

            if do_log:
                with torch.no_grad():
                    rv, ev = val.cpu().item(), ema_buf.cpu().item()
                print(f"📊 {name.upper():<15} | Raw: {rv:.4f} | EMA: {ev:.4f} | Weight: {w:.1f}")
        

        # Alpha 引导强度限制
        if global_step > 4000:
            with torch.no_grad():
                guide = torch.tanh((torch.tanh((p_kurt.mean() - t_kurt.mean())/5.0) + 1.5*torch.tanh((p_diff.abs().mean() - t_diff.abs().mean())/0.05)))
            total_loss -= current_alpha * guide * 0.1
            if do_log: print(f"🧬 Alpha Guidance   | Dir: {guide.item():.4f}")

        if do_log: print("="*85 + "\n")
        return total_loss if torch.isfinite(total_loss) else None

    def _get_annealed_weights(self, global_step):
        s = min(1.0, global_step / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        return {'global_vol': 8.0*s, 'heavy_tail': 4.0*s, 'vol_clustering': 4.0*s, 'spectral': 3.0*s, 'drift': 1*s, 'relative_jump': 2.0*s, 'quantile': 3.0*s, 'skewness': 1.5*s}

    # --- 采样接口 ---

    @torch.no_grad()
    def p_sample_loop(self, shape, return_noise=False, model_forward_kwargs: dict = dict()):
        was_training = self.model.training; self.model.eval()
        
        # [修正 P1] 采样阶段 Mask 规范化
        mask = model_forward_kwargs.get('mask')
        if mask is not None:
            mask = mask.float()
            if mask.dim() == 2: mask = mask.unsqueeze(1)
            model_forward_kwargs['mask'] = mask

        if self.generative_process.dlpm.A is None:
            self.generative_process.dlpm.sample_A(shape, self.num_timesteps); self.generative_process.dlpm.compute_Sigmas()
        
        pc = self._get_condition(model_forward_kwargs.get('cond_input'))
        noise0 = self.generative_process.dlpm.barsigmas[-1] * self.generative_process.dlpm.gen_eps.generate(size=shape)
        img, x_start = noise0, None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='DLPM Sampling'):
            m = self._expand_mask(model_forward_kwargs.get('mask'), img)
            if m is not None: img = img * m + noise0 * (1 - m)
            out = self.generative_process.p_sample(self.model, img, torch.full((shape[0],), t, device=img.device, dtype=torch.long), model_kwargs={'cond_input': pc, 'y_self_cond': x_start})
            pred_raw = out.get("pred_xstart", torch.zeros_like(img))
            img, x_start = out["sample"], torch.clamp(pred_raw, -2.5, 2.5)
            if t < 50: img = torch.clamp(img, -2.5, 2.5)

        if was_training: self.model.train()
        img = self.unnormalize(img)
        mf = self._expand_mask(model_forward_kwargs.get('mask'), img)
        if mf is not None: img = img * mf + noise0 * (1 - mf)
        return (img, noise0) if return_noise else img

    def forward(self, img, cond_input=None, mask=None, global_step=0, **kwargs):
        # 缺陷 1：forward 不再预处理条件，直接透传原始 tensor 
        img = self.normalize(img)
        t = torch.randint(0, self.num_timesteps, (img.shape[0],), device=img.device).long()
        return self.p_losses(img, t, cond_input=cond_input, mask=mask, global_step=global_step, **kwargs)
    
    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True, model_forward_kwargs: dict = dict(), 
                return_noise=False, sampling_timesteps=None):
        # 1. 确定采样步数
        steps = default(sampling_timesteps, self.sampling_timesteps)
    
        # 2. 生成初始噪声（固定“宇宙噪声”）
        noise0 = self.generative_process.dlpm.barsigmas[-1] * \
             self.generative_process.dlpm.gen_eps.generate(size=shape)
    
        # 3. 准备条件
        processed_cond = self._get_condition(model_forward_kwargs.get('cond_input'))
        ddim_model_kwargs = model_forward_kwargs.copy()
        ddim_model_kwargs['cond_input'] = processed_cond

        # 4. 调用底层加速循环
        img = self.generative_process.ddim_sample_loop(
        self.model,
        shape=shape,
        noise=noise0,
        clip_denoised=clip_denoised,
        model_kwargs=ddim_model_kwargs,
        eta=self.ddim_sampling_eta,
        sampling_timesteps=steps, # 关键：将步数传给底层
        progress=True             # 开启进度条显示加速后的步数
        )
    
        # 5. 后处理与 Mask 混合
        img = self.unnormalize(img)
        mask = model_forward_kwargs.get('mask')
        if exists(mask):
            mask = self._expand_mask(mask, img)
            img = img * mask + noise0 * (1 - mask)
        
        return (img, noise0) if return_noise else img
    @torch.no_grad()
    def sample(self, batch_size=16, cond_input=None, mask=None, return_noise=False, sampling_timesteps=None):
        shape = (batch_size, self.channels, self.seq_length) if self.channel_first else (batch_size, self.seq_length, self.channels)
    
        # 确定是否使用 DDIM
        steps = default(sampling_timesteps, self.sampling_timesteps)
        is_ddim = steps < self.num_timesteps
    
        # 根据判断结果调用不同的函数
        if is_ddim:
            print(f"🚀 Using DDIM Acceleration: {steps} steps (Total {self.num_timesteps})")
            return self.ddim_sample(
            shape, 
            sampling_timesteps=steps,
            model_forward_kwargs={'cond_input': cond_input, 'mask': mask}, 
            return_noise=return_noise
            )
        else:
            return self.p_sample_loop(
            shape, 
            model_forward_kwargs={'cond_input': cond_input, 'mask': mask}, 
            return_noise=return_noise
        )