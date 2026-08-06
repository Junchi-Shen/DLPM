# path_simulators.py
# 
# 这是一个纯粹的数学和计算库。
# 它包含了所有路径生成的 "How-To"。

import torch
import numpy as np
import warnings
from arch import arch_model
import gc
from pathlib import Path
from Model.Diffusion_Model.diffusion_with_condition import GaussianDiffusion1D
from Model.Diffusion_Model.diffusion_dlpm import DLPMDiffusion1D
from Model.Diffusion_Model.Unet_with_condition import Unet1D
from Model.Diffusion_Model.condition_network import EnhancedConditionNetwork
from Data.Input_preparation import DataProcessor
import json
import joblib
from scipy.stats import t as student_t
from scipy.stats import norm as normal_dist
from scipy.integrate import quad
# ==========================================================
# 1. GBM 模拟器 (来自 4.2-GBM_MC_Generator.py)
# ==========================================================

def fit_garch_model(returns_data, use_student_t=True):
    """
    在给定的收益率数据上拟合GARCH(1,1)模型。
    这段逻辑是从 4.3-Garch_MC_Generator.py 提取并整合到库中的。
    """
    print("🔄 正在实时拟合 GARCH(1,1) 模型...")
    # arch 库习惯使用“百分比”尺度的收益率
    returns = returns_data * 100.0 

    dist_name = 't' if use_student_t else 'normal'
    garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist=dist_name)
    # disp='off' 意味着在拟合时不打印收敛信息
    garch_fit = garch_model.fit(disp='off')

    # --- 提取参数 ---
    # 注意 omega 的尺度转换：从 百分比^2 转换为 小数^2
    fitted_omega = float(garch_fit.params['omega']) / 10000.0
    
    # 兼容不同版本的 arch 库的键名
    alpha_key = 'alpha[1]' if 'alpha[1]' in garch_fit.params.index else 'alpha'
    beta_key  = 'beta[1]'  if 'beta[1]'  in garch_fit.params.index else 'beta'
    
    fitted_alpha = float(garch_fit.params[alpha_key])
    fitted_beta  = float(garch_fit.params[beta_key])
    
    # 只有在使用 't' 分布时才尝试提取 'nu'
    fitted_nu = float(garch_fit.params['nu']) if 'nu' in garch_fit.params.index and use_student_t else np.nan

    FITTED_GARCH_PARAMS = {
        "omega": fitted_omega,
        "alpha": fitted_alpha,
        "beta":  fitted_beta,
        "nu":    fitted_nu
    }
    
    print("✅ GARCH 模型拟合完成:")
    print(FITTED_GARCH_PARAMS)
    print(f"   α+β = {fitted_alpha + fitted_beta:.4f}")
    if np.isfinite(fitted_nu):
        print(f"   nu = {fitted_nu:.4f}")
        
    return FITTED_GARCH_PARAMS


def simulate_gbm(S0, r, sigma, T_days, n_simulations, n_steps_total=252):
    """
    执行几何布朗运动 (GBM) 的蒙特卡洛模拟 (已对齐)。
    
    注意：此版本与 GARCH 对齐，使用 NaN 填充，而不是前向填充。
    """
    T_years = T_days / 252.0
    dt = 1.0 / 252.0
    
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    random_shocks = np.random.normal(0, 1, (T_days, n_simulations))
    log_returns = drift + diffusion * random_shocks
    log_paths = np.cumsum(log_returns, axis=0)
    
    log_paths = np.vstack([np.zeros(n_simulations), log_paths])
    log_paths += np.log(S0)
    
    price_paths_short = np.exp(log_paths).T # 形状: (n_simulations, T_days + 1)
    
    # --- 关键：使用 NaN 填充以对齐到固定长度 252 ---
    padded_paths = np.full((n_simulations, n_steps_total), np.nan)
    
    current_len = price_paths_short.shape[1]
    copy_len = min(current_len, n_steps_total) # 防止 T_days > 252
    
    padded_paths[:, :copy_len] = price_paths_short[:, :copy_len]
    
    # 最终形状: (n_simulations, 1, n_steps_total)
    return padded_paths.reshape(n_simulations, 1, n_steps_total)

# ==========================================================
# 2. GARCH 模拟器 (来自 4.3-Garch_MC_Generator.py)
# ==========================================================

def _draw_innovations(size, dist: str = "t", nu: float | None = None, rng: np.random.Generator | None = None):
    """GARCH的辅助函数：返回形状为 `size` 的创新项 z_t"""
    rng = rng or np.random.default_rng()
    if dist.lower() == "t":
        if (nu is None) or (nu <= 2.0):
            warnings.warn("nu<=2 或未提供，厚尾仿真回退为正态。")
            return rng.standard_normal(size)
        u = rng.standard_t(df=nu, size=size)
        z = u / np.sqrt(nu / (nu - 2.0)) # 标准化 => Var(z)=1
        return z
    return rng.standard_normal(size)

def fit_garch_model(returns_data, use_student_t=True):
    """
    在给定的收益率数据上拟合GARCH(1,1)模型。
    """
    print("🔄 正在实时拟合 GARCH(1,1) 模型...")
    returns = returns_data * 100.0  # arch 习惯用“百分比”尺度

    dist_name = 't' if use_student_t else 'normal'
    garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist=dist_name)
    garch_fit = garch_model.fit(disp='off')

    # 提取参数
    fitted_omega = float(garch_fit.params['omega']) / 10000.0 # 转回小数^2
    alpha_key = 'alpha[1]' if 'alpha[1]' in garch_fit.params.index else 'alpha'
    beta_key  = 'beta[1]'  if 'beta[1]'  in garch_fit.params.index else 'beta'
    fitted_alpha = float(garch_fit.params[alpha_key])
    fitted_beta  = float(garch_fit.params[beta_key])
    fitted_nu    = float(garch_fit.params['nu']) if 'nu' in garch_fit.params.index else np.nan

    FITTED_GARCH_PARAMS = {
        "omega": fitted_omega, "alpha": fitted_alpha,
        "beta":  fitted_beta,  "nu":    fitted_nu
    }
    print("✅ GARCH 模型拟合完成:")
    print(FITTED_GARCH_PARAMS)
    print(f"   α+β = {fitted_alpha + fitted_beta:.4f}")
    if np.isfinite(fitted_nu):
        print(f"   nu = {fitted_nu:.4f}")
        
    return FITTED_GARCH_PARAMS


def simulate_garch(
    S0, r, initial_vol_ann, T_days, n_simulations, n_steps_total, garch_params,
    innov_dist: str = "t", seed: int | None = 42
):
    """
    使用 GARCH(1,1) 生成价格路径。
    返回：
      paths_out: (n_simulations, 1, n_steps_total)
      sigma2_out: (n_simulations, 1, T_days+1)  # 注意：这是 *未填充* 的
    """
    rng = np.random.default_rng(seed)
    T_days = int(T_days)
    n_simulations = int(n_simulations)
    n_steps_total = int(n_steps_total)

    price_paths = np.zeros((T_days + 1, n_simulations), dtype=np.float64)
    sigma2 = np.zeros((T_days + 1, n_simulations), dtype=np.float64)
    eps_prev = np.zeros((n_simulations,), dtype=np.float64)

    price_paths[0, :] = float(S0)
    sigma2[0, :] = (float(initial_vol_ann) / np.sqrt(252.0))**2

    omega = float(garch_params['omega'])
    alpha = float(garch_params['alpha'])
    beta  = float(garch_params['beta'])
    nu    = float(garch_params.get('nu', np.nan))

    if alpha + beta >= 1.0:
        warnings.warn(f"alpha+beta = {alpha+beta:.4f} >= 1.0", UserWarning)

    r_daily = float(r) / 252.0
    z_mat = _draw_innovations(size=(n_simulations, T_days), dist=innov_dist, nu=nu, rng=rng)

    for t in range(1, T_days + 1):
        sigma2[t, :] = omega + alpha * (eps_prev**2) + beta * sigma2[t-1, :]
        sigma2[t, :] = np.maximum(sigma2[t, :], 1e-18)

        sigma_t = np.sqrt(sigma2[t, :])
        z = z_mat[:, t-1]

        log_ret = (r_daily - 0.5 * sigma_t**2) + sigma_t * z
        price_paths[t, :] = price_paths[t-1, :] * np.exp(log_ret)
        eps_prev = sigma_t * z

    # 对齐输出：价格路径 [n_sim, 1, n_steps_total]，不足部分以 NaN
    paths_T = price_paths.T  # [n_sim, T_days+1]
    paths_out = np.full((n_simulations, n_steps_total), np.nan, dtype=np.float64)
    L = min(n_steps_total, paths_T.shape[1])
    paths_out[:, :L] = paths_T[:, :L]

    # 方差路径：返回未填充的
    sig_out = sigma2.T  # [n_sim, T_days+1]
    
    return paths_out.reshape(n_simulations, 1, n_steps_total), \
           sig_out.reshape(n_simulations, 1, T_days + 1)


def _truncated_risk_neutral_innovations(size, nu, cap, rng, restandardize=True):
    """Draw symmetrically truncated Student-t innovations.

    The innovation is standardized before truncation and re-standardized after
    truncation by default, preserving the unit-innovation interpretation of
    the fitted GARCH scale.
    """
    if not np.isfinite(nu) or nu <= 2.0:
        lo, hi = -float(cap), float(cap)
        c_lo, c_hi = normal_dist.cdf(lo), normal_dist.cdf(hi)
        u = rng.uniform(c_lo, c_hi, size=size)
        z = normal_dist.ppf(u)
        return z / np.sqrt(truncated_standardized_t_variance(nu, cap)) if restandardize else z
    scale = np.sqrt(nu / (nu - 2.0))
    raw_lo, raw_hi = -float(cap) * scale, float(cap) * scale
    c_lo, c_hi = student_t.cdf(raw_lo, nu), student_t.cdf(raw_hi, nu)
    u = rng.uniform(c_lo, c_hi, size=size)
    z = student_t.ppf(u, nu) / scale
    return z / np.sqrt(truncated_standardized_t_variance(nu, cap)) if restandardize else z


def truncated_standardized_t_variance(nu, cap=8.0):
    """Return Var(z | |z| <= cap) for the standardized Student-t law."""
    if not np.isfinite(nu) or nu <= 2.0:
        prob = normal_dist.cdf(float(cap)) - normal_dist.cdf(-float(cap))
        second = quad(lambda z: z * z * normal_dist.pdf(z), -float(cap), float(cap))[0]
        return float(second / max(prob, 1e-15))
    scale = np.sqrt(float(nu) / (float(nu) - 2.0))
    raw_cap = float(cap) * scale
    prob = student_t.cdf(raw_cap, nu) - student_t.cdf(-raw_cap, nu)
    second_raw = quad(
        lambda y: y * y * student_t.pdf(y, nu), -raw_cap, raw_cap
    )[0]
    return float((second_raw / (scale * scale)) / max(prob, 1e-15))


def _risk_neutral_log_mgf(sigma, nu, cap, nodes, weights, restandardize=True):
    """Compute log E[exp(sigma*z)] under the truncated innovation law."""
    sigma = np.asarray(sigma, dtype=np.float64)
    if not np.isfinite(nu) or nu <= 2.0:
        z_nodes = float(cap) * nodes
        density = normal_dist.pdf(z_nodes)
        normalizer = normal_dist.cdf(float(cap)) - normal_dist.cdf(-float(cap))
    else:
        scale = np.sqrt(nu / (nu - 2.0))
        raw_lo, raw_hi = -float(cap) * scale, float(cap) * scale
        y_nodes = 0.5 * (raw_hi - raw_lo) * nodes + 0.5 * (raw_hi + raw_lo)
        z_nodes = y_nodes / scale
        density = student_t.pdf(y_nodes, nu)
        normalizer = student_t.cdf(raw_hi, nu) - student_t.cdf(raw_lo, nu)
        weights = weights * 0.5 * (raw_hi - raw_lo)
    if restandardize:
        z_nodes = z_nodes / np.sqrt(truncated_standardized_t_variance(nu, cap))
    if not np.isfinite(nu) or nu <= 2.0:
        weights = weights * float(cap)
    weighted_density = weights * density / max(float(normalizer), 1e-15)
    mgf = np.exp(sigma[:, None] * z_nodes[None, :]) @ weighted_density
    return np.log(np.maximum(mgf, 1e-300))


_RN_MGF_GRID_CACHE = {}


def _get_rn_mgf_grid(nu, cap, nodes, weights, sigma_max=2.5, grid_size=4097,
                      restandardize=True):
    """Cache the state-dependent martingale correction for one GARCH law."""
    key = (round(float(nu), 8), round(float(cap), 8), int(len(nodes)),
           round(float(sigma_max), 4), int(grid_size), bool(restandardize))
    if key not in _RN_MGF_GRID_CACHE:
        sigma_grid = np.linspace(0.0, float(sigma_max), int(grid_size))
        kappa_grid = _risk_neutral_log_mgf(
            sigma_grid, nu, cap, nodes, weights,
            restandardize=restandardize
        )
        _RN_MGF_GRID_CACHE[key] = (sigma_grid, kappa_grid)
    return _RN_MGF_GRID_CACHE[key]


def simulate_garch_risk_neutral(
    S0, r, initial_vol_ann, T_days, n_simulations, n_steps_total,
    garch_params, innovation_cap: float = 8.0, quadrature_order: int = 512,
    seed: int | None = 42, mgf_sigma_max: float = 2.5,
    restandardize_innovation: bool = True,
    dividend_yield: float = 0.0,
):
    """Simulate a discrete-time martingale-corrected GARCH Q benchmark.

    The historical GARCH volatility recursion is retained, but innovations are
    symmetrically truncated and each step subtracts the exact conditional
    log-moment generating function. Consequently, conditional on the current
    volatility state, E_Q[S_{t+1}|F_t] = S_t exp((r-q)/252) up to numerical
    quadrature error, where q is ``dividend_yield``. The GARCH state is a
    one-day conditional log-return variance, so no additional dt factor is
    applied to ``sqrt(sigma2)``. This is a discrete-time Q construction, not
    an Esscher transform of the untruncated Student-t law.
    """
    rng = np.random.default_rng(seed)
    T_days, n_simulations, n_steps_total = int(T_days), int(n_simulations), int(n_steps_total)
    prices = np.zeros((T_days + 1, n_simulations), dtype=np.float64)
    sigma2 = np.zeros_like(prices)
    eps_prev = np.zeros(n_simulations, dtype=np.float64)
    prices[0, :] = float(S0)
    sigma2[0, :] = (float(initial_vol_ann) / np.sqrt(252.0)) ** 2

    omega = float(garch_params["omega"])
    alpha = float(garch_params["alpha"])
    beta = float(garch_params["beta"])
    nu = float(garch_params.get("nu", np.nan))
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    sigma_grid, kappa_grid = _get_rn_mgf_grid(
        nu, innovation_cap, nodes, weights, sigma_max=mgf_sigma_max,
        restandardize=restandardize_innovation
    )
    z_mat = _truncated_risk_neutral_innovations(
        (n_simulations, T_days), nu, innovation_cap, rng,
        restandardize=restandardize_innovation
    )
    carry_daily = (float(r) - float(dividend_yield)) / 252.0

    for t in range(1, T_days + 1):
        sigma2[t, :] = np.maximum(
            omega + alpha * eps_prev ** 2 + beta * sigma2[t - 1, :], 1e-18
        )
        sigma_t = np.sqrt(sigma2[t, :])
        z = z_mat[:, t - 1]
        if np.any(sigma_t > sigma_grid[-1]):
            correction = _risk_neutral_log_mgf(
                sigma_t, nu, innovation_cap, nodes, weights,
                restandardize=restandardize_innovation
            )
        else:
            correction = np.interp(sigma_t, sigma_grid, kappa_grid)
        prices[t, :] = prices[t - 1, :] * np.exp(
            carry_daily - correction + sigma_t * z
        )
        eps_prev = sigma_t * z

    out = np.full((n_simulations, n_steps_total), np.nan, dtype=np.float64)
    out[:, :min(n_steps_total, T_days + 1)] = prices.T[:, :n_steps_total]
    return out.reshape(n_simulations, 1, n_steps_total), sigma2.T.reshape(
        n_simulations, 1, T_days + 1
    )


def simulate_garch_risk_neutral_batch(
    S0, r, initial_vol_ann, T_days, n_simulations, n_steps_total,
    garch_params, innovation_cap: float = 8.0, quadrature_order: int = 256,
    seed: int | None = 42, mgf_sigma_max: float = 2.5,
    restandardize_innovation: bool = True, mgf_grid_size: int = 513,
    dividend_yield=0.0,
):
    """Vectorized RN-Q simulation using one-day GARCH conditional variances."""
    S0 = np.asarray(S0, dtype=np.float64).reshape(-1)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    dividend_yield = np.broadcast_to(
        np.asarray(dividend_yield, dtype=np.float64), r.shape
    ).reshape(-1)
    initial_vol_ann = np.asarray(initial_vol_ann, dtype=np.float64).reshape(-1)
    T_days = np.asarray(T_days, dtype=np.int64).reshape(-1)
    n_windows = len(S0)
    max_days = int(n_steps_total) - 1
    rng = np.random.default_rng(seed)
    prices = np.full((n_windows, int(n_simulations), int(n_steps_total)), np.nan)
    sigma2 = np.zeros((n_windows, int(n_simulations)), dtype=np.float64)
    eps_prev = np.zeros_like(sigma2)
    prices[:, :, 0] = S0[:, None]
    sigma2[:, :] = (initial_vol_ann[:, None] / np.sqrt(252.0)) ** 2
    omega = float(garch_params["omega"])
    alpha = float(garch_params["alpha"])
    beta = float(garch_params["beta"])
    nu = float(garch_params.get("nu", np.nan))
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    sigma_grid, kappa_grid = _get_rn_mgf_grid(
        nu, innovation_cap, nodes, weights, sigma_max=mgf_sigma_max,
        grid_size=mgf_grid_size, restandardize=restandardize_innovation
    )
    z = _truncated_risk_neutral_innovations(
        (n_windows, int(n_simulations), max_days), nu, innovation_cap, rng,
        restandardize=restandardize_innovation
    )
    carry_daily = (r - dividend_yield)[:, None] / 252.0
    for t in range(1, int(n_steps_total)):
        sigma2 = np.maximum(omega + alpha * eps_prev ** 2 + beta * sigma2, 1e-18)
        sigma_t = np.sqrt(sigma2)
        correction = np.interp(sigma_t.ravel(), sigma_grid, kappa_grid).reshape(sigma_t.shape)
        prices[:, :, t] = prices[:, :, t - 1] * np.exp(
            carry_daily - correction + sigma_t * z[:, :, t - 1]
        )
        eps_prev = sigma_t * z[:, :, t - 1]
    for i, maturity in enumerate(T_days):
        if maturity + 1 < n_steps_total:
            prices[i, :, int(maturity) + 1:] = np.nan
    return prices.reshape(n_windows * int(n_simulations), 1, int(n_steps_total))


# ==========================================================
# 3. UNet/Diffusion 模拟器 (来自 4.1-Unet_Generator_Optimized.py)
# ==========================================================

# 辅助函数：加载模型
def load_diffusion_artifacts(
    model_dir: Path,
    processor_filename: str,
    model_filename: str,
    condition_network_filename: str | None,
    unet_config: dict,
    diffusion_config: dict,
    cond_net_config: dict | None,
    device: str,
    use_dlpm: bool = False,
    dlpm_alpha: float = 1.7 
):
    """
    自适应维度加载器：自动从权重探测维度，实现全局模型对局部资产的‘自由生成’。
    """
    print("🔄 正在加载扩散模型产出物...")

    # --- 1. 加载 DataProcessor (用于逆向转换，不用于决定维度) ---
    processor_path = model_dir / processor_filename
    data_processor = joblib.load(processor_path)

    # --- 2. 动态维度探测 (解决 Size Mismatch) ---
    condition_network = None
    if condition_network_filename and cond_net_config:
        cond_net_path = model_dir / condition_network_filename
        # 先加载权重字典以探测训练时的“舞台大小”
        state_dict_cond = torch.load(cond_net_path, map_location=device)
        state_dict_cond = {k.replace('module.', ''): v for k, v in state_dict_cond.items()}
        
        # 核心：直接从权重矩阵的 shape 提取维度 (如 7 和 18)
        trained_countries = state_dict_cond['country_embedding.weight'].shape[0]
        trained_indices = state_dict_cond['index_embedding.weight'].shape[0]
        
        print(f"      - 🚀 模型自由化：自动对齐训练维度 ({trained_countries} 国家, {trained_indices} 指数)")

        condition_network = EnhancedConditionNetwork(
            num_countries=trained_countries, 
            num_indices=trained_indices,
            **cond_net_config
        ).to(device)
        
        condition_network.load_state_dict(state_dict_cond)
        condition_network.eval()

    # --- 3. 初始化模型结构 (U-Net) ---
    model_path = model_dir / model_filename
    unet_cond_dim = cond_net_config['output_dim'] if cond_net_config else 7
    model = Unet1D(cond_dim=unet_cond_dim, **unet_config.get("model_params", {})).to(device)

    # --- 4. 构建 Diffusion 包装器 ---
    if use_dlpm:
        diffusion = DLPMDiffusion1D(model=model, condition_network=condition_network, **diffusion_config).to(device)
    else:
        diffusion = GaussianDiffusion1D(model=model, condition_network=condition_network, **diffusion_config).to(device)

    # --- 5. 权重加载与对象对齐 (修复版) ---
    print(f"   🔄 正在从权重文件同步物理参数: {model_path.name}")
    state_dict = torch.load(model_path, map_location=device)
    
    # 自动清洗 DataParallel 前缀
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # --- 关键修改点：分层加载 ---
    # 1. 提取并加载 U-Net 内部权重
    # 如果 pth 里已经是 model.xxxx 格式，直接加载；
    # 如果是 init_conv 格式，我们需要把它们对应到 diffusion.model 上
    if any(k.startswith('init_conv') for k in state_dict):
        # 说明 pth 保存的是 Unet 内部权重，将其手动挂载到 diffusion.model
        diffusion.model.load_state_dict(state_dict, strict=False)
        print("      - ✅ 已成功同步 U-Net 网络权重")
    else:
        # 说明 pth 保存的是完整的 Diffusion 对象权重
        diffusion.load_state_dict(state_dict, strict=False)
        print("      - ✅ 已成功同步全量模型权重")

    # 2. 提取并同步 Alpha (核心物理参数)
    if 'learnable_alpha' in state_dict:
        trained_alpha = state_dict['learnable_alpha'].item()
        diffusion.learnable_alpha.data = torch.tensor(float(trained_alpha)).to(device)
        
        if use_dlpm:
            # 强制同步给底层物理引擎
            diffusion.generative_process.dlpm.alpha = trained_alpha
            diffusion.generative_process.dlpm.A = None # 强制重置矩阵缓存
            diffusion.generative_process.dlpm.Sigmas = None
            print(f"      - 🎯 物理对齐：已自动提取 Alpha = {trained_alpha:.6f}")
    
    diffusion.eval()
    return diffusion, data_processor
# 核心生成函数 (批量)
def _generate_paths_for_condition_batch(condition_batch, diffusion, total_paths, batch_size, device):
    """为一批条件同时生成路径 - 核心优化函数"""
    generated_paths = []
    
    for i, condition in enumerate(condition_batch):
        single_condition = condition.unsqueeze(0).to(device)
        
        paths_for_this_condition = []
        num_batches = (total_paths + batch_size - 1) // batch_size
        
        for _ in range(num_batches):
            num_remaining = total_paths - len(paths_for_this_condition)
            current_batch_size = min(batch_size, num_remaining)
            if current_batch_size <= 0:
                break
            
            with torch.no_grad():
                conditions_batch = single_condition.repeat(current_batch_size, 1)
                generated_batch = diffusion.sample(
                    batch_size=current_batch_size, 
                    cond_input=conditions_batch
                )
                paths_for_this_condition.append(generated_batch.cpu().numpy())
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
        
        full_ensemble = np.concatenate(paths_for_this_condition, axis=0)
        generated_paths.append(full_ensemble)
    
    return generated_paths

# 格式化时间
def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分钟"

# 模式 1: 并行优化
def run_diffusion_parallel_optimized(conditions, diffusion, gen_params, device):
    """并行优化版本的路径生成函数"""
    from tqdm import tqdm
    import time

    total_paths = gen_params['num_paths_to_generate']
    batch_size = gen_params['generation_batch_size']
    condition_batch_size = gen_params.get('condition_batch_size', 8)
    
    num_conditions = conditions.shape[0]
    all_generated_paths = []
    
    print(f"🚀 开始并行生成路径...")
    print(f"   总条件数: {num_conditions}, 每条件路径数: {total_paths}")
    print(f"   条件批处理大小: {condition_batch_size}, 生成批处理大小: {batch_size}")
    
    num_condition_batches = (num_conditions + condition_batch_size - 1) // condition_batch_size
    start_time = time.time()
    
    for batch_idx in tqdm(range(num_condition_batches), desc="处理条件批次"):
        start_idx = batch_idx * condition_batch_size
        end_idx = min(start_idx + condition_batch_size, num_conditions)
        condition_batch = conditions[start_idx:end_idx]
        
        batch_paths = _generate_paths_for_condition_batch(
            condition_batch, diffusion, total_paths, batch_size, device
        )
        all_generated_paths.extend(batch_paths)
        
        if batch_idx % 5 == 0:
            gc.collect()
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
        
        # 实时进度
        current_time = time.time()
        elapsed_time = current_time - start_time
        completed_conditions = end_idx
        if completed_conditions > 0:
            avg_time_per_condition = elapsed_time / completed_conditions
            remaining_conditions = num_conditions - completed_conditions
            estimated_remaining_time = remaining_conditions * avg_time_per_condition
            conditions_per_second = completed_conditions / elapsed_time
            
            if batch_idx % 10 == 0 or batch_idx == num_condition_batches - 1:
                print(f"\n📊 进度: {completed_conditions}/{num_conditions} ({completed_conditions/num_conditions*100:.1f}%)")
                print(f"   ⏱️  已用时间: {_format_time(elapsed_time)}, 预计剩余: {_format_time(estimated_remaining_time)}")
                print(f"   🚀 生成速度: {conditions_per_second:.2f} 条件/秒")

    total_time = time.time() - start_time
    print(f"\n✅ 路径生成完成！总用时: {_format_time(total_time)}")
    return all_generated_paths

# 模式 2: 超级批处理
def run_diffusion_mega_batch(conditions, diffusion, gen_params, device):
    """超级批处理版本 - 最大化GPU利用率"""
    from tqdm import tqdm
    import time
    
    total_paths = gen_params['num_paths_to_generate']
    batch_size = gen_params['generation_batch_size']
    num_conditions = conditions.shape[0]
    all_generated_paths = []
    sampling_steps = gen_params.get('sampling_timesteps', 200)
    
    print(f"🚀 开始超级批处理生成...")
    print(f"   总条件数: {num_conditions}, 每条件路径数: {total_paths}, 生成批处理大小: {batch_size}")
    
    start_time = time.time()
    
    for i in tqdm(range(num_conditions), desc="处理市场条件"):
        single_condition = conditions[i:i+1].to(device)
        paths_for_this_condition = []
        num_batches = (total_paths + batch_size - 1) // batch_size
        
        for _ in tqdm(range(num_batches), desc=f"条件 {i} 批次", leave=False):
            num_remaining = total_paths - len(paths_for_this_condition)
            current_batch_size = min(batch_size, num_remaining)
            if current_batch_size <= 0:
                break
            
            with torch.no_grad():
                conditions_batch = single_condition.repeat(current_batch_size, 1)
                generated_batch = diffusion.sample(
                    batch_size=current_batch_size, 
                    cond_input=conditions_batch,
                    sampling_timesteps = sampling_steps
                )
                paths_for_this_condition.append(generated_batch.cpu().numpy())
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
        
        full_ensemble = np.concatenate(paths_for_this_condition, axis=0)
        all_generated_paths.append(full_ensemble)
        
        # 实时进度
        if i % 10 == 0 or i == num_conditions - 1:
            current_time = time.time()
            elapsed_time = current_time - start_time
            completed_conditions = i + 1
            if completed_conditions > 0:
                avg_time_per_condition = elapsed_time / completed_conditions
                remaining_conditions = num_conditions - completed_conditions
                estimated_remaining_time = remaining_conditions * avg_time_per_condition
                conditions_per_second = completed_conditions / elapsed_time
                print(f"\n📊 进度: {completed_conditions}/{num_conditions} ({completed_conditions/num_conditions*100:.1f}%)")
                print(f"   ⏱️  已用时间: {_format_time(elapsed_time)}, 预计剩余: {_format_time(estimated_remaining_time)}")
                print(f"   🚀 生成速度: {conditions_per_second:.2f} 条件/秒")
    
    total_time = time.time() - start_time
    print(f"\n✅ 路径生成完成！总用时: {_format_time(total_time)}")
    return all_generated_paths
        
    
