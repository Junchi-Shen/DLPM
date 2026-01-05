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


# ==========================================================
# 3. UNet/Diffusion 模拟器 (来自 4.1-Unet_Generator_Optimized.py)
# ==========================================================

# 辅助函数：加载模型
def load_diffusion_artifacts(
    model_dir: Path,
    processor_filename: str,
    model_filename: str,
    condition_network_filename: str | None, # <-- 设为可选
    unet_config: dict,      # <-- 重命名为 unet_config
    diffusion_config: dict,
    cond_net_config: dict | None, # <-- 设为可选
    device: str,
    use_dlpm: bool = False,  # <-- 新增：是否使用DLPM
    dlpm_alpha: float = 1.7  # <-- 新增：DLPM的alpha参数
):
    """
    加载扩散生成所需的所有产出物：
    DataProcessor, ConditionNetwork (如果指定), U-Net, 以及 Diffusion 包装器。
    """
    print("🔄 正在加载扩散模型产出物...")

    # --- 1. 加载 DataProcessor ---
    processor_path = model_dir / processor_filename
    if not processor_path.exists():
        raise FileNotFoundError(f"DataProcessor 文件未找到: {processor_path}")
    try:
        data_processor = joblib.load(processor_path)
        print(f"   ✅ DataProcessor 已从以下位置加载: {processor_path}")
        # 提取类别数量，用于条件网络
        num_countries = data_processor.num_countries if hasattr(data_processor, 'num_countries') and data_processor.num_countries else 1
        num_indices = data_processor.num_indices if hasattr(data_processor, 'num_indices') and data_processor.num_indices else 1
        print(f"      - 检测到 {num_countries} 个国家, {num_indices} 个指数。")
    except Exception as e:
        print(f"   ❌ 加载 DataProcessor 时出错: {e}")
        raise

    # --- 2. 加载条件网络 (如果提供了文件名) ---
    condition_network = None
    cond_net_output_dim = 7 # 默认：如果没有条件网络，U-Net 接收原始 7D 条件
    if condition_network_filename and cond_net_config: # 需要文件名和配置
        cond_net_path = model_dir / condition_network_filename
        if not cond_net_path.exists():
            # 尝试去掉可能的 '_all' 后缀查找 (兼容性)
            cond_net_filename_base = condition_network_filename.replace('_all', '')
            cond_net_path = model_dir / cond_net_filename_base
            if not cond_net_path.exists():
                raise FileNotFoundError(f"ConditionNetwork 文件未找到于 {model_dir / condition_network_filename} 或 {cond_net_path}")

        print(f"   🔄 正在从以下位置加载 EnhancedConditionNetwork: {cond_net_path}")
        try:
            # 使用从 processor 获取的类别数量和配置中的维度来初始化网络
            cond_net_output_dim = cond_net_config.get('output_dim', 128) # 获取输出维度
            condition_network = EnhancedConditionNetwork(
                num_countries=num_countries, # 使用从 processor 获取的数量
                num_indices=num_indices,   # 使用从 processor 获取的数量
                **cond_net_config        # 传递配置中的其他维度参数
            ).to(device)
            # 加载状态字典
            state_dict = torch.load(cond_net_path, map_location=device)
            # 处理可能的 DataParallel 包装
            if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            condition_network.load_state_dict(state_dict)
            condition_network.eval() # 设置为评估模式
            print(f"   ✅ EnhancedConditionNetwork 加载成功。输出维度: {cond_net_output_dim}")
        except Exception as e:
            print(f"   ❌ 加载 EnhancedConditionNetwork 时出错: {e}")
            raise
    else:
        print("   ℹ️ 未指定条件网络或其配置，将不使用 EnhancedConditionNetwork。")
        # U-Net 将接收原始 7D 条件

    # --- 3. 加载 U-Net 模型 ---
    model_path = model_dir / model_filename
    if not model_path.exists():
        # 尝试去掉可能的 '_all' 后缀查找
        model_filename_base = model_filename.replace('_all', '')
        model_path = model_dir / model_filename_base
        if not model_path.exists():
           raise FileNotFoundError(f"U-Net 模型文件未找到于 {model_dir / model_filename} 或 {model_path}")

    print(f"   🔄 正在从以下位置加载 U-Net 模型: {model_path}")
    try:
        unet_model_type = unet_config.get("model_type", "unet")
        unet_model_params = unet_config.get("model_params", {})
        # ** 关键: U-Net 的 cond_dim 必须与条件网络输出匹配 (或为 7) **
        unet_cond_dim = cond_net_output_dim

        if unet_model_type == 'unet':
            model = Unet1D(cond_dim=unet_cond_dim, **unet_model_params).to(device)
        else:
            raise ValueError(f"未知的 U-Net 模型类型: {unet_model_type}")

        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval() # 设置为评估模式
        print(f"   ✅ U-Net ({unet_model_type}) 模型加载成功。期望的 cond_dim: {unet_cond_dim}")
    except Exception as e:
        print(f"   ❌ 加载 U-Net 模型时出错: {e}")
        raise

    # --- 4. 初始化 Diffusion 包装器 ---
    # ** 关键: 将加载的 condition_network 实例 (可能是 None) 传递给扩散模型 **
    if use_dlpm:
        print(f"   🔄 正在初始化 DLPMDiffusion1D (alpha={dlpm_alpha})...")
        try:
            # DLPM特定的配置
            dlpm_config = {
                **diffusion_config,
                'alpha': dlpm_alpha,  # DLPM参数
                'isotropic': True,   # DLPM参数
                'rescale_timesteps': True,  # DLPM参数
                'scale': 'scale_preserving',  # DLPM参数
            }
            diffusion = DLPMDiffusion1D(
                model=model,
                condition_network=condition_network,
                **dlpm_config
            ).to(device)
            print(f"   ✅ DLPMDiffusion1D 初始化成功 {'带有' if condition_network else '不带'} 条件网络。")
        except Exception as e:
            print(f"   ❌ 初始化 DLPMDiffusion1D 时出错: {e}")
            raise
    else:
        print(f"   🔄 正在初始化 GaussianDiffusion1D...")
        try:
            diffusion = GaussianDiffusion1D(
                model=model,                  # 传递加载的 U-Net
                condition_network=condition_network, # 传递加载的条件网络 (或 None)
                **diffusion_config          # 传递扩散过程参数
            ).to(device)
            print(f"   ✅ GaussianDiffusion1D 初始化成功 {'带有' if condition_network else '不带'} 条件网络。")
        except TypeError as e:
             if 'condition_network' in str(e):
                  print("   ❌ 错误: GaussianDiffusion1D 的 __init__ 方法似乎不支持 'condition_network' 参数。")
                  print("       请确保你使用的是接受此参数的 diffusion_with_condition.py 版本。")
             raise
        except Exception as e:
             print(f"   ❌ 初始化 GaussianDiffusion1D 时出错: {e}")
             raise

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
                    cond_input=conditions_batch
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