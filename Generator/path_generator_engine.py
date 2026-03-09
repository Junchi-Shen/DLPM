# path_generator_engine.py
# 
# 这是一个通用的路径生成引擎。
# 它被设计为 "被导入"，而不是 "被执行"。

import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import gc
import time
import traceback
import logging
import json


# --- 导入自定义模块 ---
import Generator.path_simulators as ps

# --- 路径导入 ---
try:
    current_file_dir = Path(__file__).parent.resolve()
    project_root = current_file_dir.parent
    sys.path.append(str(project_root))
    # 动态导入 Project_Path 中的所有路径
    import Project_Path as pp
    from .path_simulators import load_diffusion_artifacts, simulate_gbm, simulate_garch,run_diffusion_mega_batch
    # 需要 DataProcessor 类定义来加载 .pkl 文件
    from Data.Input_preparation import DataProcessor
except (ImportError, NameError):
    print("❌ 严重错误: 未找到 Project_Path.py。引擎无法工作。")
    sys.exit(1)


class PathGeneratorEngine: # 重命名类以示清晰
    """
    通用、配置驱动的路径生成引擎 (已更新以集成条件网络和加载的处理器)。
    """
    def __init__(self, asset_name: str, job_spec: dict, device: str,num_paths_override: int | None = None):
        self.asset_name = asset_name
        self.spec = job_spec
        self.device = device
        self.num_paths_override = num_paths_override
        self.job_type = self.spec['type']
        # 从 spec 获取作业名，如果没有则提供默认值
        self.job_name = self.spec.get('job_name', f"Unnamed_{self.job_type}_Job")

        print(f"🚀 初始化路径生成引擎 (作业: '{self.job_name}', 类型: {self.job_type})...")
        self.actual_num_paths = self._determine_num_paths()
        print(f"   将为每个条件生成 {self.actual_num_paths:,} 条路径。")
        
        # 占位符
        self.val_df = None
        self.conditions_df = None # 将从 val_df 中选取
        self.garch_params = None
        self.diffusion_model = None # 将持有 GaussianDiffusion1D 实例
        self.data_processor = None # 将持有加载的 DataProcessor 实例

        self._load_validation_data() # 加载所有类型都需要的 val_df

        # --- !! 已修改：如果需要，加载 GARCH 参数 !! ---
        if self.job_type == 'mc' and 'garch_params_filename' in self.spec:
            self._load_garch_params()
        # ---

        # --- !! 已修改：如果需要，加载 Diffusion 产出物 !! ---
        if self.job_type == 'diffusion':
            self._load_diffusion_artifacts()
        # ---

        print("✅ 引擎初始化完成。")
    def _determine_num_paths(self) -> int:
        """根据覆盖值或配置确定最终要生成的路径数"""
        if self.num_paths_override is not None and self.num_paths_override > 0:
            print(f"   收到启动器覆盖参数：num_paths = {self.num_paths_override}")
            # 更新 spec 内部的值，以便后续函数使用正确的值
            if self.job_type == 'mc':
                 self.spec['params']['n_simulations'] = self.num_paths_override
            elif self.job_type == 'diffusion':
                 self.spec['generation_params']['num_paths_to_generate'] = self.num_paths_override
            return self.num_paths_override
        else:
            # 从配置中获取默认值
            default_paths = 0
            if self.job_type == 'mc':
                default_paths = self.spec.get('params', {}).get('n_simulations', 1024) # 默认 1024
            elif self.job_type == 'diffusion':
                default_paths = self.spec.get('generation_params', {}).get('num_paths_to_generate', 1024) # 默认 1024
            print(f"   使用配置中的默认路径数：num_paths = {default_paths}")
            return default_paths
        
    def _load_validation_data(self):
        """
        加载验证数据 (val_df) 作为生成条件。
        (已修改：从中央 merge 文件加载并按 asset_name 过滤)
        """
        print("🔄 正在加载 *中央* 验证数据 (testing_data_merged.csv)...")
        val_dir_key = "Testing_DATA_DIR"
        val_base_dir = getattr(pp, val_dir_key, None)
        if val_base_dir is None: raise AttributeError(f"Project_Path.py 缺少 '{val_dir_key}'")

        # --- 1. 加载中央 merge 文件 ---
        # 假设中央文件名 (如果你的文件名不同，请在此处修改)
        central_file_name = 'testing_data_merged.csv'
        val_file_path = val_base_dir / central_file_name
        
        # 检查文件是否存在
        if not val_file_path.exists():
            # 备用：万一文件名是 val_df.csv
            val_file_path_fallback = val_base_dir / 'val_df.csv'
            if val_file_path_fallback.exists():
                val_file_path = val_file_path_fallback
            else:
                 raise FileNotFoundError(f"中央 merge 文件未找到: {val_file_path} 或 {val_file_path_fallback}")

        try:
            full_val_df = pd.read_csv(val_file_path)
            print(f"✅ 中央验证文件加载成功: {val_file_path}")

            # --- 2. 关键：过滤子集 ---
            
            # 确定要过滤哪个资产
            asset_to_filter = self.asset_name
            if self.asset_name.lower() == 'all':
                # 'all' 模型也需要一个 *具体* 资产的条件来生成
                asset_to_filter = self.spec.get('representative_asset_for_val', 'SP500') 
                print(f"   ⚠️ 'all' 模型作业将使用代表性资产 '{asset_to_filter}' 的条件。")

            # !! 关键假设 !! 
            # 假设用于过滤的列名是 'asset_name'
            # 如果你的列名不同 (例如 'index')，请在此处修改
            filter_column_name = 'asset_underlying' 
            
            print(f"   🔄 正在从中央文件中过滤 '{asset_to_filter}' 的子集 (基于列 '{filter_column_name}')...")

            # 检查列是否存在
            if filter_column_name not in full_val_df.columns:
                 raise KeyError(f"中央 merge 文件 '{val_file_path}' 中缺少用于过滤的列: '{filter_column_name}'")
                 
            self.conditions_df = full_val_df[full_val_df[filter_column_name] == asset_to_filter].copy()
            
            if self.conditions_df.empty:
                raise ValueError(f"在 '{val_file_path}' 中找不到资产 '{asset_to_filter}' 的任何数据。")

            # 在此引擎中，val_df 和 conditions_df 相同
            self.val_df = self.conditions_df 
            print(f"✅ 验证子集加载成功。将使用 {len(self.conditions_df)} 个 '{asset_to_filter}' 市场条件。")
        
        except Exception as e:
            print(f"❌ 加载或处理中央 merge 文件时出错: {e}")
            raise
    # --- !! 新方法：加载 GARCH 参数 !! ---
    def _load_garch_params(self):
        """从 JSON 加载预先拟合的 GARCH 参数。"""
        print("🔄 正在加载预拟合的 GARCH 参数...")
        filename = self.spec['garch_params_filename']
        # 参数通常保存在 Model/Garch_Model/<asset>/filename
        params_root_key = self.spec.get('garch_params_dir_key', 'Model_Results_DIR') # 默认 Model_Results_DIR
        params_subfolder = self.spec.get('garch_params_subfolder', 'Garch_Fit_Results') # 默认 Garch_Fit_Results
        params_root_dir = getattr(pp, params_root_key, None)
        if params_root_dir is None:
            raise AttributeError(f"Project_Path.py 缺少 '{params_root_key}'")
        params_base_dir = params_root_dir / params_subfolder

        params_path = None
        # GARCH 参数通常是特定于资产的，即使在 'all' 训练上下文中也可能如此
        # 假设参数文件总是按资产存储
        asset_for_garch_params = self.asset_name
        if self.asset_name.lower() == 'all':
             # 如果是 'all' 作业，需要决定加载哪个资产的 GARCH 参数
             # 可能需要从配置指定，或使用默认值
             asset_for_garch_params = self.spec.get('representative_asset_for_garch', 'SP500') # 示例
             print(f"   ⚠️ 使用代表性资产 '{asset_for_garch_params}' 的 GARCH 参数为 'all' 作业。")

        params_path = params_base_dir / asset_for_garch_params / filename

        if not params_path.exists():
            raise FileNotFoundError(
                f"GARCH 参数文件未找到: {params_path}。"
                f"请先运行 GARCH 拟合脚本。"
            )
        try:
            with open(params_path, 'r') as f:
                # 加载时将 null 转换回 np.nan
                self.garch_params = json.load(f, object_hook=lambda d: {k: (np.nan if v is None else v) for k, v in d.items()})
            print(f"✅ 从以下位置加载 GARCH 参数成功: {params_path}")
            if not all(k in self.garch_params for k in ['omega', 'alpha', 'beta']):
                raise ValueError("加载的 GARCH 参数缺少必需键。")
        except Exception as e:
            print(f"❌ 加载或解析 GARCH 参数时出错: {e}")
            raise RuntimeError("加载 GARCH 参数失败。") from e
    # ---

    # --- !! 已修改：加载 Diffusion 产出物 !! ---
    def _load_diffusion_artifacts(self):
        """加载 DataProcessor, CondNet, U-Net, 和 Diffusion 包装器。"""
        print("🔄 正在加载扩散模型产出物...")
        loader_params = self.spec.get('model_loader_params')
        if not loader_params: raise ValueError("配置中缺少 'model_loader_params'。")

        # 确定模型产出物目录 (通常在 Results/Model_Results)
        model_dir_root_key = loader_params.get('model_dir_root_key', 'Model_Results_DIR') # 默认键名
        model_base_dir = getattr(pp, model_dir_root_key, None)
        if model_base_dir is None: raise AttributeError(f"Project_Path.py 缺少 '{model_dir_root_key}'")

        # 处理 'all' vs 特定资产路径
        # model_folder_name = self.spec.get('model_source_folder', self.asset_name)
        # model_dir = model_base_dir / model_folder_name

        model_folder_name = self.spec.get('model_source_folder', self.asset_name)
        model_type_subfolder = self.spec.get('model_type_subfolder', None)
        if model_type_subfolder:
            model_dir = model_base_dir / model_type_subfolder / model_folder_name
        else:
            model_dir = model_base_dir / model_folder_name

        if not model_dir.exists():
            raise FileNotFoundError(f"模型产出物目录未找到: {model_dir}。请确保已为 '{self.asset_name}' 运行训练。")

        try:
            # 使用 path_simulators 中的更新后的加载函数
            # 检查是否使用DLPM
            use_dlpm = self.spec.get('use_dlpm', False)
            dlpm_alpha = self.spec.get('dlpm_alpha', 1.7)
            
            self.diffusion_model, self.data_processor = load_diffusion_artifacts(
                model_dir=model_dir,
                processor_filename=loader_params['processor_filename'],
                model_filename=loader_params['model_filename'],
                # 使用 .get 以允许条件网络是可选的
                condition_network_filename=loader_params.get('condition_network_filename'),
                unet_config=self.spec['unet_config'],       # 传递 unet 配置
                diffusion_config=self.spec['diffusion_config'],  # 传递 diffusion 配置
                cond_net_config=self.spec.get('cond_net_config'), # 传递条件网络配置 (可能是 None)
                device=self.device,
                use_dlpm=use_dlpm,  # 传递DLPM标志
                dlpm_alpha=dlpm_alpha  # 传递DLPM alpha参数
            )
            print("✅ 扩散模型产出物加载成功。")
            if not isinstance(self.data_processor, DataProcessor):
                 print("   ⚠️ 警告: 加载的 data_processor 类型不是预期的 DataProcessor。")

        except Exception as e:
             print(f"❌ 从 {model_dir} 加载扩散产出物时出错: {e}")
             traceback.print_exc()
             raise RuntimeError("加载扩散产出物失败。") from e
    # ---

    def run(self):
        """执行主生成逻辑。"""
        print(f"\n🏁 开始生成作业: {self.job_name} ...")
        start_time = time.time()
        
        try:
            if self.job_type == 'mc':
                self._run_mc_generation()
            elif self.job_type == 'diffusion':
                self._run_diffusion_generation()
            else:
                raise ValueError(f"未知的作业类型: {self.job_type}")
        except Exception as e:
             print(f"❌ 作业 '{self.job_name}' 执行过程中失败: {e}")
             traceback.print_exc()
             # 即使失败也打印结束信息
        
        end_time = time.time(); duration = end_time - start_time
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        print(f"🏁 作业 {self.job_name} 结束。总用时: {duration_str}。")

    # --- !! 已修改：MC 生成逻辑 !! ---
    def _run_mc_generation(self):
        """使用加载的参数执行迭代式 MC (GBM, GARCH) 生成。"""
        simulator_func = self.spec['simulator_function']
        sim_params = self.spec.get('params', {})
        n_steps_total = sim_params.get('n_steps_total', 252)

        all_paths_list = []
        all_sigmas_list = []
        all_masks_list = []

        is_garch_job = 'garch_params_filename' in self.spec
        if is_garch_job and self.garch_params is None:
             raise RuntimeError("GARCH 作业已指定，但参数未加载。")

        print(f"   模拟 {len(self.conditions_df)} 个条件...")
        # 确保 T_days 是整数
        self.conditions_df['actual_trading_days'] = pd.to_numeric(self.conditions_df['actual_trading_days'], errors='coerce').fillna(0).astype(int)

        for _, row in tqdm(self.conditions_df.iterrows(), total=len(self.conditions_df), desc="模拟 MC 路径"):
            # 检查是否有 NaN 条件
            if row.isnull().any():
                 logging.warning(f"跳过包含 NaN 条件的行 (索引 {_})。")
                 continue

            params_for_sim = {
                "S0": row['start_price'], "r": row['risk_free_rate'],
                "initial_vol_ann": row['volatility'], "sigma": row['volatility'],
                "T_days": row['actual_trading_days'], # 确保是整数
                **sim_params
            }
            if is_garch_job:
                params_for_sim['garch_params'] = self.garch_params
            # --- !! 在这里添加清理逻辑 !! ---
            if simulator_func.__name__ == 'simulate_gbm':
                params_for_sim.pop('initial_vol_ann', None) # 移除 GARCH 专用参数
            elif simulator_func.__name__ == 'simulate_garch':
                params_for_sim.pop('sigma', None) # 移除 GBM 专用参数
        # --- !! 添加结束 !! ---
            try:
                result = simulator_func(**params_for_sim)
                if isinstance(result, tuple): # GARCH
                    paths_out, sigma2_out = result
                    all_paths_list.append(paths_out)
                    if self.spec.get('save_extra_outputs', False):
                        n_sim, _, current_len = sigma2_out.shape
                        padded_arr = np.full((n_sim, 1, n_steps_total), np.nan, dtype=np.float64)
                        L = min(current_len, n_steps_total); padded_arr[:, :, :L] = sigma2_out[:, :, :L]
                        all_sigmas_list.append(padded_arr)
                        all_masks_list.append(~np.isnan(paths_out))
                else: # GBM
                    paths_out = result
                    all_paths_list.append(paths_out)
            except Exception as sim_e:
                 logging.error(f"模拟条件 {_} 时失败: {sim_e}")
                 # 可以选择填充 NaN 或跳过
                 n_sim = sim_params.get("n_simulations", 1) # 获取 n_sim
                 all_paths_list.append(np.full((n_sim, 1, n_steps_total), np.nan)) # 填充 NaN 路径
                 if is_garch_job and self.spec.get('save_extra_outputs', False):
                      all_sigmas_list.append(np.full((n_sim, 1, n_steps_total), np.nan))
                      all_masks_list.append(np.full((n_sim, 1, n_steps_total), False))


        # 整合结果 - 沿条件维度堆叠 (axis=0)
        if not all_paths_list: print("⚠️ 没有成功模拟任何路径。"); return
        final_paths_array = np.concatenate(all_paths_list, axis=0)
        extra_arrays_to_save = {}
        if self.spec.get('save_extra_outputs', False):
            if all_sigmas_list: extra_arrays_to_save['sigma2'] = np.concatenate(all_sigmas_list, axis=0)
            if all_masks_list: extra_arrays_to_save['mask'] = np.concatenate(all_masks_list, axis=0)

        self._save_results(final_paths_array, **extra_arrays_to_save)
    # ---

    # --- !! 已修改：Diffusion 生成逻辑 !! ---
    def _run_diffusion_generation(self):
        """执行矢量化 Diffusion (UNet) 生成。"""
        if self.diffusion_model is None or self.data_processor is None:
            raise RuntimeError("扩散模型产出物未加载。")

        # --- 1. 使用加载的 DataProcessor 准备条件 ---
        print("🔄 正在使用加载的 DataProcessor 处理验证集条件...")
        X_test = None
        try:
            # ** 关键：调用 create_condition_tensors (fit_scaler=False) **
            # 需要先运行 process_price_data 来准备必要的列 (如 S_0)
            df_processed_val = self.data_processor.process_price_data(self.conditions_df)
            df_processed_val = df_processed_val.dropna(subset=['S_0', 'price_series']) # 移除无效行
            if df_processed_val.empty: raise ValueError("验证集中没有有效的条件。")

            # 使用已拟合的 price_scaler (来自训练) 进行转换
            condition_dict = self.data_processor.create_condition_tensors(df_processed_val, fit_scaler=False)
            X_test_np = condition_dict['conditions'] # 7D numpy array
            X_test = torch.FloatTensor(X_test_np).to(self.device) # 转为 Tensor 并移到设备
            print(f"✅ 条件准备完毕，用于生成。形状: {X_test.shape}")

        except AttributeError as ae:
             # 捕获 DataProcessor 没有 scaler 的错误 (如果加载失败或未训练)
             if 'price_scaler' in str(ae) and 'n_features' in str(ae):
                  print("❌ 错误: 加载的 DataProcessor 中的 price_scaler 似乎未拟合。")
                  print("   请确保用于训练的 data_processor.pkl 文件已正确保存并包含拟合的标准化器。")
                  raise RuntimeError("DataProcessor scaler 未拟合。") from ae
             else: raise # 重新抛出其他 AttributeError
        except Exception as e:
             print(f"❌ 使用加载的 DataProcessor 准备条件时出错: {e}")
             traceback.print_exc()
             raise RuntimeError("为扩散模型生成准备条件失败。") from e

        # --- 2. 获取运行器函数 ---
        runner_func_spec = self.spec['generation_params'].get('runner_function')
        if runner_func_spec is None: raise ValueError("配置中缺少 'runner_function'。")

        runner_func = None
        if callable(runner_func_spec):
            runner_func = runner_func_spec
        elif isinstance(runner_func_spec, str): # 如果存储的是函数名字符串
             if hasattr(ps, runner_func_spec):
                 runner_func = getattr(ps, runner_func_spec)
             else:
                 raise ValueError(f"运行器函数 '{runner_func_spec}' 在 path_simulators 中未找到。")
        else: raise TypeError("'runner_function' 必须是可调用对象或函数名字符串。")

        # --- 3. 运行生成 ---
        # 运行器函数接收 条件(X_test), 扩散模型实例, 生成参数, 设备
        all_paths_list = runner_func(
            conditions=X_test,               # 传递处理后的 7D 条件
            diffusion=self.diffusion_model,  # 传递加载的 GaussianDiffusion1D 实例
            gen_params=self.spec['generation_params'],
            device=self.device
        ) # 返回每个条件对应的 numpy 数组列表

        # --- 4. 整合并保存 ---
        if not all_paths_list:
             print("⚠️ 生成器返回了空的路径列表。无文件保存。")
             return

        try:
            # 沿新的维度 (条件维度, axis=0) 堆叠列表中的数组
            final_paths_array = np.stack(all_paths_list, axis=0)
            # 预期的形状应该是 [num_conditions, num_paths, channels, seq_len]
            # 例如 [827, 4096, 1, 252]
            self._save_results(final_paths_array)
        except ValueError as e:
             print(f"❌ 合并生成的路径时出错: {e}。请检查每个条件的生成数组形状是否一致。")
             # 可以尝试保存为一个 .npz 文件或其他格式作为回退
             # fallback_path = (self.report_dir / f"{self.job_name_safe}_generated_paths.npz").with_suffix('.npz')
             # np.savez_compressed(fallback_path, *all_paths_list)
             # print(f"   ⚠️ 已将路径作为单独数组保存到 .npz 文件: {fallback_path}")
             raise RuntimeError("合并生成的扩散路径失败。") from e
    # ---


    # --- !! 已修改：保存结果逻辑 !! ---
    def _save_results(self, paths_array, **extra_arrays):
        """通用保存逻辑 - 使用 Path_Generator_Results_DIR 并动态生成文件名"""
        print("\n💾 正在保存结果...")

        # 1. 确定输出目录
        output_dir_key = self.spec.get('output_dir', 'Path_Generator_Results_DIR')
        base_output_dir = getattr(pp, output_dir_key, None)
        if base_output_dir is None: raise AttributeError(f"PP 缺少 '{output_dir_key}'")
        output_dir = base_output_dir / self.asset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 2. 动态构建主文件名 ---
        filename_base = self.spec.get('output_filename_base', f"{self.job_type}_generated") # 从配置获取基础名或默认
        # 使用实际生成的路径数 (self.actual_num_paths)
        output_filename = f"{filename_base}_{self.actual_num_paths}_samples.npy"
        output_path = output_dir / output_filename
        # ---

        try:
            np.save(output_path, paths_array)
            print(f"   ✅ 主路径文件保存成功 (形状: {paths_array.shape})")
            print(f"      -> {output_path}")
            try: print(f"      -> 文件大小: {output_path.stat().st_size / 1024**2:.2f} MB")
            except OSError: pass
        except Exception as e: print(f"   ❌ 保存主路径文件时出错: {e}")

        # 3. 保存额外文件 (文件名也动态生成)
        if extra_arrays:
            # 使用不含数量的基础名来构建额外文件名
            extra_base = f"{filename_base}"
            if self.job_type == 'mc' and 'garch' in self.job_name.lower():
                 extra_base += "_fitted" # 保持 GARCH 文件名一致性

            for key, array_data in extra_arrays.items():
                extra_filename = f"{extra_base}_{key}_{self.actual_num_paths}_samples.npy" # 加入数量
                extra_output_path = output_dir / extra_filename
                try:
                    np.save(extra_output_path, array_data)
                    print(f"   ✅ 额外文件 '{key}' 保存成功 (形状: {array_data.shape})")
                    print(f"      -> {extra_output_path}")
                except Exception as e: print(f"   ❌ 保存额外文件 '{key}' 时出错: {e}")
        # 4. 清理内存
        del paths_array, extra_arrays; gc.collect()
        if self.device.startswith('cuda'):
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                 print(f"   GPU 显存占用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
                 torch.cuda.empty_cache()
                 print(f"   GPU 显存 (缓存清理后): {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            elif self.device == 'cuda': print(f"   设备是 CUDA 但 torch.cuda 不可用/未初始化。")
    # ---