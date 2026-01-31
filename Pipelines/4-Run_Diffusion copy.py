# Pipelines/4-Run_Diffusion.py
# (使用我们之前整合 EnhancedConditionNetwork 的版本)

# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import os
import time
from datetime import datetime

# --- 关键：路径设置与 sys.path 修改 ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

diffusion_model_dir = project_root / 'Model' / 'Diffusion_Model'
if diffusion_model_dir.exists():
    sys.path.insert(0, str(diffusion_model_dir))
    print(f"[路径修正] 临时将 {diffusion_model_dir} 添加到 sys.path")
else:
    print(f"⚠️ [路径警告] 无法找到目录 {diffusion_model_dir}。")

# --- 标准 Imports ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import joblib
import traceback
import json

# --- Project-Specific Imports ---
try:
    import Project_Path as pp
    from Data.Input_preparation import DataProcessor # 更新后的 DataProcessor
    # !! 使用修改后的 GaussianDiffusion1D 或 DLPMDiffusion1D !!
    from Model.Diffusion_Model.diffusion_with_condition import GaussianDiffusion1D
    from Model.Diffusion_Model.diffusion_dlpm import DLPMDiffusion1D
    from Model.Diffusion_Model.trainer_with_condition import Trainer1D, Dataset1D
    from Model.Diffusion_Model.Unet_with_condition import Unet1D
    # !! 重新导入 EnhancedConditionNetwork !!
    from Model.Diffusion_Model.condition_network import EnhancedConditionNetwork
    import Config.Diffusion_config as DiffusionConfig
except ImportError as e:
    print(f"❌ 错误：导入项目模块失败: {e}")
    sys.exit(1)
# ... (error handling) ...

# === 内嵌报告函数 (恢复 Condition Network 信息) ===
def generate_training_report(report_path: Path, config: dict, data_info: dict, model_info: dict, training_results: dict, artifact_paths: dict):
    """生成 Markdown 训练总结报告"""
    print(f"\n--- 步骤 6: 生成训练总结报告 ---")
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            # ... (报告内容与包含 Condition Network 的版本一致) ...
            f.write(f"# 扩散模型训练报告\n\n")
            f.write(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**训练资产**: {config.get('underlying_asset', 'N/A')}\n\n")

            f.write("## 1. 训练配置\n\n"); f.write("```json\n")
            serializable_config = {k: v for k, v in config.items() if not callable(v)}
            f.write(json.dumps(serializable_config, indent=4, ensure_ascii=False, default=str))
            f.write("\n```\n")

            f.write("\n## 2. 数据信息\n\n");
            f.write(f"- **数据源文件**: `{data_info.get('source_file', 'N/A')}`\n")
            # ... (其他数据信息) ...
            f.write(f"- **处理后样本数**: {data_info.get('num_samples', 'N/A'):,}\n")
            f.write(f"- **条件特征维度**: {data_info.get('condition_dim', 'N/A')}\n")
            f.write(f"- **序列长度**: {data_info.get('sequence_length', 'N/A')}\n")
            f.write(f"- **检测到的国家数**: {data_info.get('num_countries', 'N/A')}\n")
            f.write(f"- **检测到的指数数**: {data_info.get('num_indices', 'N/A')}\n\n")


            f.write("\n## 3. 模型信息\n\n"); f.write(f"- **模型类型**: {model_info.get('type', 'N/A')}\n")
            # ** 恢复条件网络信息 **
            if model_info.get('cond_net_params') is not None:
                f.write(f"- **U-Net 参数量**: {model_info.get('unet_params', 0):,}\n")
                f.write(f"- **条件网络参数量**: {model_info.get('cond_net_params', 0):,}\n")
                f.write(f"- **总可训练参数**: {model_info.get('total_params', 0):,}\n")
            else:
                f.write(f"- **U-Net (总) 参数量**: {model_info.get('unet_params', 0):,}\n")

            f.write("\n## 4. 训练结果\n\n");
            # ... (训练结果) ...
            f.write(f"- **训练总步数**: {training_results.get('steps_run', 'N/A'):,}\n")
            final_loss = training_results.get('final_loss')
            loss_str = f"{final_loss:.6f}" if final_loss is not None and np.isfinite(final_loss) else "N/A"
            f.write(f"- **最终损失值**: {loss_str}\n")
            duration_seconds = training_results.get('duration_seconds')
            duration_str = "N/A"
            if duration_seconds is not None:
                hours, remainder = divmod(int(duration_seconds), 3600); minutes, seconds = divmod(remainder, 60)
                duration_str = f"{hours}小时 {minutes}分钟 {seconds}秒"
            f.write(f"- **训练时长**: {duration_str}\n")

            f.write("\n### 训练损失曲线\n\n"); loss_curve_path_obj = artifact_paths.get('loss_curve')
            if loss_curve_path_obj and loss_curve_path_obj.exists():
                relative_loss_path = loss_curve_path_obj.name
                f.write(f"![训练损失曲线](./{relative_loss_path})\n\n")
            else: f.write("未能生成或找到损失曲线图。\n\n")

            f.write("\n## 5. 产出物路径\n\n");
            f.write(f"- **模型文件 (.pth)**: `{artifact_paths.get('model', 'N/A')}`\n")
            # ** 恢复条件网络路径 **
            if artifact_paths.get('condition_network'):
                 f.write(f"- **条件网络文件 (.pth)**: `{artifact_paths.get('condition_network', 'N/A')}`\n")
            f.write(f"- **数据处理器 (.pkl)**: `{artifact_paths.get('processor', 'N/A')}`\n")
            f.write(f"- **损失曲线图 (.png)**: `{str(artifact_paths.get('loss_curve', 'N/A'))}`\n")
            f.write(f"- **本报告文件 (.md)**: `{str(report_path.resolve())}`\n")

        print(f"   ✅ 训练总结报告已保存到: {report_path}")
    except Exception as e: print(f"   ❌ 生成训练总结报告时出错: {e}"); traceback.print_exc()
# === 结束报告函数 ===



# <<< 关键：保留 if __name__ == '__main__' 保护块 >>>
if __name__ == '__main__':

    training_start_time = time.time()
    main_config = {}
    asset_name = "Unknown"
    # --- 1. 加载配置 ---
    # ... (代码不变) ...
    try:
        main_config = DiffusionConfig.main_config; asset_name = main_config["underlying_asset"]
        print(f"--- 开始为资产 '{asset_name}' 训练扩散模型 ---")
        print(f"--- 使用配置: Config/Diffusion_config.py ---")
    except Exception as e: print(f"❌ 加载配置时出错: {e}"); sys.exit(1)

    # --- 2. 确定输入数据路径 ---
    train_file_path = None
    # ... (代码不变) ...
    try:
        if asset_name.lower() == 'all': train_file_path = pp.Trainning_DATA_DIR / 'trainning_data_merged.csv'
        else: train_file_path = pp.Trainning_DATA_DIR / asset_name / 'train_df.csv'
        print(f"   加载数据: {train_file_path}")
        if not train_file_path.exists(): raise FileNotFoundError(f"未找到: {train_file_path}")
    except Exception as e: print(f"❌ 确定数据路径时出错: {e}"); sys.exit(1)

    # --- 3. 数据处理 ---
    print("\n--- 步骤 1: 处理数据 ---")
    data_processor = None; X_train, y_train, mask_train = None, None, None
    num_countries, num_indices = 1, 1; processed_samples_count = 0
    data_info_for_report = {'source_file': str(train_file_path)}
    try:
        data_processor = DataProcessor(main_config)
        X_train, y_train, mask_train = data_processor.process_all_data(train_file_path) # 只接收3个返回值
        num_countries = data_processor.num_countries if data_processor.num_countries is not None else 1
        num_indices = data_processor.num_indices if data_processor.num_indices is not None else 1
        processed_samples_count = len(X_train)
        data_info_for_report.update({ 'num_samples': processed_samples_count, 'condition_dim': X_train.shape[-1], 'sequence_length': y_train.shape[-1], 'num_countries': num_countries, 'num_indices': num_indices })
        print(f"   ✅ 数据加载和处理成功。有效样本数: {processed_samples_count}")
        # ... (print shapes) ...
    except Exception as e: print(f"   ❌ 数据处理过程中发生意外错误: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 4. 模型设置 ---
    print("\n--- 步骤 2: 设置模型、扩散过程和训练器 ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"   使用设备: {device}")
    model = None; condition_network = None; diffusion = None
    model_info_for_report = {}
    # ** 恢复 use_cond_net **
    use_cond_net = main_config.get('use_enhanced_condition_network', True) # 默认使用
    try:
        cond_net_output_dim = X_train.shape[-1] # 默认7D

        if use_cond_net:
            # ** 恢复条件网络初始化 **
            cond_net_params = main_config.get('cond_net_params', {})
            cond_net_output_dim = cond_net_params.get('output_dim', 128)
            print(f"   初始化 EnhancedConditionNetwork...")
            condition_network = EnhancedConditionNetwork(num_countries=num_countries+5, num_indices=num_indices+10, **cond_net_params).to(device)
            print(f"   ✅ Enhanced Condition Network 初始化成功。输出维度: {cond_net_output_dim}")
            model_info_for_report['cond_net_params'] = sum(p.numel() for p in condition_network.parameters())
        else:
            print("   ℹ️  不使用 EnhancedConditionNetwork (按配置)。")

        if main_config['model_type'] == 'unet':
            # ** U-Net cond_dim 使用条件网络输出维度 **
            unet_cond_dim = cond_net_output_dim
            print(f"   初始化 Unet1D，条件维度 = {unet_cond_dim}")
            unet_params = main_config.get('unet_params', {})
            model = Unet1D(cond_dim=unet_cond_dim, **unet_params).to(device)
            print(f"   ✅ UNet 模型初始化成功。")
            model_info_for_report['type'] = 'U-Net'
            model_info_for_report['unet_params'] = sum(p.numel() for p in model.parameters())
            if use_cond_net: model_info_for_report['total_params'] = model_info_for_report.get('unet_params', 0) + model_info_for_report.get('cond_net_params', 0)
        else: raise ValueError(f"不支持的模型类型: {main_config['model_type']}")

        # 检查是否使用DLPM
        use_dlpm = main_config.get('use_dlpm', False)
        if use_dlpm:
            print(f"   初始化 DLPMDiffusion1D (alpha={main_config.get('dlpm_alpha', 1.7)})...")
            diffusion = DLPMDiffusion1D(
                model=model,
                condition_network=condition_network,
                seq_length=main_config.get('seq_length', 252),
                timesteps=main_config['timesteps'],
                objective=main_config.get('objective', 'pred_v'),
                auto_normalize=main_config.get('auto_normalize', False),
                alpha=main_config.get('dlpm_alpha', 1.7),
                warmup_ratio=main_config.get('warmup_ratio', 0.20),   # 预热比例
                train_num_steps=main_config['train_num_steps'],       # 总步数
                ema_beta=main_config.get('ema_decay', 0.99),
                isotropic=main_config.get('dlpm_isotropic', True),
                rescale_timesteps=main_config.get('dlpm_rescale_timesteps', True),
                scale=main_config.get('dlpm_scale', 'scale_preserving'),
            ).to(device)
            print(f"   ✅ DLPM扩散过程初始化成功 {'(已集成条件网络)' if use_cond_net else ''}。")
        else:
            print(f"   初始化 GaussianDiffusion1D...")
            # ** 将 condition_network 传递给 GaussianDiffusion1D **
            diffusion = GaussianDiffusion1D(
                model=model,
                condition_network=condition_network, # <-- 传递实例
                seq_length=main_config.get('seq_length', 252),
                timesteps=main_config['timesteps'],
                objective=main_config.get('objective', 'pred_v'),
                auto_normalize=main_config.get('auto_normalize', False),
                warmup_ratio=main_config.get('warmup_ratio', 0.0),
                train_num_steps=main_config['train_num_steps'],
                # ema_beta=main_config.get('ema_decay', 0.99) # 传递 ema_beta 给 diffusion
            ).to(device)
            print(f"   ✅ 高斯扩散过程初始化成功 {'(已集成条件网络)' if use_cond_net else ''}。")

    except Exception as e: print(f"   ❌ 初始化模型或扩散过程时出错: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 5. 数据集和训练器 ---
    trainer = None
    all_trainable_params = []
    try:
        print("\n--- 步骤 3: 创建数据集和训练器 ---")
        dataset = Dataset1D(
            y_train.clone().detach().float(),
            X_train.clone().detach().float(), # 传递 7D 条件
            mask_train.clone().detach().float()
        )
        print(f"   ✅ Dataset1D 创建成功，包含 {len(dataset)} 个样本。")

        trainer_params = main_config.get('trainer_params', {})
        trainer = Trainer1D(
            diffusion_model=diffusion,
            dataset=dataset,
            train_num_steps=main_config['train_num_steps'],
            train_batch_size=trainer_params.get('train_batch_size', main_config.get('train_batch_size', 64)),
            train_lr=trainer_params.get('train_lr', main_config.get('train_lr', 1e-6)),
            gradient_accumulate_every=trainer_params.get('gradient_accumulate_every', main_config.get('gradient_accumulate_every', 1)),
            ema_decay=trainer_params.get('ema_decay', main_config.get('ema_decay', 0.995)),
            amp=trainer_params.get('amp', main_config.get('amp', True)),
            save_and_sample_every=999999999, # 禁用内部保存
        )
        print(f"   ✅ Trainer 初始化成功。")
    except Exception as e: print(f"   ❌ 创建数据集或训练器时出错: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 6. 训练 ---
    print("\n--- 步骤 4: 开始训练 ---")
    training_results_for_report = {'steps_run': 0, 'final_loss': None}
    try:
        if trainer is None: raise RuntimeError("训练器未能成功初始化。")
        trainer.train()
        print("   ✅ 训练循环完成。")
        training_results_for_report['steps_run'] = trainer.step
        training_results_for_report['final_loss'] = trainer.loss_history[-1] if trainer.loss_history else None
    except Exception as e:
        print(f"   ❌ 训练过程中发生错误: {e}")
        traceback.print_exc()
        if trainer and hasattr(trainer, 'step'): training_results_for_report['steps_run'] = trainer.step

    # --- 7. 保存产出物 ---
    print("\n--- 步骤 5: 保存产出物 ---")
    artifact_paths_for_report = {}
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        # ** 路径定义与之前一致 **
        model_artifact_dir = pp.Model_Results_DIR / asset_name
        model_artifact_dir.mkdir(parents=True, exist_ok=True)
        model_suffix = "_all" if asset_name.lower() == 'all' else ""
        model_filename = f"{main_config['model_type']}_conditional_model{model_suffix}.pth"
        cond_net_filename = f"condition_network{model_suffix}.pth" # 恢复
        processor_filename = f"data_processor{model_suffix}.pkl"
        model_save_path = model_artifact_dir / model_filename
        cond_net_save_path = model_artifact_dir / cond_net_filename # 恢复
        processor_save_path = model_artifact_dir / processor_filename
        report_save_dir = pp.Results_DIR / "training_report" / asset_name / f"{timestamp_str}_{main_config['model_type']}_report"
        report_save_dir.mkdir(parents=True, exist_ok=True)
        loss_filename = f"{main_config['model_type']}_loss_curve{model_suffix}.png"
        loss_graph_path = report_save_dir / loss_filename
        report_save_path = report_save_dir / f"training_summary{model_suffix}.md"

        artifact_paths_for_report['model'] = str(model_save_path.resolve())
        # ** 恢复条件网络路径 **
        if use_cond_net: artifact_paths_for_report['condition_network'] = str(cond_net_save_path.resolve())
        artifact_paths_for_report['processor'] = str(processor_save_path.resolve())
        artifact_paths_for_report['loss_curve'] = loss_graph_path

        # 保存 U-Net
        if model is not None:
            try: torch.save(model.state_dict(), model_save_path); print(f"   ✅ U-Net 模型已保存: {model_save_path}")
            except Exception as e: print(f"   ⚠️ 保存 U-Net 模型时出错: {e}")
        # ** 恢复保存条件网络 **
        if use_cond_net and condition_network is not None:
            try: torch.save(condition_network.state_dict(), cond_net_save_path); print(f"   ✅ 条件网络已保存: {cond_net_save_path}")
            except Exception as e: print(f"   ⚠️ 保存条件网络时出错: {e}")
        # 保存 Processor
        if data_processor is not None:
            try: joblib.dump(data_processor, processor_save_path); print(f"   ✅ 数据处理器已保存: {processor_save_path}")
            except Exception as e: print(f"   ⚠️ 保存数据处理器时出错: {e}")
        # 保存 Loss Curve
        if trainer is not None and hasattr(trainer, 'loss_history') and trainer.loss_history:
            try:
                # ... (字体配置) ...
                try: plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
                except: print("   ⚠️ 未能配置中文字体。")
                smoothing_window = main_config.get('loss_smoothing_window', 200)
                # ... (绘图代码) ...
                loss_series = pd.Series(trainer.loss_history); moving_avg = loss_series.rolling(window=smoothing_window, min_periods=1).mean()
                plt.figure(figsize=(15, 7)); plt.plot(trainer.loss_history, label='训练损失 (原始)', color='lightblue', alpha=0.6); plt.plot(moving_avg, label=f'移动平均 (窗口={smoothing_window})', color='red', linewidth=2)
                plt.title(f"训练损失曲线 - {main_config['model_type'].upper()} ({asset_name})"); plt.xlabel("训练步数"); plt.ylabel("损失 (对数刻度)"); plt.grid(True, alpha=0.5); plt.legend(); plt.yscale('log'); plt.tight_layout()
                plt.savefig(loss_graph_path); plt.close()
                print(f"   ✅ 训练损失曲线已保存: {loss_graph_path}")
            except Exception as e: print(f"   ⚠️ 保存损失曲线图时出错: {e}")
        else: print("   ⚠️ 训练器或损失历史不存在。")

    except Exception as e: print(f"   ❌ 保存产出物时发生意外错误: {e}"); traceback.print_exc()

    # --- 8. 生成总结报告 ---
    training_end_time = time.time()
    training_results_for_report['duration_seconds'] = training_end_time - training_start_time

    generate_training_report(
        report_path=report_save_path,
        config=main_config,
        data_info=data_info_for_report,
        model_info=model_info_for_report, # 包含条件网络信息 (如果使用)
        training_results=training_results_for_report,
        artifact_paths=artifact_paths_for_report # 包含条件网络路径 (如果使用)
    )

    print(f"\n--- 资产 '{asset_name}' 的训练流程结束 ---")