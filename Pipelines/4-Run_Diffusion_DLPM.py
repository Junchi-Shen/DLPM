# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import joblib
import traceback
import json
import matplotlib.pyplot as plt

# --- 1. 路径设置与 sys.path 注入 ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

# 自动定位并注入模型目录
diffusion_model_dir = project_root / 'Model' / 'Diffusion_Model_DLPM'
if diffusion_model_dir.exists():
    sys.path.insert(0, str(diffusion_model_dir))
    print(f"[路径修正] 已注入模型目录: {diffusion_model_dir}")

# --- 2. 导入项目模块 ---
try:
    import Project_Path as pp
    from Data.Input_preparation import DataProcessor
    from Model.Diffusion_Model.diffusion_with_condition import GaussianDiffusion1D
    from Model.Diffusion_Model.diffusion_dlpm import DLPMDiffusion1D
    from Model.Diffusion_Model.trainer_with_condition import Trainer1D, Dataset1D
    from Model.Diffusion_Model.Unet_with_condition import Unet1D
    from Model.Diffusion_Model.condition_network import EnhancedConditionNetwork
    import Config.Diffusion_config_DLPM as DiffusionDLPMConfig
except ImportError as e:
    print(f"❌ 错误：导入项目模块失败: {e}")
    sys.exit(1)

# --- 3. 报告生成函数 ---
def generate_training_report(report_path: Path, config: dict, data_info: dict, model_info: dict, training_results: dict, artifact_paths: dict):
    """生成 Markdown 训练总结报告"""
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 扩散模型训练报告 (多维度自适应版)\n\n")
            f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **训练模式**: {config.get('underlying_asset', 'N/A')} (国家过滤: {config.get('country', 'all')})\n\n")
            
            f.write("## 数据统计\n")
            f.write(f"- 最终样本数: {data_info.get('num_samples', 0):,}\n")
            f.write(f"- 检测到国家/指数: {data_info.get('num_countries')}/{data_info.get('num_indices')}\n\n")

            f.write("## 训练结果\n")
            f.write(f"- 最终 Loss: {training_results.get('final_loss', 'N/A'):.6f}\n")
            f.write(f"- 训练耗时: {training_results.get('duration_seconds', 0)/60:.2f} 分钟\n\n")
            
            if artifact_paths.get('loss_curve'):
                f.write(f"![损失曲线](./{Path(artifact_paths['loss_curve']).name})\n")
    except Exception as e:
        print(f"⚠️ 报告生成失败: {e}")

# --- 4. 主执行流程 ---
if __name__ == '__main__':
    training_start_time = time.time()
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # [1] 配置加载
    main_config = DiffusionDLPMConfig.main_config
    print(f"The alpha of this model is {main_config['dlpm_alpha']}") 
    asset_name = main_config["underlying_asset"]     # 'all' 或 'CSI1000'
    target_countries = main_config.get("country", "all") # 'CN', 'US', 'all' 或列表
    if isinstance(target_countries, str): target_countries = [target_countries]

    print(f"\n" + "="*50)
    print(f"🏁 DLPM扩散模型训练启动器")
    print(f"资产模式: {asset_name} | 国家限制: {target_countries}")
    print("="*50)

    # [2] 动态路由与数据过滤 (核心诉求实现)
    print("\n--- 步骤 1: 数据逻辑路由 ---")
    temp_csv_path = None
    try:
        # A. 确定基础源
        if asset_name.lower() == 'all':
            base_path = pp.Trainning_DATA_DIR / 'trainning_data_merged.csv'
            print(f"📂 路由：使用中央合并大表")
        else:
            base_path = pp.Trainning_DATA_DIR / asset_name / 'train_df.csv'
            print(f"📂 路由：使用特定资产目录 -> {asset_name}")

        if not base_path.exists():
            raise FileNotFoundError(f"无法定位数据源: {base_path}")

        # B. 内存过滤逻辑
        df = pd.read_csv(base_path)
        # 兼容性检测：国家列名可能为 'country' 或 'country_code'
        c_col = next((c for c in ['country_code', 'country'] if c in df.columns), None)
        
        if c_col and "all" not in [x.lower() for x in target_countries]:
            df = df[df[c_col].isin(target_countries)]
            print(f"🎯 过滤：已保留国家 {target_countries}, 剩余行数: {len(df)}")
        
        if df.empty:
            raise ValueError("❌ 过滤后无有效数据，请检查配置中的 country 参数。")

        # C. 生成动态临时文件 (解耦 DataProcessor)
        temp_csv_path = pp.Trainning_DATA_DIR / f"temp_run_{asset_name}_{timestamp_str}.csv"
        df.to_csv(temp_csv_path, index=False)
        
    except Exception as e:
        print(f"❌ 数据路由失败: {e}"); sys.exit(1)

    # [3] 特征提取
    print("\n--- 步骤 2: 特征提取与归一化 ---")
    try:
        data_processor = DataProcessor(main_config)
        X_train, y_train, mask_train = data_processor.process_all_data(temp_csv_path)
        for name, tensor in [("条件特征(X)", X_train), ("目标序列(y)", y_train), ("有效性Mask", mask_train)]:
            if not torch.isfinite(tensor).all():
                num_nan = torch.isnan(tensor).sum().item()
                num_inf = torch.isinf(tensor).sum().item()
                print(f"❌ 数据异常：{name} 包含 {num_nan} 个 NaN, {num_inf} 个 Inf")
        
                # 定位具体的样本 ID (假设 X_train 第一维是 batch)
                error_indices = torch.where(~torch.isfinite(tensor).any(dim=-1).any(dim=-1))[0]
                print(f"🚨 出错样本索引（前5个）: {error_indices[:5].tolist()}")
                raise ValueError(f"数据源 {name} 存在数值污染，请检查 DataProcessor 逻辑。")

            print("✅ 数据源全量检查通过：Finite check passed.")
        
        data_info = {
            'source_file': str(base_path),
            'num_samples': len(X_train),
            'condition_dim': X_train.shape[-1],
            'sequence_length': y_train.shape[-1],
            'num_countries': data_processor.num_countries,
            'num_indices': data_processor.num_indices
        }
    except Exception as e:
        print(f"❌ 数据处理失败: {e}"); traceback.print_exc(); sys.exit(1)

    # [4] 模型架构与物理参数对齐
    print("\n--- 步骤 3: 架构初始化 (带全局 ID 冗余) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"The device is {device}")
    # 核心：为了让不同国家的数据集能共用模型，我们给 Embedding 留出冗余空间
    # 防止未来加载 'all' 模型时因为类别数增加而报维度错误
    cond_net_params = main_config.get('cond_net_params', {})
    condition_network = EnhancedConditionNetwork(
        num_countries=max(data_info['num_countries'] + 5, 20), # 最少预留 20 个国家位置
        num_indices=max(data_info['num_indices'] + 10, 100),   # 最少预留 100 个指数位置
        **cond_net_params
    ).to(device)

    model = Unet1D(
        cond_dim=cond_net_params.get('output_dim', 128), 
        **main_config.get('unet_params', {})
    ).to(device)

    # 扩散过程选择
    use_dlpm = main_config.get('use_dlpm', True)
    if use_dlpm:
        diffusion = DLPMDiffusion1D(
            model=model, 
            condition_network=condition_network,
            alpha=main_config.get('dlpm_alpha', 1.75),
            **main_config
        ).to(device)
    else:
        diffusion = GaussianDiffusion1D(
            model=model, 
            condition_network=condition_network,
            **main_config
        ).to(device)

    # [5] 训练循环
    print("\n--- 步骤 4: 执行训练 ---")
    
    trainer_params = main_config.get('trainer_params', {})
    dataset = Dataset1D(y_train, X_train, mask_train)
    trainer = Trainer1D(
        diffusion_model=diffusion, 
        dataset=dataset, 
        results_folder=str(pp.Model_Results_DIR / 'Diffusion_Model_DLPM' / asset_name / 'checkpoints'),
        # 从 main_config 或其子项 trainer_params 中显式提取参数
        train_batch_size=trainer_params.get('train_batch_size', main_config.get('train_batch_size', 64)),
        train_lr=trainer_params.get('train_lr', main_config.get('train_lr', 1e-6)),
        train_num_steps=main_config.get('train_num_steps', 20000),
        gradient_accumulate_every=trainer_params.get('gradient_accumulate_every', 1),
        ema_decay=trainer_params.get('ema_decay', main_config.get('ema_decay', 0.995)),
        amp=trainer_params.get('amp', main_config.get('amp', True))
    )
    
    trainer.train()

    # [6] 保存与清理
    print("\n--- 步骤 5: 产出物持久化 ---")
    model_dir = pp.Model_Results_DIR / 'Diffusion_Model_DLPM' / asset_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名定义
    suffix = "_all" if asset_name.lower() == 'all' else f"_{asset_name}"
    paths = {
        'model': model_dir / f"unet_conditional_model{suffix}.pth",
        'cond_net': model_dir / f"condition_network{suffix}.pth",
        'processor': model_dir / f"data_processor{suffix}.pkl",
        'report': pp.Results_DIR / "training_report" / "Diffusion_Model_DLPM" / asset_name / f"report_{timestamp_str}.md",
        'loss_curve': pp.Results_DIR / "training_report" / "Diffusion_Model_DLPM" / asset_name / f"loss_{timestamp_str}.png"
    }

    torch.save(model.state_dict(), paths['model'])
    torch.save(condition_network.state_dict(), paths['cond_net'])
    joblib.dump(data_processor, paths['processor'])
    
    paths['loss_curve'].parent.mkdir(parents=True, exist_ok=True)
    # 绘制损失曲线
    if trainer.loss_history:
        plt.figure(figsize=(10, 5))
        plt.plot(trainer.loss_history)
        plt.title(f"Loss Curve - {asset_name}"); plt.yscale('log')
        plt.savefig(paths['loss_curve']); plt.close()

    # 清理临时文件
    if temp_csv_path and temp_csv_path.exists():
        temp_csv_path.unlink()
        print(f"🧹 已清理临时文件")

    # 生成报告
    training_results = {
        'steps_run': trainer.step,
        'final_loss': trainer.loss_history[-1] if trainer.loss_history else None,
        'duration_seconds': time.time() - training_start_time
    }
    generate_training_report(paths['report'], main_config, data_info, {'type': 'U-Net', 'unet_params': sum(p.numel() for p in model.parameters()), 'cond_net_params': sum(p.numel() for p in condition_network.parameters()), 'total_params': sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in condition_network.parameters())}, training_results, {'model': str(paths['model']), 'condition_network': str(paths['cond_net']), 'processor': str(paths['processor']), 'loss_curve': paths['loss_curve']})

    print(f"\n✅ 资产 '{asset_name}' 的训练任务已圆满完成！")