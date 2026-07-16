# -*- coding: utf-8 -*-
"""统一扩散模型训练入口。

三种实验变体（对应论文的消融设计）：
  --variant ddpm_complex : 高斯DDPM + 复杂损失(MSE + 金融统计正则项)
  --variant ddpm_simple  : 高斯DDPM + 简单损失(仅去噪MSE)
  --variant dlpm         : DLPM (arXiv:2407.18609) + 论文标准损失

用法:
  python Pipelines/4-Run_Diffusion.py --variant dlpm
"""
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
import joblib
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 路径设置 ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

import Project_Path as pp
from Data.Input_preparation import DataProcessor
from Model.Diffusion_Model.diffusion_with_condition import GaussianDiffusion1D
from Model.Diffusion_Model.diffusion_dlpm import DLPMDiffusion1D
from Model.Diffusion_Model.trainer_with_condition import Trainer1D, Dataset1D
from Model.Diffusion_Model.Unet_with_condition import Unet1D
from Model.Diffusion_Model.condition_network import EnhancedConditionNetwork
import Config.Diffusion_config as DDPMConfig
import Config.Diffusion_config_DLPM as DLPMConfig

VARIANTS = ('ddpm_complex', 'ddpm_simple', 'dlpm')


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_and_filter_data(main_config, timestamp_str):
    asset_name = main_config['underlying_asset']
    target_countries = main_config.get('country', 'all')
    if isinstance(target_countries, str):
        target_countries = [target_countries]

    if asset_name.lower() == 'all':
        base_path = pp.Trainning_DATA_DIR / 'trainning_data_merged.csv'
    else:
        base_path = pp.Trainning_DATA_DIR / asset_name / 'train_df.csv'
    if not base_path.exists():
        raise FileNotFoundError(f'无法定位数据源: {base_path}')

    df = pd.read_csv(base_path)
    c_col = next((c for c in ['country_code', 'country'] if c in df.columns), None)
    if c_col and 'all' not in [x.lower() for x in target_countries]:
        df = df[df[c_col].isin(target_countries)]
        print(f'🎯 过滤：已保留国家 {target_countries}, 剩余行数: {len(df)}')
    if df.empty:
        raise ValueError('过滤后无有效数据，请检查配置中的 country 参数。')

    temp_csv_path = pp.Trainning_DATA_DIR / f'temp_run_{asset_name}_{timestamp_str}.csv'
    df.to_csv(temp_csv_path, index=False)
    return base_path, temp_csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, default='dlpm')
    parser.add_argument('--train_num_steps', type=int, default=None)
    parser.add_argument('--train_lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    variant = args.variant
    training_start_time = time.time()
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    main_config = dict(DLPMConfig.main_config if variant == 'dlpm' else DDPMConfig.main_config)
    if variant == 'ddpm_simple':
        main_config['loss_mode'] = 'simple'
    elif variant == 'ddpm_complex':
        main_config['loss_mode'] = 'complex'
    if args.train_num_steps is not None:
        main_config['train_num_steps'] = args.train_num_steps
    if args.train_lr is not None:
        main_config['train_lr'] = args.train_lr
    if args.batch_size is not None:
        main_config['train_batch_size'] = args.batch_size

    asset_name = main_config['underlying_asset']

    print('\n' + '=' * 60)
    print(f'🏁 扩散模型训练 | 变体: {variant}')
    if variant == 'dlpm':
        print(f"   DLPM alpha = {main_config['dlpm_alpha']} | 标准损失 Lp, p={main_config['dlpm_lploss']}")
    else:
        print(f"   DDPM loss_mode = {main_config['loss_mode']} | objective = {main_config['objective']}")
    print('=' * 60)

    # [1] 数据
    temp_csv_path = None
    try:
        base_path, temp_csv_path = load_and_filter_data(main_config, timestamp_str)
        data_processor = DataProcessor(main_config)
        X_train, y_train, mask_train = data_processor.process_all_data(temp_csv_path)
        for name, tensor in [('条件特征(X)', X_train), ('目标序列(y)', y_train), ('有效性Mask', mask_train)]:
            if not torch.isfinite(tensor).all():
                bad = (~torch.isfinite(tensor)).sum().item()
                raise ValueError(f'数据源 {name} 存在 {bad} 个非有限值，请检查 DataProcessor。')
        print('✅ 数据源全量检查通过')
        data_info = {
            'source_file': str(base_path),
            'num_samples': len(X_train),
            'condition_dim': X_train.shape[-1],
            'sequence_length': y_train.shape[-1],
            'num_countries': data_processor.num_countries,
            'num_indices': data_processor.num_indices,
        }
    except Exception as e:
        print(f'❌ 数据处理失败: {e}')
        traceback.print_exc()
        sys.exit(1)

    # [2] 模型
    device = pick_device()
    print(f'The device is {device}')

    cond_net_params = main_config.get('cond_net_params', {})
    condition_network = EnhancedConditionNetwork(
        num_countries=max(data_info['num_countries'] + 5, 20),
        num_indices=max(data_info['num_indices'] + 10, 100),
        **cond_net_params
    ).to(device)

    model = Unet1D(
        cond_dim=cond_net_params.get('output_dim', 128),
        **main_config.get('unet_params', {})
    ).to(device)

    if variant == 'dlpm':
        diffusion = DLPMDiffusion1D(
            model=model,
            condition_network=condition_network,
            alpha=main_config['dlpm_alpha'],
            **{k: v for k, v in main_config.items() if k != 'alpha'}
        ).to(device)
    else:
        diffusion = GaussianDiffusion1D(
            model=model,
            condition_network=condition_network,
            **main_config
        ).to(device)

    # [3] 训练
    results_root = pp.Model_Results_DIR / 'Diffusion_Comparison' / variant / asset_name
    dataset = Dataset1D(y_train, X_train, mask_train)
    train_num_steps = main_config.get('train_num_steps', 4000)
    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=dataset,
        results_folder=str(results_root / 'checkpoints'),
        train_batch_size=main_config.get('train_batch_size', 64),
        train_lr=main_config.get('train_lr', 1e-4),
        train_num_steps=train_num_steps,
        gradient_accumulate_every=1,
        ema_decay=main_config.get('ema_decay', 0.995),
        amp=main_config.get('amp', False),
        save_and_sample_every=train_num_steps,  # 只在训练结束时采样/存档一次
    )
    trainer.train()

    # [4] 保存产物（原始 + EMA 两套权重）
    results_root.mkdir(parents=True, exist_ok=True)
    ema_diffusion = trainer.ema.ema_model

    paths = {
        'model': results_root / 'unet_conditional_model.pth',
        'cond_net': results_root / 'condition_network.pth',
        'model_ema': results_root / 'unet_conditional_model_ema.pth',
        'cond_net_ema': results_root / 'condition_network_ema.pth',
        'processor': results_root / 'data_processor.pkl',
        'config': results_root / 'train_config.json',
        'loss_history': results_root / 'loss_history.json',
        'loss_curve': results_root / 'loss_curve.png',
    }
    torch.save(model.state_dict(), paths['model'])
    torch.save(condition_network.state_dict(), paths['cond_net'])
    torch.save(ema_diffusion.model.state_dict(), paths['model_ema'])
    if ema_diffusion.condition_network is not None:
        torch.save(ema_diffusion.condition_network.state_dict(), paths['cond_net_ema'])
    joblib.dump(data_processor, paths['processor'])

    serializable_cfg = {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in main_config.items() if not isinstance(v, dict)}
    serializable_cfg['unet_params'] = {k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in main_config.get('unet_params', {}).items()}
    serializable_cfg['cond_net_params'] = dict(cond_net_params)
    serializable_cfg['variant'] = variant
    serializable_cfg['data_info'] = data_info
    with open(paths['config'], 'w') as f:
        json.dump(serializable_cfg, f, indent=2, ensure_ascii=False)
    with open(paths['loss_history'], 'w') as f:
        json.dump(trainer.loss_history, f)

    if trainer.loss_history:
        plt.figure(figsize=(10, 5))
        plt.plot(trainer.loss_history)
        plt.title(f'Loss Curve - {variant} - {asset_name}')
        plt.yscale('log')
        plt.savefig(paths['loss_curve'])
        plt.close()

    if temp_csv_path and temp_csv_path.exists():
        temp_csv_path.unlink()

    duration = time.time() - training_start_time
    print('\n' + '=' * 60)
    print(f"✅ 变体 '{variant}' 训练完成 | 步数: {trainer.step} | "
          f"最终loss: {trainer.loss_history[-1] if trainer.loss_history else float('nan'):.5f} | "
          f'耗时: {duration/60:.1f} 分钟')
    print(f'   产物目录: {results_root}')
    print('=' * 60)


if __name__ == '__main__':
    main()
