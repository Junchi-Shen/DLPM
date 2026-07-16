# -*- coding: utf-8 -*-
"""
为 P-Q 博弈生成路径文件（供 Game/backtest_engine.Backtester 消费）。
- P 模型: Diffusion_Comparison 下的 dlpm (DLIM-50) 与 ddpm_complex (DDIM-50)
  输出 (N_cond, N_sim, 251) 缩放对数收益 -> {base}_{ts}_samples.npy
- Q 模型: GBM Monte Carlo (逐窗口用历史波动率+期限匹配无风险利率)
  输出 (N_cond*N_sim_q, 1, 253) 价格 -> gbm_generated_paths_{ts}_samples.npy
- 窗口顺序与 Backtester 一致: testing_data_merged.csv 按 asset_underlying 过滤后的原始顺序
"""
import sys
import json
import time
import shutil
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path('/Users/junchishen/Downloads/DLPM')
sys.path.append(str(project_root))
import Project_Path as pp  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'cmp', str(project_root / 'Pipelines' / '11-Compare_Models.py'))
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

ASSET = 'CSI1000'
OUT_DIR = Path(pp.Path_Generator_Results_DIR) / ASSET
N_SIM_P = 128
N_SIM_Q = 512
SAMPLING_STEPS = 50
SEED = 42


def load_val_df():
    df = pd.read_csv(Path(pp.Testing_DATA_DIR) / 'testing_data_merged.csv')
    val_df = df[df['asset_underlying'] == ASSET].copy()
    print(f'val_df: {len(val_df)} 条 {ASSET} 测试窗口')
    return val_df


def build_conditions(processor, val_df):
    dfp = processor.process_price_data(val_df.copy())
    dfp = processor.transform_price_sequence(dfp)
    cond = processor.create_condition_tensors(dfp, fit_scaler=False)
    X = torch.FloatTensor(cond['conditions'])
    mask = torch.FloatTensor(np.array(dfp['validity_mask'].tolist())).unsqueeze(1)
    return X, mask


def gen_diffusion_paths(variant, val_df, device, ts):
    diffusion, processor, cfg = cmp.load_variant(variant, device)
    X, mask = build_conditions(processor, val_df)
    N = len(val_df)
    T = X.shape[0]
    assert T == N

    if variant != 'dlpm':  # GaussianDiffusion1D: 切换到 DDIM-50
        diffusion.sampling_timesteps = SAMPLING_STEPS
        diffusion.is_ddim_sampling = True

    base = {'dlpm': 'dlpm_generated_paths', 'ddpm_complex': 'ddpm_generated_paths'}[variant]
    out = np.zeros((N, N_SIM_P, 251), dtype=np.float32)

    windows_per_chunk = 8
    t0 = time.time()
    for s in range(0, N, windows_per_chunk):
        e = min(s + windows_per_chunk, N)
        k = e - s
        Xb = X[s:e].repeat_interleave(N_SIM_P, dim=0).to(device)
        Mb = mask[s:e].repeat_interleave(N_SIM_P, dim=0).to(device)
        with torch.inference_mode():
            if variant == 'dlpm':
                g = diffusion.sample(batch_size=k * N_SIM_P, cond_input=Xb, mask=Mb,
                                     sampling_timesteps=SAMPLING_STEPS)
            else:
                g = diffusion.sample(batch_size=k * N_SIM_P, cond_input=Xb, mask=Mb)
        g = (g * Mb)[:, 0, 1:].float().cpu().numpy()          # 去掉标记位, 掩掉无效段
        out[s:e] = g.reshape(k, N_SIM_P, 251)
        if (s // windows_per_chunk) % 10 == 0:
            done = e * N_SIM_P
            rate = done / max(time.time() - t0, 1e-9)
            eta = (N * N_SIM_P - done) / max(rate, 1e-9) / 60
            print(f'  [{variant}] {e}/{N} 窗口 | {rate:.0f} paths/s | ETA {eta:.0f} min', flush=True)

    path = OUT_DIR / f'{base}_{ts}_samples.npy'
    np.save(path, out)
    print(f'✅ [{variant}] 已保存 {path.name} 形状 {out.shape} 耗时 {(time.time()-t0)/60:.1f} min')
    del diffusion
    if device == 'mps':
        torch.mps.empty_cache()


def gen_gbm_paths(val_df, ts):
    rng = np.random.default_rng(SEED)
    N = len(val_df)
    out = np.zeros((N * N_SIM_Q, 1, 253), dtype=np.float32)
    dt = 1.0 / 252.0
    for i, (_, row) in enumerate(val_df.iterrows()):
        S0 = float(row['start_price'])
        sigma = float(row['volatility'])
        r = float(row['risk_free_rate'])
        z = rng.standard_normal((N_SIM_Q, 252))
        log_inc = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        log_prices = np.log(S0) + np.concatenate(
            [np.zeros((N_SIM_Q, 1)), np.cumsum(log_inc, axis=1)], axis=1)
        out[i * N_SIM_Q:(i + 1) * N_SIM_Q, 0, :] = np.exp(log_prices)
    path = OUT_DIR / f'gbm_generated_paths_{ts}_samples.npy'
    np.save(path, out)
    print(f'✅ [GBM] 已保存 {path.name} 形状 {out.shape}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['dlpm', 'ddpm_complex'])
    ap.add_argument('--skip_gbm', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    val_df = load_val_df()

    # Backtester 需要的处理器副本
    proc_dst = Path(pp.Model_Results_DIR) / 'Diffusion_Model_DLPM' / 'all'
    proc_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(pp.Model_Results_DIR) / 'Diffusion_Comparison' / 'dlpm' / 'all' / 'data_processor.pkl',
                proc_dst / 'data_processor_all.pkl')
    print(f'✅ 处理器已就位: {proc_dst / "data_processor_all.pkl"}')

    if not args.skip_gbm:
        gen_gbm_paths(val_df, ts)
    for v in args.variants:
        gen_diffusion_paths(v, val_df, device, ts)

    json.dump({'timestamp': ts, 'n_windows': len(val_df), 'n_sim_p': N_SIM_P,
               'n_sim_q': N_SIM_Q, 'sampling_steps': SAMPLING_STEPS},
              open(OUT_DIR / f'game_paths_meta_{ts}.json', 'w'), indent=2)
    print('🎉 全部路径生成完毕')


if __name__ == '__main__':
    main()
