# -*- coding: utf-8 -*-
"""三模型路径生成质量对比：ddpm_complex / ddpm_simple / dlpm。

对每个已训练变体：
  1. 加载 EMA 权重与训练期拟合的 DataProcessor
  2. 在测试集(中国指数)条件下生成价格路径(完整反向链)
  3. 与真实路径逐样本比较：均值差/年化波动差/峰度差/KS/Wasserstein/QQ R^2
  4. 汇总池化收益的尾部统计(峰度、极端分位数、|r|滞后自相关)

结果写入 Results/Model_Results/Diffusion_Comparison/comparison_results.json
以及 comparison_table.md。

用法: python Pipelines/11-Compare_Models.py [--max_samples 400] [--dlim_steps 50]
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

import Project_Path as pp
from Model.Diffusion_Model.diffusion_with_condition import GaussianDiffusion1D
from Model.Diffusion_Model.diffusion_dlpm import DLPMDiffusion1D
from Model.Diffusion_Model.Unet_with_condition import Unet1D
from Model.Diffusion_Model.condition_network import EnhancedConditionNetwork

VARIANTS = ('ddpm_simple', 'ddpm_complex', 'dlpm')
COMP_ROOT = pp.Model_Results_DIR / 'Diffusion_Comparison'


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_variant(variant, device):
    root = COMP_ROOT / variant / 'all'
    with open(root / 'train_config.json') as f:
        cfg = json.load(f)
    processor = joblib.load(root / 'data_processor.pkl')

    cond_net_params = cfg['cond_net_params']
    condition_network = EnhancedConditionNetwork(
        num_countries=max(cfg['data_info']['num_countries'] + 5, 20),
        num_indices=max(cfg['data_info']['num_indices'] + 10, 100),
        **cond_net_params
    ).to(device)
    unet_params = dict(cfg['unet_params'])
    if isinstance(unet_params.get('dim_mults'), list):
        unet_params['dim_mults'] = tuple(unet_params['dim_mults'])
    model = Unet1D(cond_dim=cond_net_params.get('output_dim', 128), **unet_params).to(device)

    model.load_state_dict(torch.load(root / 'unet_conditional_model_ema.pth', map_location=device))
    condition_network.load_state_dict(torch.load(root / 'condition_network_ema.pth', map_location=device))
    model.eval(); condition_network.eval()

    if variant == 'dlpm':
        diffusion = DLPMDiffusion1D(
            model=model, condition_network=condition_network,
            alpha=cfg['dlpm_alpha'],
            **{k: v for k, v in cfg.items() if k not in ('alpha', 'variant', 'data_info', 'unet_params', 'cond_net_params')}
        ).to(device)
    else:
        diffusion = GaussianDiffusion1D(
            model=model, condition_network=condition_network,
            **{k: v for k, v in cfg.items() if k not in ('variant', 'data_info', 'unet_params', 'cond_net_params')}
        ).to(device)
    diffusion.eval()
    return diffusion, processor, cfg


def build_test_tensors(processor, cfg, max_samples):
    """用训练期的 scaler / ID 映射构造测试条件（不重新拟合）"""
    test_path = pp.Testing_DATA_DIR / 'testing_data_merged.csv'
    df = pd.read_csv(test_path)
    c_col = next((c for c in ['country_code', 'country'] if c in df.columns), None)
    countries = cfg.get('country', ['China'])
    if c_col and countries and 'all' not in [str(x).lower() for x in countries]:
        df = df[df[c_col].isin(countries)]
    # 只保留训练期见过的指数类别
    if processor.index_table is not None and 'asset_underlying' in df.columns:
        df = df[df['asset_underlying'].astype(str).isin(processor.index_table.keys())]
    df = df.reset_index(drop=True)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    print(f'测试样本数: {len(df)}')

    df = processor.process_price_data(df)
    df = processor.transform_price_sequence(df)

    # 数值特征：与 create_condition_tensors 完全一致，但用已拟合 scaler
    prices = df['S_0'].astype(float).values / cfg['base_trading_days']
    contract_days = df['contract_calendar_days'].astype(int).values / 365.0
    trading_days = df['actual_trading_days'].astype(int).values / float(cfg['base_trading_days'])
    volatility = df['volatility'].astype(float).values
    risk_free_rate = df['risk_free_rate'].astype(float).values
    numerical = np.column_stack([prices, volatility, risk_free_rate,
                                 contract_days, trading_days / contract_days]).astype(np.float32)
    scaled = processor.price_scaler.transform(numerical)

    country_id = df['country'].astype(str).map(processor.country_table).fillna(0).values \
        if processor.country_table is not None and 'country' in df.columns else np.zeros(len(df))
    index_id = df['asset_underlying'].astype(str).map(processor.index_table).fillna(0).values \
        if processor.index_table is not None and 'asset_underlying' in df.columns else np.zeros(len(df))

    X = np.column_stack([scaled, country_id.reshape(-1, 1), index_id.reshape(-1, 1)]).astype(np.float32)
    y = np.array(df['transformed_sequence'].tolist(), dtype=np.float32).reshape(len(df), 1, -1)
    m = np.array(df['validity_mask'].tolist(), dtype=np.float32).reshape(len(df), 1, -1)
    return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(m)


@torch.no_grad()
def generate(diffusion, X, mask, device, batch=128, sampling_timesteps=None):
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].to(device)
        mb = mask[i:i + batch].to(device)
        if sampling_timesteps is not None and hasattr(diffusion, 'ddim_sample') and isinstance(diffusion, DLPMDiffusion1D):
            s = diffusion.sample(batch_size=xb.shape[0], cond_input=xb, mask=mb,
                                 sampling_timesteps=sampling_timesteps)
        else:
            s = diffusion.sample(batch_size=xb.shape[0], cond_input=xb, mask=mb)
        outs.append(s.cpu())
    return torch.cat(outs, dim=0)


def per_sample_metrics(gen, real, mask, vol_scale):
    """gen/real/mask: (N,1,T) tensors; 收益从位置1开始(位置0是起始标记)"""
    rows = []
    N = gen.shape[0]
    for i in range(N):
        m = mask[i, 0].numpy() > 0.5
        idx = np.where(m)[0]
        idx = idx[idx >= 1]
        if len(idx) < 10:
            continue
        g = gen[i, 0].numpy()[idx] * vol_scale
        r = real[i, 0].numpy()[idx] * vol_scale
        ks_stat, ks_p = st.ks_2samp(g, r)
        wd = st.wasserstein_distance(g, r)
        qs = np.linspace(0.01, 0.99, 99)
        gq, rq = np.quantile(g, qs), np.quantile(r, qs)
        ss_res = np.sum((gq - np.poly1d(np.polyfit(rq, gq, 1))(rq)) ** 2)
        ss_tot = np.sum((gq - gq.mean()) ** 2)
        qq_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append({
            'mean_diff': abs(g.mean() - r.mean()),
            'vol_diff': abs(g.std(ddof=1) * np.sqrt(252) - r.std(ddof=1) * np.sqrt(252)),
            'kurt_diff': abs(st.kurtosis(g) - st.kurtosis(r)),
            'ks_stat': ks_stat, 'ks_p': ks_p,
            'wasserstein': wd, 'qq_r2': qq_r2,
        })
    return pd.DataFrame(rows)


def pooled_returns(gen, mask, vol_scale):
    out = []
    for i in range(gen.shape[0]):
        m = mask[i, 0].numpy() > 0.5
        idx = np.where(m)[0]
        idx = idx[idx >= 1]
        if len(idx) >= 30:
            out.append(gen[i, 0].numpy()[idx] * vol_scale)
    return np.concatenate(out)


def plot_tail_density(pooled_dict, out_path):
    """对数密度图：真实 vs 各模型池化日收益分布（尾部可视化）"""
    plt.figure(figsize=(8, 5))
    styles = {'Real': ('black', '-'), 'DDPM-simple': ('tab:blue', '--'),
              'DDPM-complex': ('tab:green', '-.'), 'DLPM': ('tab:red', '-')}
    lo = min(np.quantile(x, 0.0005) for x in pooled_dict.values())
    hi = max(np.quantile(x, 0.9995) for x in pooled_dict.values())
    bins = np.linspace(lo, hi, 120)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for name, x in pooled_dict.items():
        dens, _ = np.histogram(x, bins=bins, density=True)
        color, ls = styles.get(name, ('gray', ':'))
        plt.semilogy(centers, np.where(dens > 0, dens, np.nan),
                     color=color, linestyle=ls, label=name, lw=1.6)
    plt.xlabel('Daily log return')
    plt.ylabel('Density (log scale)')
    plt.legend(frameon=False)
    plt.title('Pooled return distributions: tails')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'  📈 尾部密度图已保存: {out_path}')


def pooled_metrics(gen, real, mask, vol_scale):
    g_all, r_all, acf_g, acf_r = [], [], [], []
    for i in range(gen.shape[0]):
        m = mask[i, 0].numpy() > 0.5
        idx = np.where(m)[0]
        idx = idx[idx >= 1]
        if len(idx) < 30:
            continue
        g = gen[i, 0].numpy()[idx] * vol_scale
        r = real[i, 0].numpy()[idx] * vol_scale
        g_all.append(g); r_all.append(r)
        ag = np.abs(g) - np.abs(g).mean(); ar = np.abs(r) - np.abs(r).mean()
        if ag.std() > 0:
            acf_g.append(np.corrcoef(ag[:-1], ag[1:])[0, 1])
        if ar.std() > 0:
            acf_r.append(np.corrcoef(ar[:-1], ar[1:])[0, 1])
    g_all = np.concatenate(g_all); r_all = np.concatenate(r_all)
    out = {}
    for tag, x in [('gen', g_all), ('real', r_all)]:
        out[tag] = {
            'kurtosis': float(st.kurtosis(x)),
            'skewness': float(st.skew(x)),
            'std_annualized': float(x.std(ddof=1) * np.sqrt(252)),
            'q_0.1%': float(np.quantile(x, 0.001)),
            'q_1%': float(np.quantile(x, 0.01)),
            'q_99%': float(np.quantile(x, 0.99)),
            'q_99.9%': float(np.quantile(x, 0.999)),
            'abs_ret_acf1': float(np.mean(acf_g if tag == 'gen' else acf_r)),
        }
    out['pooled_ks'] = float(st.ks_2samp(g_all, r_all)[0])
    out['pooled_wasserstein'] = float(st.wasserstein_distance(g_all, r_all))
    out['kurtosis_gap'] = float(abs(out['gen']['kurtosis'] - out['real']['kurtosis']))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=400)
    parser.add_argument('--dlim_steps', type=int, default=50)
    args = parser.parse_args()

    device = pick_device()
    print(f'device = {device}')
    torch.manual_seed(42); np.random.seed(42)

    results = {}
    tables = {}
    density_data = {}
    for variant in VARIANTS:
        root = COMP_ROOT / variant / 'all'
        if not (root / 'unet_conditional_model_ema.pth').exists():
            print(f'⚠️ 跳过 {variant}（未找到训练产物）')
            continue
        print(f'\n===== {variant} =====')
        diffusion, processor, cfg = load_variant(variant, device)
        X, y, m = build_test_tensors(processor, cfg, args.max_samples)
        vol_scale = cfg['volatility_scale']

        t0 = time.time()
        gen = generate(diffusion, X, m, device)
        gen_time = time.time() - t0
        print(f'  生成 {X.shape[0]} 条路径耗时 {gen_time:.0f}s')

        df = per_sample_metrics(gen, y, m, vol_scale)
        pooled = pooled_metrics(gen, y, m, vol_scale)
        label = {'ddpm_simple': 'DDPM-simple', 'ddpm_complex': 'DDPM-complex', 'dlpm': 'DLPM'}[variant]
        density_data[label] = pooled_returns(gen, m, vol_scale)
        if 'Real' not in density_data:
            density_data['Real'] = pooled_returns(y, m, vol_scale)
        summary = {c: {'mean': float(df[c].mean()), 'std': float(df[c].std())} for c in df.columns}
        results[variant] = {'per_sample': summary, 'pooled': pooled,
                            'n_eval': int(len(df)), 'gen_seconds': gen_time}
        tables[variant] = df
        print(json.dumps(summary, indent=2))
        print('pooled kurtosis: gen={:.2f} real={:.2f}'.format(
            pooled['gen']['kurtosis'], pooled['real']['kurtosis']))

        # DLPM 附加：DLIM 加速采样质量
        if variant == 'dlpm' and args.dlim_steps:
            t0 = time.time()
            gen_fast = generate(diffusion, X, m, device, sampling_timesteps=args.dlim_steps)
            fast_time = time.time() - t0
            df_fast = per_sample_metrics(gen_fast, y, m, vol_scale)
            pooled_fast = pooled_metrics(gen_fast, y, m, vol_scale)
            results['dlpm_dlim'] = {
                'per_sample': {c: {'mean': float(df_fast[c].mean()), 'std': float(df_fast[c].std())} for c in df_fast.columns},
                'pooled': pooled_fast, 'n_eval': int(len(df_fast)),
                'gen_seconds': fast_time, 'dlim_steps': args.dlim_steps,
            }
            print(f'  DLIM({args.dlim_steps}步) 生成耗时 {fast_time:.0f}s (完整链 {gen_time:.0f}s)')

    COMP_ROOT.mkdir(parents=True, exist_ok=True)
    with open(COMP_ROOT / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if density_data:
        ordered = {k: density_data[k] for k in ['Real', 'DDPM-simple', 'DDPM-complex', 'DLPM']
                   if k in density_data}
        plot_tail_density(ordered, COMP_ROOT / 'tail_density.png')

    # Markdown 汇总表
    metric_names = ['mean_diff', 'vol_diff', 'kurt_diff', 'ks_stat', 'ks_p', 'wasserstein', 'qq_r2']
    lines = ['| Metric | ' + ' | '.join(results.keys()) + ' |',
             '|---|' + '---|' * len(results)]
    for mname in metric_names:
        row = [f"{results[v]['per_sample'][mname]['mean']:.4f} ± {results[v]['per_sample'][mname]['std']:.4f}"
               if mname in results[v]['per_sample'] else '-' for v in results]
        lines.append(f'| {mname} | ' + ' | '.join(row) + ' |')
    lines.append('')
    lines.append('| Pooled | ' + ' | '.join(results.keys()) + ' |')
    lines.append('|---|' + '---|' * len(results))
    for pname in ['kurtosis_gap', 'pooled_ks', 'pooled_wasserstein']:
        row = [f"{results[v]['pooled'][pname]:.4f}" for v in results]
        lines.append(f'| {pname} | ' + ' | '.join(row) + ' |')
    row = [f"{results[v]['pooled']['gen']['kurtosis']:.2f}" for v in results]
    lines.append('| gen kurtosis (real={:.2f}) | '.format(
        results[list(results.keys())[0]]['pooled']['real']['kurtosis']) + ' | '.join(row) + ' |')
    with open(COMP_ROOT / 'comparison_table.md', 'w') as f:
        f.write('\n'.join(lines))
    print('\n' + '\n'.join(lines))
    print(f'\n✅ 结果已写入 {COMP_ROOT}/comparison_results.json 和 comparison_table.md')


if __name__ == '__main__':
    main()
