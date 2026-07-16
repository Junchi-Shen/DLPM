# -*- coding: utf-8 -*-
"""
Neural SDE 基线：dX = mu_theta(h,c) dt + sigma_theta(h,c) dW（Euler–Maruyama, 日频）
- GRU 编码历史收益状态，MLP 输出漂移与扩散系数，高斯 MLE 训练
- 与扩散模型同一数据、同一评估协议（复用 11-Compare_Models 的指标函数）
"""
import sys
import json
import time
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

project_root = Path('/Users/junchishen/Downloads/DLPM')
sys.path.append(str(project_root))

from Data.Input_preparation import DataProcessor  # noqa: E402
import Project_Path as pp  # noqa: E402

# 复用对比脚本的指标函数
_spec = importlib.util.spec_from_file_location(
    'cmp', str(project_root / 'Pipelines' / '11-Compare_Models.py'))
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

COMP_ROOT = Path(pp.Model_Results_DIR) / 'Diffusion_Comparison'
OUT_DIR = COMP_ROOT / 'neural_sde'
TRAIN_CSV = Path(pp.Trainning_DATA_DIR) / 'trainning_data_merged.csv'
TEST_CSV = Path(pp.Testing_DATA_DIR) / 'testing_data_merged.csv'

CFG = {
    'volatility_scale': 0.09,
    'input_sequence_length': 252,
    'base_trading_days': 252,
    'country': ['China'],
    'hidden': 64,
    'cond_emb': 32,
    'train_steps': 4000,
    'batch': 64,
    'lr': 1e-3,
    'seed': 42,
}


class NeuralSDE(nn.Module):
    """条件 GRU-SDE：状态 h_k 吸收历史收益，输出每步高斯增量的 (mu, log sigma)。"""

    def __init__(self, cond_dim=7, hidden=64, cond_emb=32):
        super().__init__()
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, cond_emb), nn.SiLU(),
            nn.Linear(cond_emb, cond_emb))
        self.gru = nn.GRUCell(2 + cond_emb, hidden)  # 输入: [r_{k-1}, k/L, cond]
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.hidden = hidden

    def _step_params(self, h):
        out = self.head(h)
        mu = out[:, 0]
        log_sigma = out[:, 1].clamp(-6.0, 2.0)
        return mu, log_sigma

    def nll(self, y, mask, cond):
        """teacher forcing 的逐步高斯负对数似然。y:(B,1,T) 含首位=1标记。"""
        B, _, T = y.shape
        r = y[:, 0, :]          # 位置0是标记，收益从1开始
        m = mask[:, 0, :]
        c = self.cond_net(cond)
        h = torch.zeros(B, self.hidden, device=y.device)
        prev = torch.zeros(B, device=y.device)
        total, count = 0.0, 0.0
        for k in range(1, T):
            frac = torch.full((B,), k / T, device=y.device)
            h = self.gru(torch.cat([prev.unsqueeze(1), frac.unsqueeze(1), c], dim=1), h)
            mu, log_sigma = self._step_params(h)
            var = (2 * log_sigma).exp()
            step_nll = 0.5 * ((r[:, k] - mu) ** 2 / var + 2 * log_sigma + np.log(2 * np.pi))
            total = total + (step_nll * m[:, k]).sum()
            count = count + m[:, k].sum()
            prev = r[:, k] * m[:, k]  # 无效位喂0
        return total / count.clamp(min=1.0)

    @torch.no_grad()
    def sample(self, cond, mask):
        """自回归采样，形状与扩散模型输出一致 (B,1,T)。"""
        B = cond.shape[0]
        T = CFG['input_sequence_length']
        m = mask[:, 0, :]
        c = self.cond_net(cond)
        h = torch.zeros(B, self.hidden)
        prev = torch.zeros(B)
        out = torch.zeros(B, 1, T)
        out[:, 0, 0] = 1.0
        for k in range(1, T):
            frac = torch.full((B,), k / T)
            h = self.gru(torch.cat([prev.unsqueeze(1), frac.unsqueeze(1), c], dim=1), h)
            mu, log_sigma = self._step_params(h)
            r_k = mu + log_sigma.exp() * torch.randn(B)
            r_k = r_k * m[:, k]
            out[:, 0, k] = r_k
            prev = r_k
        return out


def load_split(csv_path, processor, fit):
    df = pd.read_csv(csv_path)
    if 'country' in df.columns and CFG['country']:
        df = df[df['country'].isin(CFG['country'])].reset_index(drop=True)
    df = processor.process_price_data(df)
    df = processor.transform_price_sequence(df)
    cond = processor.create_condition_tensors(df, fit_scaler=fit)
    X = torch.FloatTensor(cond['conditions'])
    y = torch.FloatTensor(np.array(df['transformed_sequence'].tolist())).unsqueeze(1)
    mask = torch.FloatTensor(np.array(df['validity_mask'].tolist())).unsqueeze(1)
    return X, y, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_samples', type=int, default=400)
    args = ap.parse_args()

    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('📥 处理训练/测试数据 ...')
    processor = DataProcessor(CFG)
    Xtr, ytr, mtr = load_split(TRAIN_CSV, processor, fit=True)
    Xte, yte, mte = load_split(TEST_CSV, processor, fit=False)
    joblib.dump(processor, OUT_DIR / 'data_processor.pkl')
    print(f'   train={len(ytr)}, test={len(yte)}')

    model = NeuralSDE(hidden=CFG['hidden'], cond_emb=CFG['cond_emb'])
    opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['train_steps'])

    print(f"🚀 训练 Neural SDE ({CFG['train_steps']} 步, batch {CFG['batch']}, CPU)")
    t0 = time.time()
    hist = []
    N = len(ytr)
    for step in range(CFG['train_steps']):
        idx = torch.randint(0, N, (CFG['batch'],))
        loss = model.nll(ytr[idx], mtr[idx], Xtr[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss))
        if step % 500 == 0:
            print(f'  step {step}: nll={loss:.4f}')
    dur = time.time() - t0
    print(f'✅ 训练完成: {dur/60:.1f} 分钟, 最终 nll={np.mean(hist[-100:]):.4f}')
    torch.save(model.state_dict(), OUT_DIR / 'neural_sde.pth')
    json.dump(hist, open(OUT_DIR / 'loss_history.json', 'w'))

    # ---- 评估：与扩散模型同一协议 ----
    n_eval = min(args.max_samples, len(yte))
    g = torch.Generator().manual_seed(123)
    order = torch.randperm(len(yte), generator=g)[:n_eval]
    Xe, ye, me = Xte[order], yte[order], mte[order]

    print(f'🎲 采样 {n_eval} 条路径 ...')
    model.eval()
    gen = model.sample(Xe, me)

    df_metrics = cmp.per_sample_metrics(gen, ye, me, CFG['volatility_scale'])
    pooled = cmp.pooled_metrics(gen, ye, me, CFG['volatility_scale'])
    summary = {
        'variant': 'neural_sde',
        'n_eval': int(n_eval),
        'train_minutes': dur / 60,
        'per_sample': {c: {'mean': float(df_metrics[c].mean()),
                           'std': float(df_metrics[c].std())}
                       for c in df_metrics.columns},
        'pooled': pooled,
    }
    json.dump(summary, open(OUT_DIR / 'results.json', 'w'), indent=2)
    print(json.dumps(summary['per_sample'], indent=2))
    print('pooled:', {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in pooled.items() if not isinstance(v, dict)})
    print(f'✅ 结果已写入 {OUT_DIR}/results.json')


if __name__ == '__main__':
    main()
