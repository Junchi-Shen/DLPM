# -*- coding: utf-8 -*-
"""
Chronos (TSFM) 基线：amazon/chronos-t5-small 零样本概率预测。
- 上下文 = 窗口起始日(含)前最多 252 个交易日的指数收盘价
- 预测 = 自回归采样 L-1 步价格延续（num_samples=1，与其他模型的单路径协议一致）
- 指标 = 复用 11-Compare_Models 的 per_sample / pooled 协议
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

project_root = Path('/Users/junchishen/Downloads/DLPM')
sys.path.append(str(project_root))

from Data.Input_preparation import DataProcessor  # noqa: E402
import Project_Path as pp  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'cmp', str(project_root / 'Pipelines' / '11-Compare_Models.py'))
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

OUT_DIR = Path(pp.Model_Results_DIR) / 'Diffusion_Comparison' / 'chronos'
MODEL_DIR = project_root / 'Pretrained' / 'chronos-t5-small'
TRAIN_CSV = Path(pp.Trainning_DATA_DIR) / 'trainning_data_merged.csv'
TEST_CSV = Path(pp.Testing_DATA_DIR) / 'testing_data_merged.csv'

CFG = {'volatility_scale': 0.09, 'input_sequence_length': 252,
       'base_trading_days': 252, 'country': ['China'],
       'context_len': 252, 'seed': 42}


def build_price_history(train_df, test_df):
    """由滑窗行重建各指数的日频价格序列: (asset, date) -> start_price"""
    cols = ['asset_underlying', 'start_date', 'start_price']
    hist = pd.concat([train_df[cols], test_df[cols]])
    hist['start_date'] = pd.to_datetime(hist['start_date'])
    hist = hist.drop_duplicates(['asset_underlying', 'start_date'])
    hist = hist.sort_values(['asset_underlying', 'start_date'])
    return {a: g.set_index('start_date')['start_price']
            for a, g in hist.groupby('asset_underlying')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_samples', type=int, default=400)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('📥 读取数据 ...')
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    test_df = test_df[test_df['country'].isin(CFG['country'])].reset_index(drop=True)
    history = build_price_history(train_df, test_df)

    # 与其他基线一致的测试窗口选择（seed=123 的 randperm）
    processor = DataProcessor(CFG)
    dfp = processor.process_price_data(test_df.copy())
    dfp = processor.transform_price_sequence(dfp)
    y = torch.FloatTensor(np.array(dfp['transformed_sequence'].tolist())).unsqueeze(1)
    mask = torch.FloatTensor(np.array(dfp['validity_mask'].tolist())).unsqueeze(1)
    n_eval = min(args.max_samples, len(y))
    g = torch.Generator().manual_seed(123)
    order = torch.randperm(len(y), generator=g)[:n_eval]
    ye, me = y[order], mask[order]
    rows = dfp.iloc[order.numpy()]

    print(f'🤖 加载 Chronos ({MODEL_DIR}) 到 {args.device} ...')
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(
        str(MODEL_DIR), device_map=args.device, torch_dtype=torch.float32)

    print(f'🎲 逐窗口采样 {n_eval} 条延续路径 ...')
    t0 = time.time()
    T = CFG['input_sequence_length']
    gen = torch.zeros(n_eval, 1, T)
    gen[:, 0, 0] = 1.0
    skipped = 0
    for i, (_, row) in enumerate(rows.iterrows()):
        L = len(row['price_series'])          # 价格点数, 收益数 = L-1
        s_date = pd.to_datetime(row['start_date'])
        h = history[row['asset_underlying']]
        ctx = h[h.index <= s_date].tail(CFG['context_len']).values
        if len(ctx) < 30:
            skipped += 1
            continue
        context = torch.tensor(ctx, dtype=torch.float32)
        with torch.inference_mode():
            fc = pipe.predict(context=context, prediction_length=L - 1,
                              num_samples=1, limit_prediction_length=False)
        levels = fc[0, 0].numpy().astype(np.float64)
        levels = np.clip(levels, 1e-6, None)
        prices = np.concatenate([[float(row['S_0'])], levels])
        rets = np.diff(np.log(prices)) / CFG['volatility_scale']
        n = min(len(rets), T - 1)
        gen[i, 0, 1:n + 1] = torch.tensor(rets[:n], dtype=torch.float32)
        if i % 50 == 0:
            print(f'  {i}/{n_eval} ({time.time()-t0:.0f}s)')
    dur = time.time() - t0
    print(f'✅ 采样完成: {dur/60:.1f} 分钟, 跳过 {skipped} 条(历史不足)')

    df_metrics = cmp.per_sample_metrics(gen, ye, me, CFG['volatility_scale'])
    pooled = cmp.pooled_metrics(gen, ye, me, CFG['volatility_scale'])
    summary = {
        'variant': 'chronos_t5_small',
        'n_eval': int(n_eval), 'skipped': int(skipped),
        'gen_minutes': dur / 60,
        'per_sample': {c: {'mean': float(df_metrics[c].mean()),
                           'std': float(df_metrics[c].std())}
                       for c in df_metrics.columns},
        'pooled': pooled,
    }
    json.dump(summary, open(OUT_DIR / 'results.json', 'w'), indent=2)
    print(json.dumps(summary['per_sample'], indent=2))
    print(f'✅ 结果已写入 {OUT_DIR}/results.json')


if __name__ == '__main__':
    main()
