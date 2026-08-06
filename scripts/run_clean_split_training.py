# -*- coding: utf-8 -*-
"""ç»Ÿä¸€æ‰©æ•£æ¨¡åž‹è®­ç»ƒå…¥å£ã€‚

ä¸‰ç§å®žéªŒå˜ä½“ï¼ˆå¯¹åº”è®ºæ–‡çš„æ¶ˆèžè®¾è®¡ï¼‰ï¼š
  --variant ddpm_complex : é«˜æ–¯DDPM + å¤æ‚æŸå¤±(MSE + é‡‘èžç»Ÿè®¡æ­£åˆ™é¡¹)
  --variant ddpm_simple  : é«˜æ–¯DDPM + ç®€å•æŸå¤±(ä»…åŽ»å™ªMSE)
  --variant dlpm         : DLPM (arXiv:2407.18609) + è®ºæ–‡æ ‡å‡†æŸå¤±

ç”¨æ³•:
  python Pipelines/4-Run_Diffusion.py --variant dlpm
"""
import sys
import argparse
import time
import json
import ast
import random
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch
try:
    import joblib
except ImportError:
    joblib = None
import pickle
import traceback
import re
import os
import hashlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# --- è·¯å¾„è®¾ç½® ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root / "src" / "dlpm"))

import Project_Path as pp
from Data.Input_preparation import DataProcessor
from Model.Diffusion_Model.diffusion_with_condition import GaussianDiffusion1D
from Model.Diffusion_Model.diffusion_dlpm import DLPMDiffusion1D
from Model.Diffusion_Model.trainer_with_condition import Trainer1D, Dataset1D
from Model.Diffusion_Model.Unet_with_condition import Unet1D
from Model.Diffusion_Model.condition_network import EnhancedConditionNetwork
import Config.Diffusion_config as DDPMConfig
import Config.Diffusion_config_DLPM as DLPMConfig
import Config.Diffusion_config_DLPM_hybrid as HybridDLPMConfig

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')

VARIANTS = ('ddpm_complex', 'ddpm_simple', 'dlpm', 'dlpm_hybrid')


def jsonable(value):
    return json.loads(json.dumps(value, default=str))


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    os.replace(temporary, path)


def config_fingerprint(payload):
    canonical = json.dumps(jsonable(payload), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def safe_tag(value):
    if value is None:
        return None
    cleaned = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))
    return cleaned.strip('_') or None


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_and_filter_data(main_config, timestamp_str, max_train_samples=None):
    asset_name = main_config['underlying_asset']
    target_countries = main_config.get('country', 'all')
    if isinstance(target_countries, str):
        target_countries = [target_countries]

    if main_config.get('train_data_filename'):
        base_path = pp.Trainning_DATA_DIR / main_config['train_data_filename']
    elif asset_name.lower() == 'all':
        base_path = pp.Trainning_DATA_DIR / 'trainning_data_merged.csv'
    else:
        base_path = pp.Trainning_DATA_DIR / asset_name / 'train_df.csv'
    if not base_path.exists():
        raise FileNotFoundError(f'æ— æ³•å®šä½æ•°æ®æº: {base_path}')

    df = pd.read_csv(base_path, low_memory=False)
    c_col = next((c for c in ['country_code', 'country'] if c in df.columns), None)
    if c_col and 'all' not in [x.lower() for x in target_countries]:
        df = df[df[c_col].isin(target_countries)]
        print(f'[FILTER] retained countries {target_countries}, rows: {len(df)}')
    if df.empty:
        raise ValueError('è¿‡æ»¤åŽæ— æœ‰æ•ˆæ•°æ®ï¼Œè¯·æ£€æŸ¥é…ç½®ä¸­çš„ country å‚æ•°ã€‚')

    if max_train_samples is not None and max_train_samples > 0 and len(df) > max_train_samples:
        # A head() sample silently selects the first asset in global data.
        # Keep small smoke runs representative across indices instead.
        rng = np.random.default_rng(20260720)
        groups = [g for _, g in df.groupby('asset_underlying', sort=True)]
        total = len(df)
        allocations = [max(1, int(round(max_train_samples * len(g) / total))) for g in groups]
        while sum(allocations) > max_train_samples:
            idx = int(np.argmax(allocations))
            if allocations[idx] <= 1:
                break
            allocations[idx] -= 1
        while sum(allocations) < max_train_samples:
            idx = int(np.argmin(allocations))
            allocations[idx] += 1
        pieces = []
        for g, n in zip(groups, allocations):
            n = min(n, len(g))
            pieces.append(g.iloc[rng.choice(len(g), size=n, replace=False)])
        df = pd.concat(pieces, ignore_index=True).sort_values(
            ['asset_underlying', 'start_date']
        ).copy()
        print(f'Small-sample training: keeping {len(df)} rows')

    # Keep temporary filtered files in this branch.  The frozen v1 data root
    # is read-only by convention and must never receive run artefacts.
    scratch_dir = project_root / 'Results' / '_scratch'
    scratch_dir.mkdir(parents=True, exist_ok=True)
    temp_csv_path = scratch_dir / f'temp_run_{asset_name}_{timestamp_str}.csv'
    df.to_csv(temp_csv_path, index=False)
    return base_path, temp_csv_path


def _quantile_bucket(series, quantiles=(0.33, 0.67), labels=('low', 'mid', 'high')):
    values = pd.to_numeric(series, errors='coerce')
    if values.notna().sum() < len(labels):
        return pd.Series([labels[1]] * len(series), index=series.index)
    cuts = values.quantile(list(quantiles)).to_list()
    if len(set(float(x) for x in cuts)) < len(cuts):
        ranks = values.rank(method='average', pct=True)
        return pd.cut(ranks, bins=[-0.01, *quantiles, 1.01], labels=labels).astype(str)
    return pd.cut(values, bins=[-float('inf'), *cuts, float('inf')], labels=labels).astype(str)


def _grouped_quantile_bucket(df, value_col, group_col, quantiles, labels):
    """Apply the same percentile labels within every asset or other group."""
    result = pd.Series(index=df.index, dtype='object')
    for _, group in df.groupby(group_col, sort=False):
        result.loc[group.index] = _quantile_bucket(
            group[value_col], quantiles=quantiles, labels=labels
        )
    return result


def _bounded_mean_one_weights(raw, floor, cap):
    """Scale positive weights to mean one while preserving strict bounds."""
    if not (0.0 < floor <= 1.0 <= cap):
        raise ValueError(
            f'regime weights require 0 < floor <= 1 <= cap; got floor={floor}, cap={cap}'
        )
    raw = pd.to_numeric(raw, errors='coerce').fillna(1.0).clip(lower=1e-12)

    low, high = 0.0, 1.0
    while float((raw * high).clip(lower=floor, upper=cap).mean()) < 1.0:
        high *= 2.0
        if high > 1e12:
            raise RuntimeError('could not normalize bounded regime weights')

    for _ in range(80):
        mid = 0.5 * (low + high)
        mean = float((raw * mid).clip(lower=floor, upper=cap).mean())
        if mean < 1.0:
            low = mid
        else:
            high = mid

    weights = (raw * (0.5 * (low + high))).clip(lower=floor, upper=cap)
    if abs(float(weights.mean()) - 1.0) > 1e-6:
        raise RuntimeError('bounded regime weights did not normalize to mean one')
    return weights


def _regime_weights_for_frame(df, cfg):
    """Compute strictly bounded regime weights for one configured frame."""
    parts = []

    asset_col = cfg.get('asset_column', 'asset_underlying')
    percentile_group_col = cfg.get('percentile_group_column', asset_col)
    grouped_percentiles = bool(cfg.get('normalize_features_within_group', False))
    if grouped_percentiles and percentile_group_col not in df.columns:
        raise ValueError(
            f'normalize_features_within_group requires column {percentile_group_col!r}'
        )

    def bucket(column, quantiles, labels):
        if grouped_percentiles:
            return _grouped_quantile_bucket(
                df, column, percentile_group_col, quantiles, labels
            )
        return _quantile_bucket(df[column], quantiles=quantiles, labels=labels)

    if bool(cfg.get('use_asset_bucket', True)) and asset_col in df.columns:
        parts.append(df[asset_col].astype(str).fillna('unknown'))

    vol_col = cfg.get('vol_column', 'hist_rv_60')
    if vol_col not in df.columns:
        vol_col = 'volatility'
    if vol_col in df.columns:
        parts.append(bucket(
            vol_col, (0.33, 0.67), ('vol_low', 'vol_mid', 'vol_high')
        ))

    trend_col = cfg.get('trend_column', 'hist_trend_60')
    if trend_col in df.columns:
        parts.append(bucket(
            trend_col, (0.33, 0.67), ('trend_down', 'trend_flat', 'trend_up')
        ))

    dd_col = cfg.get('drawdown_column', 'hist_max_drawdown_60')
    if dd_col in df.columns:
        stress_q = float(cfg.get('drawdown_stress_quantile', 0.25))
        parts.append(bucket(
            dd_col, (stress_q,), ('dd_stress', 'dd_normal')
        ))

    if not parts:
        return None, {'enabled': False, 'reason': 'no regime columns found'}

    regime = parts[0].astype(str)
    for p in parts[1:]:
        regime = regime + '|' + p.astype(str)

    freq = regime.value_counts(normalize=True)
    occupied = max(len(freq), 1)
    power = float(cfg.get('inverse_frequency_power', 1.0))
    if not (0.0 <= power <= 1.0):
        raise ValueError(f'inverse_frequency_power must be in [0, 1], got {power}')
    floor = float(cfg.get('weight_floor', 0.5))
    cap = float(cfg.get('weight_cap', 3.0))
    balance_group_col = cfg.get('balance_group_column')
    shrinkage = float(cfg.get('hierarchical_shrinkage_strength', 0.0))
    group_summaries = {}

    if balance_group_col:
        if balance_group_col not in df.columns:
            raise ValueError(f'balance_group_column {balance_group_col!r} not found')
        target = 1.0 / occupied
        weights = pd.Series(index=df.index, dtype=float)
        for group_name, group in df.groupby(balance_group_col, sort=True):
            group_regime = regime.loc[group.index]
            local_freq = group_regime.value_counts(normalize=True)
            group_n = float(len(group))

            def shrunk_probability(regime_name):
                local = float(local_freq.get(regime_name, 0.0))
                pooled = float(freq.get(regime_name, 0.0))
                if shrinkage <= 0.0:
                    return local
                return (group_n * local + shrinkage * pooled) / (group_n + shrinkage)

            raw = group_regime.map(
                lambda r: (target / max(shrunk_probability(r), 1e-12)) ** power
            ).astype(float)
            group_weights = _bounded_mean_one_weights(raw, floor, cap)
            weights.loc[group.index] = group_weights
            group_summaries[str(group_name)] = {
                'samples': int(group_n),
                'occupied_regimes': int(len(local_freq)),
                'min_weight': float(group_weights.min()),
                'median_weight': float(group_weights.median()),
                'max_weight': float(group_weights.max()),
                'mean_weight': float(group_weights.mean()),
            }
    else:
        raw = regime.map(
            lambda r: ((1.0 / occupied) / max(float(freq.get(r, 0.0)), 1e-12)) ** power
        ).astype(float)
        weights = _bounded_mean_one_weights(raw, floor, cap)

    summary = {
        'enabled': True,
        'scheme': (
            'group_normalized_hierarchical_tempered_balance'
            if balance_group_col else
            'strictly_bounded_inverse_frequency'
        ),
        'num_regimes': int(occupied),
        'inverse_frequency_power': power,
        'normalize_features_within_group': grouped_percentiles,
        'percentile_group_column': percentile_group_col if grouped_percentiles else None,
        'balance_group_column': balance_group_col,
        'hierarchical_shrinkage_strength': shrinkage,
        'weight_floor': floor,
        'weight_cap': cap,
        'mean_weight_after_normalization': float(weights.mean()),
        'min_weight': float(weights.min()),
        'median_weight': float(weights.median()),
        'max_weight': float(weights.max()),
        'top_regimes': freq.head(10).to_dict(),
        'bottom_regimes': freq.tail(10).to_dict(),
        'groups': group_summaries,
        'columns': {
            'asset': asset_col if asset_col in df.columns else None,
            'volatility': vol_col if vol_col in df.columns else None,
            'trend': trend_col if trend_col in df.columns else None,
            'drawdown': dd_col if dd_col in df.columns else None,
        }
    }
    return weights, summary


def compute_regime_sample_weights(csv_path, main_config):
    """Country-aware, strictly bounded weights for ex-ante market regimes."""
    cfg = main_config.get('regime_balance', {}) or {}
    if not bool(cfg.get('enabled', False)):
        return None, None

    df = pd.read_csv(csv_path)
    country_policies = cfg.get('country_policies', {}) or {}
    if not country_policies:
        weights, summary = _regime_weights_for_frame(df, cfg)
        return torch.tensor(weights.to_numpy(dtype='float32')), summary

    country_col = cfg.get('country_column')
    if not country_col:
        country_col = next((c for c in ['country_code', 'country'] if c in df.columns), None)
    if country_col is None or country_col not in df.columns:
        raise ValueError('country_policies require a valid country column')

    weights = pd.Series(1.0, index=df.index, dtype=float)
    country_summaries = {}
    for country, country_df in df.groupby(country_col, sort=True):
        policy = country_policies.get(str(country), country_policies.get('default', {})) or {}
        local_cfg = dict(cfg)
        local_cfg.update(policy)
        local_cfg.pop('country_policies', None)
        if not bool(local_cfg.get('enabled', False)):
            country_summaries[str(country)] = {
                'enabled': False,
                'samples': int(len(country_df)),
                'min_weight': 1.0,
                'median_weight': 1.0,
                'max_weight': 1.0,
            }
            continue
        country_weights, country_summary = _regime_weights_for_frame(country_df, local_cfg)
        weights.loc[country_df.index] = country_weights
        country_summary['samples'] = int(len(country_df))
        country_summaries[str(country)] = country_summary

    summary = {
        'enabled': True,
        'scheme': 'country_aware_strictly_bounded_inverse_frequency',
        'country_column': country_col,
        'mean_weight_after_normalization': float(weights.mean()),
        'min_weight': float(weights.min()),
        'median_weight': float(weights.median()),
        'max_weight': float(weights.max()),
        'countries': country_summaries,
        'num_regimes': int(sum(
            item.get('num_regimes', 0) for item in country_summaries.values()
        )),
    }
    return torch.tensor(weights.to_numpy(dtype='float32')), summary


def validate_effective_training_config(main_config, data_processor, data_info, variant):
    """Fail fast when a declared production feature is inactive or miswired."""
    errors = []
    checks = {}

    if variant in ('dlpm', 'dlpm_hybrid'):
        alpha = float(main_config.get('dlpm_alpha', 0.0))
        lp = float(main_config.get('dlpm_lploss', 0.0))
        checks['use_dlpm'] = bool(main_config.get('use_dlpm', False))
        checks['dlpm_alpha'] = alpha
        checks['dlpm_lploss'] = lp
        if not checks['use_dlpm']:
            errors.append(f'use_dlpm is disabled for variant {variant}')
        configured_eps_clip = main_config.get('dlpm_clamp_eps')
        checks['resolved_dlpm_clamp_eps'] = (
            float(configured_eps_clip)
            if configured_eps_clip is not None
            else float(main_config.get('unet_params', {}).get('output_scaling', 10.0))
        )
        if not 1.0 < alpha <= 2.0:
            errors.append(f'dlpm_alpha must be in (1, 2], got {alpha}')
        if not 0.0 < lp <= alpha:
            errors.append(f'dlpm_lploss must satisfy 0 < p <= alpha, got p={lp}, alpha={alpha}')
        outer = int(main_config.get('dlpm_monte_carlo_outer', 1))
        inner = int(main_config.get('dlpm_monte_carlo_inner', 1))
        reduction = str(main_config.get('dlpm_loss_monte_carlo', 'mean'))
        checks['dlpm_monte_carlo'] = {
            'outer': outer,
            'inner': inner,
            'reduction': reduction,
        }
        if outer < 1 or inner < 1:
            errors.append(f'DLPM Monte Carlo counts must be positive, got outer={outer}, inner={inner}')
        if reduction not in ('mean', 'median'):
            errors.append(f'Unsupported dlpm_loss_monte_carlo={reduction}')
        if reduction == 'median' and outer < 2:
            errors.append('median DLPM loss reduction requires dlpm_monte_carlo_outer >= 2')

    sequence_length = int(main_config.get('seq_length', 0))
    input_sequence_length = int(main_config.get('input_sequence_length', 0))
    checks['sequence_lengths'] = {
        'diffusion': sequence_length,
        'target': input_sequence_length,
        'processed': int(data_info.get('sequence_length', 0)),
    }
    if len(set(checks['sequence_lengths'].values())) != 1:
        errors.append(f"Sequence-length mismatch: {checks['sequence_lengths']}")
    checks['model_type'] = str(main_config.get('model_type', ''))
    if checks['model_type'] != 'unet':
        errors.append(f"Production training requires model_type='unet', got {checks['model_type']!r}")
    checks['enhanced_condition_network'] = bool(
        main_config.get('use_enhanced_condition_network', False)
    )
    if not checks['enhanced_condition_network']:
        errors.append('use_enhanced_condition_network is disabled')

    numerical_dim = int(data_info['condition_dim'] - 2)
    checks['numerical_condition_dim'] = numerical_dim
    checks['history_context_enabled'] = bool(main_config.get('use_history_context', False))
    checks['financial_regularizers_enabled'] = bool(main_config.get('use_financial_regularizers', False))
    checks['temporal_history_encoder_enabled'] = bool(
        main_config.get('cond_net_params', {}).get('temporal_history_encoder', False)
    )
    checks['regime_balance_enabled'] = bool(
        (main_config.get('regime_balance', {}) or {}).get('enabled', False)
    )
    checks['lr_warmup_ratio'] = float(main_config.get('warmup_ratio', 0.0))
    checks['financial_warmup_steps'] = int(main_config.get('financial_loss_warmup_steps', 0))
    checks['amp_enabled'] = bool(main_config.get('amp', False))

    if variant == 'dlpm_hybrid':
        for key in (
            'history_context_enabled',
            'financial_regularizers_enabled',
            'temporal_history_encoder_enabled',
        ):
            if not checks[key]:
                errors.append(f'{key} is disabled for the dlpm_hybrid production variant')

        weights = main_config.get('financial_loss_weights', {}) or {}
        active_weights = {k: float(v) for k, v in weights.items() if float(v) > 0.0}
        checks['active_financial_regularizers'] = active_weights
        if not active_weights:
            errors.append('financial regularizers are enabled but no positive weights are configured')

        aliases = {
            'returns_60': 'hist_returns_60',
            'returns_252': 'hist_returns_252',
            'vol_path_60': 'hist_vol_path_60',
        }
        layout = main_config.get('cond_net_params', {}).get('history_feature_layout', {}) or {}
        slices = getattr(data_processor, 'history_feature_slices', {}) or {}
        verified_layout = {}
        for alias, column in aliases.items():
            if alias not in layout:
                errors.append(f'temporal history layout is missing {alias}')
                continue
            if column not in slices:
                errors.append(f'processed history features are missing {column}')
                continue
            configured_start, configured_length = map(int, layout[alias])
            relative_start, actual_length = map(int, slices[column])
            actual_start = 5 + relative_start
            verified_layout[alias] = {
                'configured': [configured_start, configured_length],
                'actual': [actual_start, actual_length],
            }
            if (configured_start, configured_length) != (actual_start, actual_length):
                errors.append(
                    f'{alias} layout mismatch: configured '
                    f'[{configured_start}, {configured_length}], actual '
                    f'[{actual_start}, {actual_length}]'
                )
            if configured_start + configured_length > numerical_dim:
                errors.append(f'{alias} layout exceeds numerical condition dimension {numerical_dim}')
        checks['verified_history_layout'] = verified_layout

    if errors:
        raise ValueError('Effective training configuration failed preflight:\n- ' + '\n- '.join(errors))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, default='dlpm')
    parser.add_argument('--train_num_steps', type=int, default=None)
    parser.add_argument('--train_lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--skip_final_sample', action='store_true',
                        help='skip the expensive final sampling pass during small tuning runs')
    parser.add_argument('--checkpoint_every', type=int, default=500,
                        help='save resumable checkpoints every N steps (default: 500; use 0 to disable)')
    parser.add_argument('--sample_every', type=int, default=None,
                        help='generate intermediate samples every N steps; use 0 to disable')
    parser.add_argument('--underlying_asset', default=None)
    parser.add_argument('--countries', nargs='+', default=None)
    parser.add_argument('--experiment_tag', default=None,
                        help='Optional non-overwriting output folder under Diffusion_Comparison/<variant>/')
    parser.add_argument('--init_from_run', default=None,
                        help='Optional trained run folder to initialize model and condition network weights from')
    parser.add_argument('--resume_run', default=None,
                        help='Optional trained run folder containing checkpoints/model-M.pt to resume optimizer state')
    parser.add_argument('--resume_milestone', type=int, default=None,
                        help='Checkpoint milestone to resume from; omitted means latest available checkpoint')
    parser.add_argument('--config_overrides_json', default=None,
                        help='JSON object with temporary config overrides for sweep runs')
    parser.add_argument('--config_overrides_file', default=None,
                        help='Path to a JSON file with temporary config overrides')
    parser.add_argument('--seed', type=int, default=20260730,
                        help='Shared seed for controlled training comparisons')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    variant = args.variant
    training_start_time = time.time()
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    if variant == 'dlpm_hybrid':
        main_config = dict(HybridDLPMConfig.main_config)
    elif variant == 'dlpm':
        main_config = dict(DLPMConfig.main_config)
    else:
        main_config = dict(DDPMConfig.main_config)
    override_text = args.config_overrides_json
    if args.config_overrides_file:
        override_path = Path(args.config_overrides_file)
        if not override_path.is_absolute():
            override_path = (project_root / override_path).resolve()
        override_text = override_path.read_text(encoding='utf-8-sig')
    if override_text:
        try:
            overrides = json.loads(override_text)
        except json.JSONDecodeError:
            overrides = ast.literal_eval(override_text)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(main_config.get(key), dict):
                merged = dict(main_config[key])
                merged.update(value)
                main_config[key] = merged
            else:
                main_config[key] = value

    # Explicit CLI arguments have the highest precedence. This is important
    # for controlled gates that shorten a formal configuration without
    # silently inheriting its full training budget.
    if args.underlying_asset is not None:
        main_config['underlying_asset'] = args.underlying_asset
    if args.countries is not None:
        main_config['country'] = args.countries
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
    experiment_tag = safe_tag(args.experiment_tag)

    print('\n' + '=' * 60)
    print(f'[TRAIN] diffusion model | variant: {variant}')
    if variant in ('dlpm', 'dlpm_hybrid'):
        print(f"   DLPM alpha = {main_config['dlpm_alpha']} | Lp loss, p={main_config['dlpm_lploss']}")
    else:
        print(f"   DDPM loss_mode = {main_config['loss_mode']} | objective = {main_config['objective']}")
    print('=' * 60)

    # [1] æ•°æ®
    temp_csv_path = None
    try:
        data_t0 = time.time()
        base_path, temp_csv_path = load_and_filter_data(main_config, timestamp_str, args.max_train_samples)
        data_processor = DataProcessor(main_config)
        X_train, y_train, mask_train = data_processor.process_all_data(temp_csv_path)
        sample_weights, regime_balance_summary = compute_regime_sample_weights(temp_csv_path, main_config)
        if sample_weights is not None:
            if len(sample_weights) != len(X_train):
                raise ValueError(
                    f'regime sample weights length mismatch: {len(sample_weights)} vs {len(X_train)}'
                )
            print(
                'Regime-balanced loss enabled | '
                f"regimes={regime_balance_summary['num_regimes']} | "
                f"weight range={regime_balance_summary['min_weight']:.3f}-"
                f"{regime_balance_summary['max_weight']:.3f}"
            )
        data_seconds = time.time() - data_t0
        for name, tensor in [('æ¡ä»¶ç‰¹å¾(X)', X_train), ('ç›®æ ‡åºåˆ—(y)', y_train), ('æœ‰æ•ˆæ€§Mask', mask_train)]:
            if not torch.isfinite(tensor).all():
                bad = (~torch.isfinite(tensor)).sum().item()
                raise ValueError(f'æ•°æ®æº {name} å­˜åœ¨ {bad} ä¸ªéžæœ‰é™å€¼ï¼Œè¯·æ£€æŸ¥ DataProcessorã€‚')
        print('[OK] data source validation passed')
        data_info = {
            'source_file': str(base_path),
            'num_samples': len(X_train),
            'condition_dim': X_train.shape[-1],
            'sequence_length': y_train.shape[-1],
            'num_countries': data_processor.num_countries,
            'num_indices': data_processor.num_indices,
            'regime_balance': regime_balance_summary,
        }
        preflight_checks = validate_effective_training_config(
            main_config, data_processor, data_info, variant
        )
        print('[OK] effective config preflight passed')
    except Exception as e:
        if temp_csv_path and temp_csv_path.exists():
            temp_csv_path.unlink()
        safe_error = repr(e).encode('ascii', 'backslashreplace').decode('ascii')
        print(f'[ERROR] data processing failed: {safe_error}')
        traceback.print_exc()
        sys.exit(1)

    # [2] æ¨¡åž‹
    device = pick_device()
    print(f'The device is {device}')

    cond_net_params = main_config.get('cond_net_params', {})
    cond_net_params['numerical_input_dim'] = int(data_info['condition_dim'] - 2)
    condition_network = EnhancedConditionNetwork(
        num_countries=max(data_info['num_countries'] + 5, 20),
        num_indices=max(data_info['num_indices'] + 10, 100),
        **cond_net_params
    ).to(device)

    model = Unet1D(
        cond_dim=cond_net_params.get('output_dim', 128),
        **main_config.get('unet_params', {})
    ).to(device)

    init_from_run = Path(args.init_from_run) if args.init_from_run else None
    if init_from_run is not None:
        if not init_from_run.is_absolute():
            init_from_run = (project_root / init_from_run).resolve()
        model_path = init_from_run / 'unet_conditional_model_ema.pth'
        cond_path = init_from_run / 'condition_network_ema.pth'
        if not model_path.exists():
            model_path = init_from_run / 'unet_conditional_model.pth'
        if not cond_path.exists():
            cond_path = init_from_run / 'condition_network.pth'
        if not model_path.exists() or not cond_path.exists():
            raise FileNotFoundError(f'init_from_run missing model weights: {init_from_run}')
        model.load_state_dict(torch.load(model_path, map_location=device))
        condition_network.load_state_dict(torch.load(cond_path, map_location=device))
        print(f'Initialized weights from: {init_from_run}')

    if variant in ('dlpm', 'dlpm_hybrid'):
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

    # [3] è®­ç»ƒ
    resume_run_path = Path(args.resume_run) if args.resume_run else None
    output_name = experiment_tag or (resume_run_path.name if resume_run_path is not None else asset_name)
    if resume_run_path is not None and experiment_tag is None:
        results_root = resume_run_path
        if not results_root.is_absolute():
            results_root = (project_root / results_root).resolve()
    else:
        results_root = pp.Model_Results_DIR / 'Diffusion_Comparison' / variant / output_name
    reproducibility_payload = {
        'variant': variant,
        'seed': int(args.seed),
        'max_train_samples': args.max_train_samples,
        'effective_config': main_config,
    }
    fingerprint = config_fingerprint(reproducibility_payload)
    run_manifest_path = results_root / 'run_manifest.json'
    previous_manifest = {}
    if run_manifest_path.exists() and not args.resume_run:
        raise FileExistsError(
            f'Run folder already contains a manifest: {results_root}. '
            'Choose a new experiment tag or use --resume_run.'
        )
    if args.resume_run and run_manifest_path.exists():
        previous_manifest = json.loads(run_manifest_path.read_text(encoding='utf-8'))
        previous_fingerprint = previous_manifest.get('config_fingerprint')
        if previous_fingerprint and previous_fingerprint != fingerprint:
            raise ValueError(
                'Resume configuration differs from the original run. '
                'Use the same seed, sample count, architecture, loss, and training settings.'
            )
    atomic_write_json(run_manifest_path, {
        **reproducibility_payload,
        'config_fingerprint': fingerprint,
        'data_info': data_info,
        'preflight_checks': preflight_checks,
        'checkpoint_every': (
            args.checkpoint_every if args.checkpoint_every and args.checkpoint_every > 0 else None
        ),
        'cumulative_training_seconds': float(
            previous_manifest.get('cumulative_training_seconds', 0.0)
        ),
        'status': 'initialized',
    })
    dataset = Dataset1D(y_train, X_train, mask_train, sample_weights)
    train_num_steps = main_config.get('train_num_steps', 4000)
    save_and_sample_every = train_num_steps + 1 if args.skip_final_sample else train_num_steps
    sample_every = args.sample_every
    if sample_every == 0:
        sample_every = None
    elif sample_every is None:
        sample_every = None if args.skip_final_sample else save_and_sample_every
    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=dataset,
        results_folder=str(results_root / 'checkpoints'),
        train_batch_size=main_config.get('train_batch_size', 64),
        train_lr=main_config.get('train_lr', 1e-4),
        train_num_steps=train_num_steps,
        warmup_ratio=main_config.get('warmup_ratio', 0.0),
        gradient_accumulate_every=1,
        ema_decay=main_config.get('ema_decay', 0.995),
        amp=main_config.get('amp', False),
        save_and_sample_every=save_and_sample_every,  # tuning can skip the expensive final sample
        checkpoint_every=args.checkpoint_every if args.checkpoint_every and args.checkpoint_every > 0 else None,
        sample_every=sample_every,
    )
    if args.resume_run:
        resume_run = Path(args.resume_run)
        if not resume_run.is_absolute():
            resume_run = (project_root / resume_run).resolve()
        resume_source = resume_run / 'checkpoints'
        resume_milestone = args.resume_milestone
        if resume_milestone is None:
            candidates = []
            for checkpoint_path in resume_source.glob('model-*.pt'):
                match = re.fullmatch(r'model-(\d+)\.pt', checkpoint_path.name)
                if match:
                    candidates.append(int(match.group(1)))
            if not candidates:
                raise FileNotFoundError(f'No numeric checkpoints found in {resume_source}')
            resume_milestone = max(candidates)
        trainer.load(resume_milestone, source_folder=resume_source)
        print(f'Resumed optimizer/model/RNG state from: {resume_source / f"model-{resume_milestone}.pt"}')
    start_step = int(trainer.step)
    train_t0 = time.time()
    try:
        trainer.train()
    except KeyboardInterrupt:
        if trainer.step > 0:
            trainer.save(trainer.step)
            print(f'Interrupted safely; checkpoint saved at step {trainer.step}.')
        interrupted_manifest = json.loads(run_manifest_path.read_text(encoding='utf-8'))
        interrupted_manifest.update({'status': 'interrupted', 'last_completed_step': int(trainer.step)})
        atomic_write_json(run_manifest_path, interrupted_manifest)
        raise
    train_seconds = time.time() - train_t0
    steps_this_invocation = int(trainer.step) - start_step
    cumulative_training_seconds = (
        float(previous_manifest.get('cumulative_training_seconds', 0.0))
        + train_seconds
    )

    # [4] ä¿å­˜äº§ç‰©ï¼ˆåŽŸå§‹ + EMA ä¸¤å¥—æƒé‡ï¼‰
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
        'loss_components': results_root / 'loss_components.json',
        'loss_curve': results_root / 'loss_curve.png',
    }
    torch.save(model.state_dict(), paths['model'])
    torch.save(condition_network.state_dict(), paths['cond_net'])
    torch.save(ema_diffusion.model.state_dict(), paths['model_ema'])
    if ema_diffusion.condition_network is not None:
        torch.save(ema_diffusion.condition_network.state_dict(), paths['cond_net_ema'])
    if joblib is not None:
        joblib.dump(data_processor, paths['processor'])
    else:
        with open(paths['processor'], 'wb') as handle:
            pickle.dump(data_processor, handle)

    serializable_cfg = {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in main_config.items() if not isinstance(v, dict)}
    serializable_cfg['unet_params'] = {k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in main_config.get('unet_params', {}).items()}
    serializable_cfg['cond_net_params'] = dict(cond_net_params)
    for dict_key in ['financial_loss_weights', 'history_feature_lengths', 'regime_balance']:
        if dict_key in main_config:
            serializable_cfg[dict_key] = dict(main_config[dict_key])
    serializable_cfg['variant'] = variant
    serializable_cfg['seed'] = int(args.seed)
    serializable_cfg['experiment_tag'] = experiment_tag
    serializable_cfg['checkpoint_every'] = (
        args.checkpoint_every if args.checkpoint_every and args.checkpoint_every > 0 else None
    )
    serializable_cfg['resumed_from_run'] = str(resume_run_path) if resume_run_path is not None else None
    serializable_cfg['resumed_from_milestone'] = (
        int(resume_milestone) if args.resume_run else None
    )
    serializable_cfg['data_split_protocol'] = {
        'fit_split': 'purged_train',
        'checkpoint_selection_split': 'purged_validation_external',
        'final_evaluation_split': 'purged_test_external',
        'validation_used_in_optimizer': False,
        'test_used_in_optimizer': False,
        'embargo_trading_days': 20,
        'sliding_windows_are_independent': False,
    }
    serializable_cfg['init_from_run'] = str(init_from_run) if init_from_run is not None else None
    serializable_cfg['output_folder'] = output_name
    serializable_cfg['data_info'] = data_info
    serializable_cfg['preflight_checks'] = preflight_checks
    serializable_cfg['timing'] = {
        'data_processing_seconds': data_seconds,
        'training_seconds': train_seconds,
        'training_steps_this_invocation': steps_this_invocation,
        'cumulative_training_seconds': cumulative_training_seconds,
        'total_seconds': time.time() - training_start_time,
        'seconds_per_step_this_invocation': (
            train_seconds / steps_this_invocation if steps_this_invocation > 0 else None
        ),
    }
    with open(paths['config'], 'w') as f:
        json.dump(serializable_cfg, f, indent=2, ensure_ascii=False)
    with open(paths['loss_history'], 'w') as f:
        json.dump(trainer.loss_history, f)
    with open(paths['loss_components'], 'w') as f:
        json.dump(trainer.loss_component_history, f, indent=2)

    completed_manifest = json.loads(run_manifest_path.read_text(encoding='utf-8'))
    completed_manifest.update({
        'status': 'completed',
        'last_completed_step': int(trainer.step),
        'cumulative_training_seconds': cumulative_training_seconds,
        'timing': serializable_cfg['timing'],
    })
    atomic_write_json(run_manifest_path, completed_manifest)

    if trainer.loss_history and plt is not None:
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
    print(f"[OK] variant '{variant}' training complete | steps: {trainer.step} | "
          f"final_loss: {trainer.loss_history[-1] if trainer.loss_history else float('nan'):.5f} | "
          f'elapsed_min: {duration/60:.1f}')
    print(f'   output_dir: {results_root}')
    print('=' * 60)


if __name__ == '__main__':
    main()

