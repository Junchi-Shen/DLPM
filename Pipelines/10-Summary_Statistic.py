# 10-Summary_Statistic.py
#
# ==========================================================
#   四模型对比表：MC(GBM) / GARCH / DDPM / DLPM
#   指标：1～5 阶矩、KS、AD、R²、推土机距离
# ==========================================================
# 读取各模型 Path Explainer 已生成的 detailed_statistics.csv，
# 按条件求平均后汇总为一张表，输出 CSV 与 Markdown。
# 不修改 Explainer 或 7 号管道，仅读取既有报告目录。
#
import sys
import os
from pathlib import Path
from datetime import datetime

# --- Path Setup ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

import pandas as pd
import Config.path_explainer_config as path_explainer_config
import Project_Path as pp


# 与 path_explainer_engine 中一致的 job_name_safe 规则
def _job_name_safe(display_name):
    return display_name.replace(' ', '_').replace('(', '').replace(')', '')


# 需要的指标列（detailed_statistics.csv 中的列名）-> 输出列名
COMPARISON_COLUMNS = [
    ('mean_diff', 'moment1_diff_mean'),       # 1 阶矩差
    ('std_diff', 'moment2_diff_mean'),       # 2 阶矩差
    ('skew_diff', 'moment3_diff_mean'),       # 3 阶矩差
    ('kurt_diff', 'moment4_diff_mean'),       # 4 阶矩差
    ('moment5_diff', 'moment5_diff_mean'),    # 5 阶矩差（若 CSV 未含则填 NaN）
    ('ks_statistic', 'ks_statistic_mean'),   # KS
    ('ad_statistic', 'ad_statistic_mean'),   # AD
    ('qq_r_squared', 'qq_r_squared_mean'),   # R²
    ('wasserstein_distance', 'wasserstein_distance_mean'),  # 推土机距离
]


def _find_latest_report_dir(base_dir, job_name_safe):
    """在 base_dir / job_name_safe 下找最新的 *_validation_report 目录"""
    job_dir = base_dir / job_name_safe
    if not job_dir.exists():
        return None
    report_dirs = list(job_dir.glob("*_validation_report"))
    if not report_dirs:
        return None
    latest = max(report_dirs, key=os.path.getmtime)
    return latest


def _read_and_aggregate(detailed_csv_path):
    """
    读 detailed_statistics.csv，对 COMPARISON_COLUMNS 中的列求 mean。
    缺失列填 NaN。返回 dict：{ output_col_name: mean_value }
    """
    if not detailed_csv_path.exists():
        return None
    df = pd.read_csv(detailed_csv_path, encoding='utf-8-sig')
    row = {}
    for col_in_csv, col_out in COMPARISON_COLUMNS:
        if col_in_csv in df.columns:
            row[col_out] = df[col_in_csv].dropna().mean()
        else:
            row[col_out] = float('nan')
    return row


def generate_four_model_comparison(asset_name, output_dir=None, jobs=None):
    """
    生成四模型对比表（CSV + Markdown）。

    Parameters
    ----------
    asset_name : str
        标的资产名，如 'CSI1000'，用于定位报告根目录。
    output_dir : pathlib.Path or None
        输出目录；若为 None，则使用 Report_Results_DIR / Path_Generator_Report / asset_name。
    jobs : list of str or None
        要汇总的作业名列表；若为 None，则使用 ['validate_gbm','validate_garch','validate_ddpm','validate_dlpm']。
    """
    if output_dir is None:
        output_dir = getattr(pp, "Report_Results_DIR") / "Path_Generator_Report" / asset_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if jobs is None:
        jobs = ['validate_gbm', 'validate_garch', 'validate_ddpm', 'validate_dlpm']

    base_dir = getattr(pp, "Report_Results_DIR") / "Path_Generator_Report" / asset_name
    if not base_dir.exists():
        print(f"⚠️ 报告根目录不存在: {base_dir}")
        return

    rows = []
    for job_name in jobs:
        if job_name not in path_explainer_config.PATH_JOBS:
            print(f"⚠️ 跳过未知作业: {job_name}")
            continue
        spec = path_explainer_config.PATH_JOBS[job_name]
        display_name = spec['display_name']
        job_name_safe = _job_name_safe(display_name)
        report_dir = _find_latest_report_dir(base_dir, job_name_safe)
        if report_dir is None:
            print(f"⚠️ 未找到 {display_name} 的报告目录: {base_dir / job_name_safe}")
            rows.append({'model': display_name, **{col_out: float('nan') for _, col_out in COMPARISON_COLUMNS}})
            continue
        csv_path = report_dir / 'detailed_statistics.csv'
        agg = _read_and_aggregate(csv_path)
        if agg is None:
            print(f"⚠️ 无法读取: {csv_path}")
            rows.append({'model': display_name, **{col_out: float('nan') for _, col_out in COMPARISON_COLUMNS}})
            continue
        rows.append({'model': display_name, **agg})

    if not rows:
        print("❌ 没有可汇总的模型结果。")
        return

    df = pd.DataFrame(rows)
    # 列顺序：model, moment1..moment5, ks, ad, r2, wasserstein
    col_order = ['model'] + [col_out for _, col_out in COMPARISON_COLUMNS]
    df = df[[c for c in col_order if c in df.columns]]

    csv_path = output_dir / 'four_model_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 四模型对比表 CSV: {csv_path}")

    md_path = output_dir / 'four_model_comparison.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 四模型对比表 (1～5 阶矩 / KS / AD / R² / 推土机距离)\n\n")
        f.write(f"**标的资产**: {asset_name}\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".6f"))
    print(f"✅ 四模型对比表 Markdown: {md_path}")
    return df


if __name__ == '__main__':
    TARGET_ASSET = 'CSI1000'
    JOBS = ['validate_gbm', 'validate_garch', 'validate_ddpm', 'validate_dlpm']

    print("--- 10-Summary_Statistic: 四模型对比表 ---")
    print(f"资产: {TARGET_ASSET}, 作业: {JOBS}")
    generate_four_model_comparison(asset_name=TARGET_ASSET, jobs=JOBS)
    print("--- 完成 ---")
