# Pipelines/run_garch_fitting.py
#
# 一键执行 GARCH 参数拟合并保存 JSON 文件。
# (已修正：所有任务均使用训练集数据)

# Pipelines/5-Run_Garch_Fitting.py
# 升级版：自动扫描全市场资产并执行一键拟合

import sys
import os
import json
import pandas as pd
from pathlib import Path

# --- 1. 项目路径设置 ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

try:
    import Project_Path as pp
    from Config.garch_fitter_config import GARCH_FIT_TEMPLATE
    from Model.Garch_Model.Garch_fitter import fit_and_save_garch_params
except ImportError as e:
    print(f"❌ 启动器错误：导入失败: {e}")
    sys.exit(1)

if __name__ == '__main__':
    print("--- 🚀 GARCH 全自动全市场拟合启动器 ---")
    
    # --- 2. 自动探测数据集资产 ---
    data_dir = getattr(pp, GARCH_FIT_TEMPLATE['data_source_dir_key'])
    data_path = data_dir / GARCH_FIT_TEMPLATE['central_data_file']
    
    print(f"🔍 正在从 {data_path.name} 扫描可用资产...")
    try:
        # 只读取资产列以节省内存
        full_df = pd.read_csv(data_path, usecols=[GARCH_FIT_TEMPLATE['filter_column']])
        # 自动获取所有标的名
        ALL_ASSETS = full_df[GARCH_FIT_TEMPLATE['filter_column']].unique().tolist()
        print(f"✅ 扫描完成。检测到 {len(ALL_ASSETS)} 个标的: {ALL_ASSETS}")
    except Exception as e:
        print(f"❌ 扫描数据集失败: {e}")
        sys.exit(1)

    # --- 3. 循环执行全量拟合 ---
    success_count = 0
    fail_assets = []

    for asset_name in ALL_ASSETS:
        job_name = f"fit_{asset_name.lower()}"
        
        print(f"\n" + "="*60)
        print(f"🏁 正在为标的 [{asset_name}] 执行 GARCH 拟合...")
        print("="*60)
        
        try:
            # 构建 Spec
            spec = GARCH_FIT_TEMPLATE.copy()
            spec["asset_to_fit"] = asset_name

            # 构建输出路径
            base_output_dir = getattr(pp, spec['output_dir_key'])
            output_path = base_output_dir / spec['output_subfolder'] / asset_name / spec['output_filename']
            
            # 执行拟合
            # 内部逻辑已在 Garch_fitter.py 中对齐
            garch_fit, params = fit_and_save_garch_params(data_path, output_path, spec)
            
            print(f"   📈 拟合摘要: α+β = {params['alpha'] + params['beta']:.4f}")
            print(f"   ✅ 参数已对齐至: {output_path.parent.name}/{output_path.name}")
            success_count += 1

        except Exception as e:
            print(f"❌ 标的 '{asset_name}' 拟合失败: {e}")
            fail_assets.append(asset_name)

    # --- 4. 最终总结 ---
    print(f"\n" + "="*60)
    print(f"🎉 GARCH 全量拟合任务结束！")
    print(f"📊 成功: {success_count} | 失败: {len(fail_assets)}")
    if fail_assets:
        print(f"⚠️ 失败名单: {fail_assets}")
    print(f"==========================================================")