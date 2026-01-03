# Pipelines/run_garch_fitting.py
#
# 一键执行 GARCH 参数拟合并保存 JSON 文件。
# (已修正：所有任务均使用训练集数据)

import sys
import os
import json
from pathlib import Path

# --- 1. 项目路径设置 ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

try:
    import Project_Path as pp
    # (修改) 导入 *唯一的* 模板
    from Config.garch_fitter_config import GARCH_FIT_TEMPLATE
    from Model.Garch_Model.Garch_fitter import fit_and_save_garch_params
except ImportError as e:
    print(f"❌ 启动器错误：导入失败: {e}")
    print("  请确保 Project_Path.py, Config/garch_fitter_config.py, 和 Model/Garch_Model/garch_fitter.py 文件存在。")
    sys.exit(1)


if __name__ == '__main__':

    # --- !! 1. 在这里输入你要拟合的标的 !! ---
    # (所有资产都将使用 *训练集* 数据进行拟合)
    ASSETS_TO_FIT = [
        'CSI1000', #
        'CSI300',  #
        # 'CSI500', 
        # ... 在这里添加更多资产
    ]
    # ----------------------------------------------

    print("--- GARCH 拟合启动器 ---")
    
    # 将列表合并为待办任务
    jobs_to_run = []
    for asset in ASSETS_TO_FIT:
        jobs_to_run.append({
            "asset_to_fit": asset,
            "template": GARCH_FIT_TEMPLATE # 修正：所有任务都使用唯一的训练集模板
        })

    if not jobs_to_run:
        print("⚠️ 警告: 任务列表 (ASSETS_TO_FIT) 为空。")
        sys.exit(0)

    print(f"🔍 找到 {len(jobs_to_run)} 个拟合作业。将 *全部* 使用训练集数据。")

    # --- 2. 循环执行所有作业 ---
    for job in jobs_to_run:
        asset_name = job["asset_to_fit"]
        job_name = f"fit_{asset_name.lower()}" # 动态生成作业名
        
        print(f"\n==========================================================")
        print(f"🏁 开始执行作业: '{job_name}' (资产: {asset_name})")
        print(f"==========================================================")
        
        try:
            # --- 3. 动态构建作业规范(Spec) ---
            # 复制模板并添加特定资产名称
            spec = job["template"].copy()
            spec["asset_to_fit"] = asset_name

            # --- 4. 构建路径 ---
            data_dir = getattr(pp, spec['data_source_dir_key']) #
            data_path = data_dir / spec['central_data_file'] #
            
            base_output_dir = getattr(pp, spec['output_dir_key']) #
            garch_subfolder = spec['output_subfolder'] #
            asset_subfolder = spec['asset_to_fit']
            
            # 最终路径: .../Model_Results_DIR/Garch_Fit_Results/CSI1000/garch_params.json
            output_path = base_output_dir / garch_subfolder / asset_subfolder / spec['output_filename'] #
            
            # --- 5. 调用库函数 ---
            garch_fit, params = fit_and_save_garch_params(data_path, output_path, spec)
            
            # --- 6. 打印结果 ---
            print("\n--- GARCH(1,1) 模型拟合摘要 ---")
            print(garch_fit.summary())
            
            print("\n--- 已保存的参数 (JSON) ---")
            print(json.dumps(params, indent=4))
            
            print(f"\n✅ 作业 '{job_name}' 成功完成。")

        except Exception as e:
            print(f"\n❌ 作业 '{job_name}' 执行失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n==========================================================")
    print("🎉 GARCH 拟合启动器执行完毕。")
    print(f"==========================================================")