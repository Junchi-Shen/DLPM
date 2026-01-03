# Pipelines/run_generation.py
# (已修改为多任务批处理)

import sys
import traceback
import torch
from pathlib import Path

# --- Path Setup ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

# --- Imports ---
try:
    import Config.generator_config as generator_config
    from Generator.path_generator_engine import PathGeneratorEngine
except ImportError as e: print(f"❌ 启动器错误：导入失败: {e}"); sys.exit(1)


if __name__ == '__main__':

    # --- 1. 定义所有要运行的作业 ---
    # (job_name 必须匹配 generator_config.py)
    # (asset 必须是 'all' 或 'CSI1000' 等具体资产名)
    JOBS_TO_RUN = [
        {'job_name': 'GARCH', 'asset': 'CSI1000'},
        {'job_name': 'GBM',  'asset': 'CSI1000'},
        {'job_name': 'UNet', 'asset': 'CSI1000'}
    ]
    # -------------------------------

    # --- 2. 设置全局生成参数 ---
    # 所有作业将使用相同的路径数量
    NUM_PATHS_TO_GENERATE = 1024 # <-- 在这里修改为你想要的值
    # ------------------------------------

    # --- 3. 定义环境 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # -------------------------

    print(f"--- 路径生成器启动器 (多任务模式) ---")
    num_paths_str = f"{NUM_PATHS_TO_GENERATE:,}" if NUM_PATHS_TO_GENERATE is not None else "默认值"
    print(f"--- 将为 {len(JOBS_TO_RUN)} 个作业生成 {num_paths_str} 条路径, 设备: {DEVICE} ---")

    # --- 4. 循环运行所有作业 ---
    failed_jobs = []
    for job in JOBS_TO_RUN:
        RUN_JOB_NAME = job['job_name']
        TARGET_ASSET = job['asset']
        
        print(f"\n==========================================================")
        print(f"🏁 开始执行作业: '{RUN_JOB_NAME}' (资产: {TARGET_ASSET})")
        print(f"==========================================================")

        try:
            # 4.1 获取作业配置
            job_spec = generator_config.GENERATOR_JOBS[RUN_JOB_NAME]
            if 'job_name' not in job_spec: job_spec['job_name'] = RUN_JOB_NAME

            # 4.2 运行生成
            engine = PathGeneratorEngine(
                asset_name=TARGET_ASSET,
                job_spec=job_spec,
                device=DEVICE,
                num_paths_override=NUM_PATHS_TO_GENERATE # 传递覆盖值
            )
            engine.run()
            print(f"✅ 作业 '{RUN_JOB_NAME}' (资产: {TARGET_ASSET}) 执行完毕。")

        except Exception as e: # 捕获错误并继续
            print(f"\n❌ 启动器在运行作业 '{RUN_JOB_NAME}' 时遇到错误。")
            print(f"   错误详情: {e}")
            traceback.print_exc() # 打印完整堆栈信息
            failed_jobs.append(RUN_JOB_NAME)
    
    # --- 5. 最终总结 ---
    print(f"\n==========================================================")
    print(f"✅ 所有生成作业执行完毕。")
    if failed_jobs:
        print(f"❌ 失败的作业: {failed_jobs}")
    else:
        print(f"🎉 所有作业均已成功完成。")
    print(f"==========================================================")