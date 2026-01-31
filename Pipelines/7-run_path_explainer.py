# run_path_explainer.py
# 
# ==========================================================
#             模型验证 (Path Explainer) 一键启动
# ==========================================================
# (已修改为多任务批处理)
#
import sys
import traceback
from pathlib import Path

# --- Path Setup ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

# 导入 "设定集" 和 "引擎"
import Config.path_explainer_config as path_explainer_config
from Explainer.path_explainer_engine import PathExplainerEngine


if __name__ == '__main__':
    
    # --- 1. 在这里选择你要运行的作业名称 ---
    # (必须与 path_explainer_config.py 中的键名完全一致)
    
    # 修改：从单个作业改为作业列表
    JOBS_TO_VALIDATE = [
        #'validate_gbm',
        #'validate_garch',
        'validate_dlpm',
        'validate_ddpm'
        ]
    # -------------------------------------
    
    # --- 2. 定义运行时环境 ---
    # 所有验证都针对同一个目标资产
    TARGET_ASSET = 'CSI1000'
    # 所有验证都使用相同的分析样本量
    #INDICES_TO_ANALYZE = list(range(20)) 
    INDICES_TO_ANALYZE = list(range(827)) # 分析全部
    # -------------------------------------

    print(f"--- 启动器: 准备执行 {len(JOBS_TO_VALIDATE)} 个模型验证作业 ---")
    print(f"--- 资产: {TARGET_ASSET}, 分析 {len(INDICES_TO_ANALYZE)} 个条件 ---")

    # --- 3. 循环运行所有验证作业 ---
    failed_jobs = []
    for RUN_JOB_NAME in JOBS_TO_VALIDATE:
        
        print(f"\n==========================================================")
        print(f"🏁 开始验证作业: '{RUN_JOB_NAME}' (资产: {TARGET_ASSET})")
        print(f"==========================================================")

        try:
            # 3.1 获取作业参数
            job_spec = path_explainer_config.PATH_JOBS[RUN_JOB_NAME]

            # 3.2 运行分析
            # 1. 初始化引擎
            engine = PathExplainerEngine(
                asset_name=TARGET_ASSET,
                job_spec=job_spec
            )
            
            # 2. 加载数据
            engine.load_data()
            
            # 3. 运行分析
            all_results = engine.run_analysis(INDICES_TO_ANALYZE)
            
            # 4. 生成报告
            engine.generate_report(all_results)
            
            print(f"✅ 验证作业 '{RUN_JOB_NAME}' (资产: {TARGET_ASSET}) 执行完毕。")
            
        except FileNotFoundError as e:
            print(f"\n❌ 启动器错误：找不到必需的文件。")
            print(f"  - 详情: {e}")
            failed_jobs.append(RUN_JOB_NAME)
        except KeyError as e:
            print(f"\n❌ 启动器错误：在 .csv 或配置中找不到指定的键。")
            print(f"  - 详情: 找不到键 '{e}'")
            failed_jobs.append(RUN_JOB_NAME)
        except Exception as e:
            print(f"\n❌ 启动器发生未知错误: {e}")
            traceback.print_exc()
            failed_jobs.append(RUN_JOB_NAME)

    # --- 4. 最终总结 ---
    print(f"\n==========================================================")
    print(f"✅ 所有验证作业执行完毕。")
    if failed_jobs:
        print(f"❌ 失败的作业: {failed_jobs}")
    else:
        print(f"🎉 所有作业均已成功完成。")
    print(f"==========================================================")