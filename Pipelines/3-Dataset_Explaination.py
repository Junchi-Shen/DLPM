# dataset_explainer.py
# 
# ==========================================================
#              数据分析器 (Data Explainer) 一键启动
# ==========================================================
# 
# (已修改为支持批量执行多个分析作业)
#
import sys
import traceback
from pathlib import Path

# --- 路径设置 ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))


# --- 导入 "设定集" 和 "引擎" ---
try:
    # 假设你的配置文件在 Config 目录下
    from Config.data_explainer_config import DATA_JOBS 
    # 假设你的引擎在 Explainer 目录下
    from Explainer.data_explainer_engine import DataExplainerEngine
except ImportError as e:
    print(f"❌ 启动器错误：无法导入必要的模块。请检查您的文件结构和导入路径。")
    print(f"  错误详情: {e}")
    sys.exit(1)

if __name__ == '__main__':
    
    # --- 1. 在这里定义你要批量运行的作业名称列表 ---
    # (名称必须与 data_explainer_config.py 中的键名完全一致)
    JOBS_TO_RUN = [
        'explain_train_data', 
        #'explain_val_data',   # 通常验证集也需要分析
        'explain_test_data'
    ]
    # ----------------------------------------------------
    
    # --- 2. 定义运行时环境 (可以对所有作业通用) ---
    TARGET_ASSET = 'all' 
    # -----------------------------------------------

    print(f"--- 启动器: 准备批量执行数据分析作业 ---")
    print(f"--- 资产: {TARGET_ASSET} ---")
    print(f"--- 作业列表: {JOBS_TO_RUN} ---")

    # --- 3. 循环执行每个作业 ---
    for run_job_name in JOBS_TO_RUN:
        
        print(f"\n H{'='*15} 开始执行作业: {run_job_name} H{'='*15}")
        
        try:
            # --- 3a. 获取当前作业的参数 ---
            job_spec = DATA_JOBS[run_job_name]
            
            # --- 3b. 运行分析 (将原步骤4放入循环内) ---
            # 1. 初始化引擎
            print(f"   初始化引擎...")
            engine = DataExplainerEngine(
                asset_name=TARGET_ASSET,
                job_spec=job_spec
            )
            
            # 2. 加载数据
            print(f"   加载数据...")
            engine.load_data()
            
            # 3. 运行分析并生成报告
            print(f"   执行分析...")
            engine.run_analysis()
            
            print(f" V 作业 '{run_job_name}' 执行成功 V ")
            
        except FileNotFoundError as e:
            print(f"\n❌ 作业 '{run_job_name}' 失败：找不到必需的文件。")
            print(f"  - 详情: {e}")
        except KeyError as e:
            print(f"\n❌ 作业 '{run_job_name}' 失败：在 .csv 或配置中找不到指定的键。")
            print(f"  - 详情: 找不到键 '{e}'")
        except Exception as e:
            print(f"\n❌ 作业 '{run_job_name}' 失败：发生未知错误。")
            print(f"  - 详情: {e}")
            traceback.print_exc() # 打印详细错误信息有助于调试
            
        print(f" H{'='*40}") # 分隔符

    print("\n✅ 所有指定的分析作业已尝试执行完毕。")