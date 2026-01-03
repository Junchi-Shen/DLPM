import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 设置路径
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

from Project_Path import  Trainning_DATA_DIR, Testing_DATA_DIR

def merge_csv_files_in_directory(target_dir, output_filename="merged_data.csv"):
    """
    遍历指定目录，合并所有子文件夹中的CSV文件，输出到该目录下
    
    参数:
    target_dir: str/Path, 要遍历的目录
    output_filename: str, 输出文件名
    
    返回:
    pandas.DataFrame: 合并后的数据框
    """
    target_path = Path(target_dir)
    
    if not target_path.exists():
        print(f"错误: 目录 {target_path} 不存在")
        return pd.DataFrame()
    
    print(f"正在处理目录: {target_path}")
    
    # 输出文件的完整路径
    output_path = target_path / output_filename
    
    all_dataframes = []
    
    # 遍历所有子目录
    for subfolder in target_path.iterdir():
        if not subfolder.is_dir():
            continue
            
        subfolder_name = subfolder.name
        print(f"  处理子文件夹: {subfolder_name}")
        
        # 查找子文件夹中的所有CSV文件
        csv_files = list(subfolder.glob("*.csv"))
        
        if not csv_files:
            print(f"    子文件夹 {subfolder_name} 中未找到CSV文件")
            continue
            
        # 读取并合并该子文件夹中的所有CSV文件
        subfolder_dfs = []
        for csv_file in csv_files:
            # 跳过输出文件本身（避免循环包含）
            if csv_file.resolve() == output_path.resolve():
                print(f"    跳过输出文件: {csv_file.name}")
                continue
                
            try:
                df = pd.read_csv(csv_file)
                
                # 移除不需要的标识列（如果存在的话）
                columns_to_remove = ['data_type', 'source_file', 'parent_directory', 'source_folder']
                for col in columns_to_remove:
                    if col in df.columns:
                        df = df.drop(col, axis=1)
                        print(f"      移除列: {col}")
                
                # 添加资产标的列
                df['asset_underlying'] = subfolder_name
                subfolder_dfs.append(df)
                print(f"    读取文件: {csv_file.name} ({len(df)} 行，{len(df.columns)} 列)")
                
            except Exception as e:
                print(f"    错误: 无法读取文件 {csv_file.name}: {e}")
                continue
        
        if subfolder_dfs:
            # 合并该子文件夹的所有数据
            subfolder_combined = pd.concat(subfolder_dfs, ignore_index=True)
            all_dataframes.append(subfolder_combined)
            print(f"    子文件夹 {subfolder_name} 合并完成: {len(subfolder_combined)} 行")
    
    if not all_dataframes:
        print(f"  目录 {target_path.name} 中未找到任何有效CSV文件")
        return pd.DataFrame()
    
    # 合并所有数据
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 保存到父目录下
    try:
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  合并完成: {len(final_df)} 行数据已保存到 {output_path}")
    except Exception as e:
        print(f"  保存失败: {e}")
        return final_df
    
    return final_df

def analyze_directory_data(df, directory_name):
    """分析单个目录的合并数据"""
    if df.empty:
        print(f"  {directory_name} 目录数据为空")
        return
    
    print(f"  === {directory_name} 数据分析 ===")
    print(f"    数据形状: {df.shape}")
    print(f"    包含的资产标的: {df['asset_underlying'].nunique()} 个")
    
    # 各资产标的数据分布
    asset_counts = df['asset_underlying'].value_counts()
    for asset, count in asset_counts.head(10).items():  # 显示前10个
        print(f"      {asset}: {count} 行")
    
    if len(asset_counts) > 10:
        print(f"      ... 还有 {len(asset_counts) - 10} 个资产标的")

def process_both_directories():
    """处理两个指定目录"""
    directories = [
        (Trainning_DATA_DIR, "trainning_data_merged.csv"),
        (Testing_DATA_DIR, "testing_data_merged.csv")
    ]
    
    results = {}
    
    print("开始处理指定目录...")
    
    for target_dir, output_filename in directories:
        print(f"\n{'='*50}")
        print(f"处理目录: {target_dir}")
        print(f"输出文件: {output_filename}")
        
        try:
            merged_df = merge_csv_files_in_directory(target_dir, output_filename)
            results[Path(target_dir).name] = merged_df
            
            # 分析数据
            analyze_directory_data(merged_df, Path(target_dir).name)
            
        except Exception as e:
            print(f"处理目录 {target_dir} 时出错: {e}")
            import traceback
            traceback.print_exc()
            results[Path(target_dir).name] = pd.DataFrame()
    
    return results

def main():
    """主执行函数"""
    try:
        # 处理两个目录
        results = process_both_directories()
        
        # 汇总统计
        print(f"\n{'='*50}")
        print("处理汇总:")
        
        total_rows = 0
        for dir_name, df in results.items():
            if not df.empty:
                print(f"  {dir_name}: {len(df)} 行数据")
                total_rows += len(df)
            else:
                print(f"  {dir_name}: 无有效数据")
        
        print(f"  总计: {total_rows} 行数据")
        
        return results
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 独立的工具函数：处理单个目录（可复用）
def merge_single_directory(directory_path, output_name=None):
    """
    独立处理单个目录的工具函数
    
    参数:
    directory_path: str/Path, 目录路径
    output_name: str, 输出文件名（可选，默认为 merged_data.csv）
    """
    if output_name is None:
        output_name = f"{Path(directory_path).name}_merged.csv"
    
    return merge_csv_files_in_directory(directory_path, output_name)

# 如果作为脚本直接运行
if __name__ == "__main__":
    results = main()
    
    if results:
        print("\n任务完成！")
        for dir_name, df in results.items():
            if not df.empty:
                print(f"{dir_name} 目录处理成功: {len(df)} 行")
            else:
                print(f"{dir_name} 目录无数据或处理失败")
    else:
        print("任务执行失败")