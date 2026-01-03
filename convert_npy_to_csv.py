#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将生成的npy文件转换为CSV格式的脚本
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

def convert_npy_to_csv(npy_file_path, output_csv_path=None):
    """
    将npy文件转换为CSV格式
    
    Args:
        npy_file_path: npy文件路径
        output_csv_path: 输出CSV文件路径（可选）
    """
    npy_path = Path(npy_file_path)
    
    if not npy_path.exists():
        print(f"❌ 文件不存在: {npy_path}")
        return
    
    # 加载npy数据
    data = np.load(npy_path)
    print(f"📊 数据形状: {data.shape}")
    
    # 确定输出文件名
    if output_csv_path is None:
        csv_path = npy_path.with_suffix('.csv')
    else:
        csv_path = Path(output_csv_path)
    
    # 根据数据维度进行转换
    if len(data.shape) == 3:  # [conditions, paths, time_steps]
        conditions, paths, time_steps = data.shape
        print(f"   条件数: {conditions}, 路径数: {paths}, 时间步数: {time_steps}")
        
        # 重塑为长格式
        df_data = []
        for cond_idx in range(conditions):
            for path_idx in range(paths):
                row_data = {
                    'condition_idx': cond_idx,
                    'path_idx': path_idx,
                    **{f'day_{i+1}': data[cond_idx, path_idx, i] 
                       for i in range(time_steps)}
                }
                df_data.append(row_data)
        
        df = pd.DataFrame(df_data)
        
    elif len(data.shape) == 2:  # [paths, time_steps]
        paths, time_steps = data.shape
        print(f"   路径数: {paths}, 时间步数: {time_steps}")
        
        df_data = []
        for path_idx in range(paths):
            row_data = {
                'path_idx': path_idx,
                **{f'day_{i+1}': data[path_idx, i] 
                   for i in range(time_steps)}
            }
            df_data.append(row_data)
        
        df = pd.DataFrame(df_data)
        
    else:
        print(f"❌ 不支持的数据维度: {data.shape}")
        return
    
    # 保存CSV
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV文件保存成功: {csv_path}")
        print(f"   DataFrame形状: {df.shape}")
    except Exception as e:
        print(f"❌ 保存CSV文件失败: {e}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python convert_npy_to_csv.py <npy_file_path> [output_csv_path]")
        print("示例: python convert_npy_to_csv.py Results/Path_Generator_Results/CSI1000/unet_generated_paths_1024_samples.npy")
        return
    
    npy_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_npy_to_csv(npy_file, output_csv)

if __name__ == "__main__":
    main()
