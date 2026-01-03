# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd # 确保导入 pandas

# --- 路径设置 ---
current_file_dir = Path(__file__).parent.resolve()
project_root = current_file_dir.parent
sys.path.append(str(project_root))

# --- 导入项目路径和组件 ---
try:
    # 使用我们建立的 'pp' 别名约定
    import Project_Path as pp
    from Data.DataProvider import MultiMarketDataProvider
    from Data.DatasetBuilder import DatasetProcessor
    from Config.Data_Collection_config import Data_Collection_Config as CONFIG
except ImportError as e:
    print(f"❌ 导入必需模块时出错: {e}")
    print("请确保所有必需的模块 (Project_Path, DataProvider, DatasetBuilder, Config) 都存在。")
    sys.exit(1)

def data_collection_and_preprocessing(ticker, name, market):
    print(f"\n--- 开始为 {name} ({ticker}) 进行数据收集与预处理 ---")

    # --- 1. 确保输出目录存在 ---
    # 直接使用 Project_Path 变量
    train_name_dir = pp.Trainning_DATA_DIR / name
    val_name_dir = pp.Testing_DATA_DIR / name # 假设验证数据存放在 Testing_DATA_DIR
    # 不再在此处创建报告目录
    
    train_name_dir.mkdir(parents=True, exist_ok=True)
    val_name_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 2. 使用 DataProvider 获取数据 ---
    print(f"   正在从 {market} 市场获取 {ticker} 的数据...")
    provider = MultiMarketDataProvider()
    try:
        df_stock, rates_dict = provider.get_data_package(
            market=market,
            ticker=ticker,
            start_date=CONFIG["start_date"],
            end_date=CONFIG["end_date"],
            periods=CONFIG["contract_calendar_days"]
        )
        print(f"   ✅ 数据获取成功。")
    except Exception as e:
        print(f"   ❌ 获取数据时出错: {e}")
        return # 如果数据获取失败，则停止处理此资产

    # --- 3. 使用 DatasetProcessor 处理数据 ---
    print(f"   正在处理数据...")
    processor = DatasetProcessor(
        periods=CONFIG["contract_calendar_days"],
        vol_lookback=CONFIG["vol_lookback"],
        cutoff_date_str=CONFIG["cutoff_date"],
        market=market,
    )
    
    try:
        train_df, val_df, estimator = processor.process_all(df_stock, rates_dict)
        print(f"   ✅ 数据处理完成。训练集形状: {train_df.shape}, 验证集形状: {val_df.shape}")
    except Exception as e:
        print(f"   ❌ 处理数据时出错: {e}")
        return # 如果处理失败，则停止

    # --- 4. 保存处理后的 DataFrame ---
    train_save_path = train_name_dir / "train_df.csv"
    val_save_path = val_name_dir / "val_df.csv"
    
    try:
        train_df.to_csv(train_save_path, index=False)
        print(f"   💾 训练数据已保存至: {train_save_path}")
        val_df.to_csv(val_save_path, index=False)
        print(f"   💾 验证数据已保存至: {val_save_path}")
    except Exception as e:
        print(f"   ❌ 保存 DataFrame 时出错: {e}")
        
    # --- 移除 Explainer 部分 ---
    # 旧的 explainer 调用已被移除。分析现在是一个独立的步骤。

    print(f"--- {name} 的数据收集与预处理完成 ---")
    return None # 表示此资产成功处理

# --- 主执行块 ---
if __name__ == "__main__":
    assets_to_process = [
        {'ticker': '000852', 'name': 'CSI1000', 'market': 'china'}, # 中证1000
        {'ticker': '^GSPC', 'name': 'SP500', 'market': 'usa'},          # S&P 500
        {'ticker': '000001', 'name': 'SSE_Composite', 'market': 'china'}, # 上证指数
        {'ticker': '^DJI', 'name': 'Dow_Jones', 'market': 'usa'},      # 道琼斯
        {'ticker': '000300', 'name': 'CSI300', 'market': 'china'},     # 沪深300
        {'ticker': '000905', 'name': 'CSI500', 'market': 'china'},     # 中证500
        {'ticker': '^IXIC', 'name': 'NASDAQ', 'market': 'usa'},       # 纳斯达克
        {'ticker': '^RUT', 'name': 'Russell_2000', 'market': 'usa'},    # 罗素2000
    ]

    for asset in assets_to_process:
        data_collection_and_preprocessing(**asset)

    print("\n✅ 所有指定的资产已处理完毕。")