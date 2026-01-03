import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
import logging
import torch
import torch.nn as nn

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedStandardScaler:
    """增强版标准化器，结合了异常值处理和多种变换方法"""
    
    def __init__(self, outlier_threshold=3.0):
        self.transformers = []
        self.robust_scaler = RobustScaler()
        self.n_features = None
        self.outlier_threshold = outlier_threshold
        
    def handle_outliers(self, X):
        X_copy = X.copy()
        for i in range(X.shape[1]):
            column = X[:, i]
            mean = np.mean(column)
            std = np.std(column)
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std
            X_copy[:, i] = np.clip(column, lower_bound, upper_bound)
        return X_copy
        
    def fit_transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        self.n_features = X.shape[1]
        X_robust = self.robust_scaler.fit_transform(X)
        X_no_outliers = self.handle_outliers(X_robust)
        
        transformed_X = np.zeros_like(X_no_outliers)
        for i in range(self.n_features):
            transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            transformed_X[:, i] = transformer.fit_transform(X_no_outliers[:, i].reshape(-1, 1)).ravel()
            self.transformers.append(transformer)
            
        return transformed_X
    
    def transform(self, X):
        if self.n_features is None:
            raise ValueError("Scaler has not been fitted yet.")
            
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        X_robust = self.robust_scaler.transform(X)
        X_no_outliers = self.handle_outliers(X_robust)
        
        transformed_X = np.zeros_like(X_no_outliers)
        for i in range(self.n_features):
            transformed_X[:, i] = self.transformers[i].transform(X_no_outliers[:, i].reshape(-1, 1)).ravel()
        
        return transformed_X
    
    def inverse_transform(self, X):
        if self.n_features is None:
            raise ValueError("Scaler has not been fitted yet.")
            
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        inverse_X = np.zeros_like(X)
        for i in range(self.n_features):
            inverse_X[:, i] = self.transformers[i].inverse_transform(X[:, i].reshape(-1, 1)).ravel()
        
        inverse_X = self.robust_scaler.inverse_transform(inverse_X)
        return inverse_X


class DataProcessor:
    """修复版数据处理器 - 正确处理类别特征"""
    
    def __init__(self, config):
        """
        初始化数据处理器
        
        Args:
            config (dict): 配置参数字典
        """
        self.config = config
        # 只对前5个数值特征标准化
        self.price_scaler = EnhancedStandardScaler(outlier_threshold=3.0)
        
        # 记录类别特征信息
        self.num_countries = None
        self.num_indices = None
        self.country_table = None
        self.index_table = None
        
    def load_data(self, path_or_df):
        """既可读CSV路径，也可直接吃DataFrame"""
        logging.info("步骤1: 正在加载数据...")
        if isinstance(path_or_df, (str, os.PathLike)):
            path = str(path_or_df)
            if not os.path.exists(path):
                raise FileNotFoundError(f"错误：找不到文件 {path}")
            df = pd.read_csv(path)
            logging.info(f"加载完成: {len(df)} 条样本 (from CSV)")
            return df
        elif isinstance(path_or_df, pd.DataFrame):
            logging.info(f"加载完成: {len(path_or_df)} 条样本 (from DataFrame)")
            return path_or_df.copy()
        else:
            raise TypeError("path_or_df 必须是 CSV 路径 或 pandas.DataFrame")
            
    def _encode_ids(self, df, col, default_name="__default__"):
        """把 df[col] 映射为 0..K-1 的稳定整数ID；若列不存在则全部用默认类别"""
        if col in df.columns:
            col_s = df[col].astype(str)
        else:
            col_s = pd.Series([default_name] * len(df), index=df.index, dtype=str)
        cats = sorted(col_s.unique().tolist())
        table = {c: i for i, c in enumerate(cats)}
        ids = col_s.map(table).astype(int)
        return ids, table
        
    def process_price_data(self, df):
        """处理价格数据"""
        df_processed = df.copy()
        
        if 'start_price' not in df_processed.columns:
            raise ValueError("数据中缺少'start_price'列")
        
        try:
            df_processed['start_price'] = pd.to_numeric(df_processed['start_price'], errors='coerce')
            if df_processed['start_price'].isnull().any():
                logging.warning("存在无效的start_price值，已被转换为NaN")
        except Exception as e:
            logging.error(f"处理start_price时发生错误: {str(e)}")
            raise
            
        df_processed.rename(columns={'start_price': 'S_0'}, inplace=True)
        
        try:
            df_processed['price_series'] = df_processed['price_series'].apply(
                lambda x: np.array(eval(x), dtype=np.float32)
            )
        except Exception as e:
            logging.error(f"处理price_series时发生错误: {str(e)}")
            raise
            
        return df_processed
        
    def transform_price_sequence(self, df):
        """转换价格序列"""
        def process_row(price_series):
            try:
                prices = np.array(price_series, dtype=np.float32)
                if len(prices) < 2:
                    raise ValueError("价格序列长度必须大于1")
                    
                log_returns = np.diff(np.log(prices))
                scaled_returns = log_returns / self.config['volatility_scale']
                
                target = np.zeros(self.config['input_sequence_length'], dtype=np.float32)
                mask = np.zeros(self.config['input_sequence_length'], dtype=np.float32)
                
                target[0] = 1.0
                mask[0] = 1.0
                
                num_returns = min(len(scaled_returns), self.config['input_sequence_length'] - 1)
                target[1:num_returns + 1] = scaled_returns[:num_returns]
                mask[1:num_returns + 1] = 1.0
                
                return target, mask
            except Exception as e:
                logging.error(f"处理价格序列时发生错误: {str(e)}")
                raise
        
        df_transformed = df.copy()
        sequences_and_masks = df_transformed['price_series'].apply(process_row)
        df_transformed['transformed_sequence'] = sequences_and_masks.apply(lambda x: x[0])
        df_transformed['validity_mask'] = sequences_and_masks.apply(lambda x: x[1])
        return df_transformed
        
    def create_condition_tensors(self, df, fit_scaler=False):
        """
        创建条件张量 - 关键修复：类别ID不标准化
        
        Returns:
            dict: {
                'conditions': [batch, 7] 所有条件特征（前5个标准化，后2个是原始整数ID）
                'num_countries': 国家数量
                'num_indices': 指数数量
            }
        """
        try:
            # === 1. 数值特征（需要标准化）===
            prices = df['S_0'].astype(float).values / self.config['base_trading_days']
            contract_days = df['contract_calendar_days'].astype(int).values / 365.0
            trading_days = df['actual_trading_days'].astype(int).values / float(self.config['base_trading_days'])
            volatility = df['volatility'].astype(float).values
            risk_free_rate = df['risk_free_rate'].astype(float).values
            
            numerical_features = np.column_stack([
                prices,
                volatility,
                risk_free_rate,
                contract_days,
                trading_days/contract_days
            ]).astype(np.float32)
            
            # === 2. 类别特征（不标准化）===
            if 'country' in df.columns:
                country_id, self.country_table = self._encode_ids(df, 'country')
                country_id = country_id.values.astype(np.float32)  # 转float32但保持整数值
                self.num_countries = len(self.country_table)
                logging.info(f"检测到 {self.num_countries} 个国家: {list(self.country_table.keys())}")
            else:
                country_id = np.zeros(len(df), dtype=np.float32)
                self.num_countries = 1
                
            if 'asset_underlying' in df.columns:
                index_id, self.index_table = self._encode_ids(df, 'asset_underlying')
                index_id = index_id.values.astype(np.float32)
                self.num_indices = len(self.index_table)
                logging.info(f"检测到 {self.num_indices} 个指数: {list(self.index_table.keys())}")
            else:
                index_id = np.zeros(len(df), dtype=np.float32)
                self.num_indices = 1

            # === 3. 只对前5个数值特征标准化 ===
            if fit_scaler:
                scaled_numerical = self.price_scaler.fit_transform(numerical_features)
                logging.info("已拟合数值特征标准化器")
            else:
                scaled_numerical = self.price_scaler.transform(numerical_features)

            # === 4. 拼接所有7个条件（前5个标准化，后2个原始ID）===
            conditions = np.column_stack([
                scaled_numerical,           # 5个标准化的数值特征
                country_id.reshape(-1, 1),  # 国家ID（整数，未标准化）
                index_id.reshape(-1, 1)     # 指数ID（整数，未标准化）
            ]).astype(np.float32)
            
            logging.info(f"条件张量形状: {conditions.shape} [前5列标准化，后2列是原始类别ID]")
            
            return {
                'conditions': conditions,
                'num_countries': self.num_countries,
                'num_indices': self.num_indices
            }
            
        except Exception as e:
            logging.error(f"创建条件张量时发生错误: {str(e)}")
            raise
            
    def recover_from_prediction(self, x_sample, y_pred):
        """从预测结果恢复价格序列"""
        try:
            # 只取前5个数值特征进行逆变换
            x_numerical = x_sample[:5].reshape(1, -1)
            unscaled_features = self.price_scaler.inverse_transform(x_numerical)
            
            normalized_price = unscaled_features[0, 0]
            start_price = normalized_price * self.config['base_trading_days']
            
            y_pred_flat = y_pred.flatten()
            valid_returns = y_pred_flat[1:][y_pred_flat[1:] != 0]
            unscaled_returns = valid_returns * self.config['volatility_scale']
            
            log_prices = np.concatenate([
                [np.log(start_price)],
                np.log(start_price) + np.cumsum(unscaled_returns)
            ])
            
            restored_prices = np.exp(log_prices)
            return restored_prices
        except Exception as e:
            logging.error(f"从预测结果恢复价格序列时发生错误: {str(e)}")
            raise
            
    def process_all_data(self, path):
        """一键处理所有数据"""
        try:
            print("=" * 60)
            print("开始数据处理流程...")
            print("=" * 60)
            
            df_raw = self.load_data(path)
            df_processed = self.process_price_data(df_raw)
            df_transformed = self.transform_price_sequence(df_processed)
            
            print("\n4. 创建条件张量...")
            condition_dict = self.create_condition_tensors(df_transformed, fit_scaler=True)
            
            print("\n5. 准备目标变量和有效性mask...")
            y_t = np.array(df_transformed['transformed_sequence'].tolist())
            mask_t = np.array(df_transformed['validity_mask'].tolist())
            
            y_t = y_t.reshape(len(y_t), 1, -1)
            mask_t = mask_t.reshape(len(mask_t), 1, -1)
            
            X_t = torch.FloatTensor(condition_dict['conditions'])
            y_t = torch.FloatTensor(y_t)
            mask_t = torch.FloatTensor(mask_t)
            
            print("\n" + "=" * 60)
            print("数据处理完成！")
            print("=" * 60)
            print(f"条件特征: {X_t.shape} (包含7个条件)")
            print(f"  - 前5列: 标准化的数值特征")
            print(f"  - 第6列: 国家ID (共 {self.num_countries} 个国家)")
            print(f"  - 第7列: 指数ID (共 {self.num_indices} 个指数)")
            print(f"目标序列: {y_t.shape}")
            print(f"有效性Mask: {mask_t.shape}")
            print("=" * 60)
            
            return X_t, y_t, mask_t
        except Exception as e:
            logging.error(f"处理数据时发生错误: {str(e)}")
            raise


# ============================================================================
# 增强条件网络 - 强化国家和指数的权重
# ============================================================================

class EnhancedConditionNetwork(nn.Module):
    """
    增强条件网络 - 让国家ID和指数ID的权重显著大于其他5个条件
    
    核心思路：
    1. 对国家ID和指数ID使用Embedding（学习丰富的表示）
    2. 对其他5个数值条件使用简单投影
    3. Embedding的维度远大于数值特征，自然增强权重
    4. 使用加权融合机制，显式提高类别特征的重要性
    """
    
    def __init__(self, 
                 num_countries=10,
                 num_indices=50,
                 country_emb_dim=64,        # 国家embedding维度（宏观环境）
                 index_emb_dim=128,         # 指数embedding维度（风格因子，最重要）
                 numerical_proj_dim=32,     # 数值特征投影维度（较小）
                 hidden_dim=256,
                 output_dim=128):
        super().__init__()
        
        # === 1. 国家和指数的Embedding层（核心强化点）===
        self.country_embedding = nn.Embedding(num_countries, country_emb_dim)
        self.index_embedding = nn.Embedding(num_indices, index_emb_dim)
        
        # 使用较大的初始化标准差，让embedding的影响更显著
        nn.init.normal_(self.country_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.index_embedding.weight, mean=0, std=0.1)
        
        # === 2. 数值特征的简单投影（维度较小）===
        self.numerical_proj = nn.Sequential(
            nn.Linear(5, numerical_proj_dim),  # 5个数值特征 → 32维
            nn.LayerNorm(numerical_proj_dim),
            nn.SiLU()
        )
        
        # === 3. 加权融合层（显式增强类别特征权重）===
        total_dim = numerical_proj_dim + country_emb_dim + index_emb_dim
        
        # 可学习的权重参数
        self.numerical_weight = nn.Parameter(torch.tensor(0.8))  # 数值特征权重较小
        self.country_weight = nn.Parameter(torch.tensor(1))    # 国家权重
        self.index_weight = nn.Parameter(torch.tensor(2))      # 指数权重最大
        
        # === 4. 主干网络 ===
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, conditions):
        """
        Args:
            conditions: [batch, 7] - 所有条件
                - conditions[:, :5]: 标准化的数值特征
                - conditions[:, 5]: 国家ID (整数)
                - conditions[:, 6]: 指数ID (整数)
        Returns:
            condition_embedding: [batch, output_dim]
        """
        # === 1. 分离特征 ===
        numerical_features = conditions[:, :5]           # [batch, 5]
        country_ids = conditions[:, 5].long()            # [batch] 转为整数
        index_ids = conditions[:, 6].long()              # [batch]
        
        # === 2. 获取表示 ===
        numerical_feat = self.numerical_proj(numerical_features)  # [batch, 32]
        country_emb = self.country_embedding(country_ids)         # [batch, 64]
        index_emb = self.index_embedding(index_ids)               # [batch, 128]
        
        # === 3. 加权融合（关键：显式增强类别特征的权重）===
        weighted_numerical = self.numerical_weight * numerical_feat
        weighted_country = self.country_weight * country_emb
        weighted_index = self.index_weight * index_emb
        
        combined = torch.cat([weighted_numerical, weighted_country, weighted_index], dim=-1)
        
        # === 4. 通过主干网络 ===
        condition_embedding = self.fusion_network(combined)
        
        return condition_embedding
    
    def get_feature_importance(self):
        """返回各特征的相对权重（用于可视化）"""
        weights = {
            'numerical': self.numerical_weight.item(),
            'country': self.country_weight.item(),
            'index': self.index_weight.item()
        }
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 配置
    config = {
        'base_trading_days': 252,
        'volatility_scale': 0.01,
        'input_sequence_length': 100
    }
    
    print("\n" + "="*60)
    print("修复版数据处理与条件网络测试")
    print("="*60)
    
    # 1. 数据处理
    processor = DataProcessor(config)
    data_dict = processor.process_all_data('your_data.csv')
    
    # 2. 创建条件网络
    print("\n创建增强条件网络...")
    condition_net = EnhancedConditionNetwork(
        num_countries=data_dict['num_countries'],
        num_indices=data_dict['num_indices'],
        country_emb_dim=64,         # 国家：宏观环境
        index_emb_dim=128,          # 指数：风格因子（维度是国家的2倍）
        numerical_proj_dim=32,      # 数值特征（维度最小）
        hidden_dim=256,
        output_dim=128
    )
    
    print(f"网络参数量: {sum(p.numel() for p in condition_net.parameters()):,}")
    
    # 3. 查看特征权重分配
    print("\n特征权重分配:")
    importance = condition_net.get_feature_importance()
    print(f"  数值特征: {importance['numerical']:.1%}")
    print(f"  国家特征: {importance['country']:.1%}")
    print(f"  指数特征: {importance['index']:.1%}")
    
    # 4. 测试前向传播
    print("\n测试前向传播...")
    with torch.no_grad():
        condition_emb = condition_net(data_dict['X'][:8])
        print(f"输入条件形状: {data_dict['X'][:8].shape}")
        print(f"输出embedding形状: {condition_emb.shape}")
        print(f"Embedding统计: mean={condition_emb.mean():.4f}, std={condition_emb.std():.4f}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    
    # 5. 打印映射表
    if data_dict['country_table']:
        print("\n国家映射表:")
        for country, idx in sorted(data_dict['country_table'].items(), key=lambda x: x[1]):
            print(f"  '{country}' → ID {idx}")
    
    if data_dict['index_table']:
        print("\n指数映射表:")
        for index, idx in sorted(data_dict['index_table'].items(), key=lambda x: x[1]):
            print(f"  '{index}' → ID {idx}")