# 金融市场分析与模拟系统

## 项目概述

本项目是一个综合性的金融市场分析与模拟系统，主要用于金融时间序列的建模、预测和策略回测。项目结合了传统的GARCH模型和现代的扩散模型（Diffusion Model）进行金融数据建模，并提供了丰富的解释工具和回测框架。

## 主要功能

1. **多市场数据获取**：支持中国和美国市场的股票和指数数据获取
2. **时间序列建模**：使用扩散模型和GARCH模型进行金融时间序列建模
3. **路径生成与模拟**：基于训练好的模型生成金融资产价格路径
4. **模型解释**：提供数据解释器和路径解释器，帮助理解模型结果
5. **策略回测**：包含回测引擎，用于评估投资策略表现
6. **期权定价**：支持期权定价和收益分析

## 项目结构

```
├── Config/                 # 配置文件目录
├── Data/                   # 数据处理模块
├── Dataset/                # 数据集存储目录
│   ├── Testing_Dataset/    # 测试数据集
│   └── Trainning_Dataset/  # 训练数据集
├── Explainer/              # 解释器模块
├── Game/                   # 回测和期权模块
├── Generator/              # 路径生成器模块
├── Model/                  # 模型实现
│   ├── Diffusion_Model/    # 扩散模型
│   └── Garch_Model/        # GARCH模型
├── Pipelines/              # 工作流管道
├── Results/                # 结果输出目录
├── Project_Path.py         # 项目路径定义
└── requirements.txt        # 项目依赖
```

## 工作流程

项目采用管道式工作流，按照以下顺序执行：

1. **数据收集**：从多个市场源获取金融数据
2. **数据合并**：将不同市场的数据合并处理
3. **数据解释**：分析和可视化原始数据特征
4. **模型训练**：训练扩散模型和GARCH模型
5. **路径生成**：使用训练好的模型生成资产价格路径
6. **路径解释**：分析生成路径的特征和质量
7. **策略回测**：在生成的路径上进行策略回测
8. **回测分析**：分析和解释回测结果

## 安装与使用

### 环境要求

- Python 3.8+
- 详细依赖见 `requirements.txt`

### 安装步骤

1. 克隆仓库
```bash
git clone <repository-url>
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

### 使用方法

项目提供了一系列管道脚本，按照编号顺序执行即可完成完整的工作流：

```bash
# 数据收集
python Pipelines/1-Data_Collection.py

# 数据合并
python Pipelines/2-Merge_Datasets.py

# 数据解释
python Pipelines/3-Dataset_Explaination.py

# 训练扩散模型
python Pipelines/4-Run_Diffusion_DDPM.py

# Or  python Pipelines/4-Run_Diffusion_DLPM.py 

# 二选一即可 若生成比对实验 两个同时运行 

# 训练GARCH模型
python Pipelines/5-Run_Garch_Fitting.py

# 生成路径
python Pipelines/6-Run_generater.py

# 路径解释
python Pipelines/7-run_path_explainer.py

# 策略回测
python Pipelines/8-run_game.py

# 回测结果解释
python Pipelines/9-Run_game_explainer.py
```

## 主要模块说明

### 数据模块 (Data/)

- `DataProvider.py`: 提供统一的数据获取接口，支持中美市场
- `DatasetBuilder.py`: 数据集构建和预处理
- `Input_preparation.py`: 模型输入数据准备

### 模型模块 (Model/)

- `Diffusion_Model/`: 基于扩散过程的时间序列建模
  - `diffusion_with_condition.py`: 条件扩散模型实现
  - `Unet_with_condition.py`: 条件U-Net网络结构
  - `trainer_with_condition.py`: 模型训练器
- `Garch_Model/`: GARCH模型实现
  - `Garch_fitter.py`: GARCH模型拟合

### 生成器模块 (Generator/)

- `path_generator_engine.py`: 路径生成引擎
- `path_simulators.py`: 不同模拟方法实现

### 解释器模块 (Explainer/)

- `data_explainer_engine.py`: 数据解释引擎
- `path_explainer_engine.py`: 路径解释引擎
- `game_explainer.py`: 回测结果解释

### 回测模块 (Game/)

- `backtest_engine.py`: 回测引擎
- `option_payoffs.py`: 期权收益计算

## 配置说明

项目使用多个配置文件管理不同模块的参数：

- `DataProvider_config.py`: 数据提供者配置
- `Data_Collection_config.py`: 数据收集配置
- `Diffusion_config.py`: 扩散模型配置
- `garch_fitter_config.py`: GARCH模型配置
- `generator_config.py`: 路径生成器配置
- `option_config.py`: 期权配置
- `path_explainer_config.py`: 路径解释器配置

## 注意事项

- 数据获取依赖于外部API，请确保网络连接正常
- 模型训练可能需要较长时间，建议使用GPU加速
- 请根据实际需求调整配置文件中的参数

## 贡献与反馈

欢迎提交问题和改进建议，共同完善项目。

## 运行 下列代码 以快速启动实验：
```

python Pipelines/4-Run_Diffusion.py && python Pipelines/5-Run_Garch_Fitting.py && python Pipelines/6-Run_generater.py && python Pipelines/7-run_path_explainer.py && python Pipelines/8-run_game.py && python Pipelines/9-Run_game_explainer.py

```
