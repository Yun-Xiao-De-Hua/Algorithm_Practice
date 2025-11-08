
# 音乐流行度预测实验

本项目旨在通过构建多种线性回归模型，对音乐的流行度进行预测。实验的核心在于探索和处理数据中的多重共线性问题，并对比不同策略（如PCA降维、正则化等）对模型性能的影响。

## 目录

- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [运行说明](#运行说明)
- [实验流程概览](#实验流程概览)
- [输出结果](#输出结果)

## 项目结构

```
.
├── song_data.csv                   # 实验所需的数据集文件
├── music_popularity_prediction.py    # 实验主脚本
├── requirements.txt                # 项目依赖的Python库列表
├── README.md                       # 项目说明文件
└── (输出文件将在运行后生成)
    ├── visualization_*.png         # 生成的可视化图表
    ├── best_model.pkl              # 性能最佳的模型文件
    ├── scaler.pkl                  # 用于数据标准化的预处理器文件
    └── pca.pkl                     # (如果最佳模型基于PCA) PCA转换器文件
```

## 环境配置

本项目依赖于一系列Python科学计算库。为了确保脚本能够顺利运行，请先配置好您的Python环境。

**1. 前提条件**
   - 已安装 Python (推荐版本 3.8 或更高版本)。
   - 已安装 `pip` 包管理器。

**2. 安装依赖**
   打开您的终端（或命令行），进入项目根目录，然后运行以下命令来安装所有必需的库：

   ```bash
   pip install -r requirements.txt
   ```
   
   该命令会自动读取 `requirements.txt` 文件，并安装所有指定版本的依赖项，包括 `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, 和 `seaborn` 等。

## 运行说明

配置好环境后，您可以直接运行实验主脚本。

**1. 确保文件就位**
   请确保 `song_data.csv` 数据文件与 `music_popularity_prediction.py` 脚本位于同一目录下。

**2. 执行脚本**
   在您的终端中，同样位于项目根目录下，运行以下命令：

   ```bash
   python music_popularity_prediction.py
   ```

**3. 查看输出**
   脚本运行时，将会在终端中按步骤打印出详细的执行信息，包括：
   - 数据加载与预处理的摘要。
   - 多重共线性检测的结果（VIF值）。
   - 所有六种模型的训练与评估性能指标（MSE和R²分数）。
   
   同时，脚本会自动弹出并保存所有分析图表，最终完成最佳模型的保存。

## 实验流程概览

该脚本将自动执行以下完整的实验流程：

1.  **数据加载与预处理**: 加载数据，进行独热编码和特征标准化。
2.  **共线性分析**: 通过相关性热力图和VIF值检测多重共线性，并使用PCA进行处理。
3.  **模型训练与评估**: 训练并评估六种不同的模型策略：
    - 线性回归 (原始数据 / PCA数据)
    - 岭回归 (原始数据 / PCA数据)
    - Lasso回归 (原始数据 / PCA数据)
4.  **结果可视化**: 生成四张核心分析图表，用于展示数据特性和模型性能。
5.  **模型保存**: 自动选出在测试集上MSE最低的模型，并将其与所需的预处理器（scaler, pca）一同保存为 `.pkl` 文件。

## 输出结果

成功运行脚本后，项目目录下将会生成以下文件：

-   **可视化图表 (`.png` files)**:
    -   `visualization_1_correlation_heatmap.png`: 原始连续特征相关性热力图。
    -   `visualization_2_pca_variance_annotated.png`: PCA累积方差贡献率曲线。
    -   `visualization_3_model_performance_comparison_six_models.png`: 六种模型性能对比柱状图。
    -   `visualization_4_coefficient_comparison_final.png`: 模型特征系数值对比图。

-   **模型文件 (`.pkl` files)**:
    -   `best_model.pkl`: 性能最佳的已训练模型。
    -   `scaler.pkl`: 用于标准化新数据的 `StandardScaler` 对象。
    -   `pca.pkl`: (可选) 如果最佳模型是基于PCA的，此文件将被创建，包含已拟合的 `PCA` 对象。