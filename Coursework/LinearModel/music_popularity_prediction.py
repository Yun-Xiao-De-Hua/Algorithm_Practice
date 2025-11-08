import pandas as pd
import numpy as np
import pickle

# 计算方差膨胀因子 (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 模型的训练与评估
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

# 结果可视化
import matplotlib.pyplot as plt
import seaborn as sns


# ================== 1. 环境设置与数据加载 ==================
print("1. 环境设置与数据加载")

# 可视化全局设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# 数据加载
try:
    df = pd.read_csv("song_data.csv")
    print("成功加载 song_data.csv 数据集")
except FileNotFoundError:
    print("song_data.csv 未找到，请将文件与该脚本放置在同一目录下")
    exit()

print("数据加载结束")
print("-" * 80)


# ================== 2. 数据预处理 ==================
print("2. 数据预处理")

# 特征输入矩阵
X = df.drop(columns=['song_name','song_popularity'])
# 目标参数
y = df['song_popularity']

# 连续变量
continuous_features = [
    'song_duration_ms', 'acousticness', 'danceability', 'energy',
    'instrumentalness', 'liveness', 'loudness', 'speechiness',
    'tempo', 'audio_valence'
]

# 离散变量
discrete_features = ['key', 'audio_mode', 'time_signature']

# ------ （1）离散属性连续化（独热编码） ------
X = pd.get_dummies(X, columns=discrete_features, drop_first=True)
print(f"已对离散属性{discrete_features}完成独热编码")

# 计算独热编码编码后模型使用的总特征数
num_continuous = len(continuous_features)
num_key_features = df['key'].nunique() - 1  # 使用原始类型数-1，避免共线性影响
num_audiomode_features = df['audio_mode'].nunique() - 1
num_time_signature_features = df['time_signature'].nunique() - 1
total_features = num_continuous + num_key_features + num_audiomode_features + num_time_signature_features

print(f"{num_continuous}(连续变量) + {num_key_features}(Key) + {num_audiomode_features}(Audiomode) + "
      f"{num_time_signature_features}(Timesignature) = {total_features}(模型使用的总特征数)")

# 单次留出法划分数据集(80%训练集/20%测试集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
# 保留编码后的最终特征名称列表，用于后续可视化实验结果
final_feature_names = X.columns
print("已将数据集划分为：80%训练集，20%测试集")

# ------ （2）连续属性归一化（采用标准化方法）------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=final_feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=final_feature_names)    # 使用训练集中学习到的均值与方差
print("训练集、数据集连续属性归一化操作完成")

# ------ （3）多重共线性的检测与处理 ------

# 共线性检测——热力图分析 + VIF计算
# 相关性热力图
print("生成相关性热力图...")
plt.figure(figsize=(12, 10))
# 计算原始连续特征之间的皮尔逊相关系数矩阵
correlation_matrix = df[continuous_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('原始连续特征相关性热力图', fontsize=18)
plt.savefig('visualization_1_correlation.heatmap.png', dpi=900)
plt.show()

# 计算方差膨胀因子
X_train_vif = add_constant(X_train_scaled)
# 存储VIF结果
vif_data = pd.DataFrame()
vif_data['feature'] = X_train_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
print("VIF 检测结果（VIF > 2）：")
print(vif_data[vif_data['VIF'] > 2].sort_values(by='VIF', ascending=False))

# 共线性处理——PCA
pca = PCA(n_components=0.95)    # 保留解释原始数据 95% 方差的主成分
X_train_pca = pca.fit_transform(X_train_scaled) # 应用PCA进行降维，并得到变换后的训练集
X_test_pca = pca.transform(X_test_scaled)   # 变换得到测试集
print(f"已将{X.shape[1]}个特征降维至{pca.n_components_}个主成分（保留95%方差）")
print("共线性检测与处理操作完成")
print("数据预处理操作完成")
print('-' * 80)


# ================== 3. 模型的构建、训练与评估 ==================
print("3. 模型的构建、训练与评估")

# --- 模型构建与训练 ---
# 基础线性回归模型
# 基于标准化输入的线性回归
lr = LinearRegression().fit(X_train_scaled, y_train)
# 基于PCA降维后数据的线性回归，用于后续性能对比
lr_pca = LinearRegression().fit(X_train_pca, y_train)

# 岭回归
# 设置alpha候选值
alpha_range = [0.001,0.01,0.1,1,10,100]
# 交叉验证确定最优alpha
ridge_cv = RidgeCV(alphas=alpha_range, cv=5).fit(X_train_scaled, y_train)
ridge_cv_pca = RidgeCV(alphas=alpha_range, cv=5).fit(X_train_pca, y_train)

# Lasso回归
lasso_cv = LassoCV(alphas=alpha_range, cv=5, random_state=66, max_iter=10000).fit(X_train_scaled, y_train)
lasso_cv_pca = LassoCV(alphas=alpha_range, cv=5, random_state=66, max_iter=10000).fit(X_train_pca, y_train)


# --- 模型性能评估辅助函数定义 ---
model_results = {}  # 创建一个空字典，存放不同模型的评估指标，用于后续分析与可视化呈现
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """辅助函数，存放模型评估指标：MSE和R^2分数"""
    # 训练集性能
    y_pred_train = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_pred_train)

    # 测试集性能
    y_pred_test = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    # 结果存入字典
    model_results[name] = {'训练MSE': train_mse, '测试MSE': test_mse, '测试R^2': test_r2}

    # 结果输出
    print(f"评估模型：{name}")
    print(f"训练均方误差：{train_mse:.4f}")
    print(f"测试均方误差：{test_mse:.4f}")
    print(f"测试R^2分数：{test_r2:.4f}")


# --- 模型性能评估 ---
evaluate_model('基础线性回归模型(基于标准化输入数据)', lr,  X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model('基础线性回归模型(基于PCA降维数据)', lr_pca,  X_train_pca, X_test_pca, y_train, y_test)
evaluate_model('岭回归(基于标准化输入数据)', ridge_cv,  X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model('岭回归(基于PCA降维数据)', ridge_cv_pca,  X_train_pca, X_test_pca, y_train, y_test)
evaluate_model('Lasso回归(基于标准化输入数据)', lasso_cv,  X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model('Lasso(基于PCA降维数据)', lasso_cv_pca,  X_train_pca, X_test_pca, y_train, y_test)

print("所有模型评估完成")
print('-' * 80)


# ================== 4. 结果可视化与分析 ==================
print("4. 结果可视化与分析")

# --- PCA累积方差贡献率曲线 ---
print("生成PCA累积方差贡献率曲线...")
pca_full = PCA().fit(X_train_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='--', label='累积方差贡献率')
ax.axhline(y=0.95, color='r', linestyle='-', label='95%方差阈值')
ax.text(0.5, 0.95, ' 95% 阈值', color='r', va='bottom', ha='left', backgroundcolor='white')
ax.axvline(x=pca.n_components_, color='g', linestyle=':', label=f'选取的 {pca.n_components_} 个主成分')
ax.text(pca.n_components_, 0.1, f' {pca.n_components_} 个主成分', color='g', va='bottom', ha='center', rotation=90,
        backgroundcolor='white')
ax.set_title('PCA累积方差贡献率', fontsize=18)
ax.set_xlabel('主成分数量')
ax.set_ylabel('累积方差贡献率')
ax.legend(loc='center right')
ax.grid(True, which="both", ls="--")
ax.set_xticks(np.arange(0, len(cumulative_variance) + 1, 2))
ax.set_yticks(np.arange(0, 1.1, 0.1))
plt.tight_layout()
plt.savefig('visualization_2_pca_variance_annotated.png', dpi=900)
plt.show()

# --- 模型性能对比柱状图 (MSE和R^2分数) ---
print("生成六种模型的性能对比柱状图...")
results_df = pd.DataFrame(model_results).T

model_names = [
    '基础线性回归模型(基于标准化输入数据)', '基础线性回归模型(基于PCA降维数据)',
    '岭回归(基于标准化输入数据)', '岭回归(基于PCA降维数据)',
    'Lasso回归(基于标准化输入数据)', 'Lasso(基于PCA降维数据)'
]

# 简化图标签
model_short_names = [
    '线性回归 (原始数据)', '线性回归 (PCA数据)',
    '岭回归 (原始数据)', '岭回归 (PCA数据)',
    'Lasso回归 (原始数据)', 'Lasso回归 (PCA数据)'
]

results_df_plot = results_df.copy()
results_df_plot.index = model_short_names

x = np.arange(len(model_short_names))
width = 0.35

fig, ax = plt.subplots(figsize=(20, 10))
rects1 = ax.bar(x - width/2, results_df_plot['训练MSE'], width, label='训练集 MSE', color='skyblue')
rects2 = ax.bar(x + width/2, results_df_plot['测试MSE'], width, label='测试集 MSE', color='salmon')

ax.set_ylabel('均方误差 (Mean Squared Error)')
ax.set_title('不同策略下模型的训练与测试的MSE及R^2分数对比', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(model_short_names)
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')

for i, rect in enumerate(rects2):
    # 按顺序获取R^2分数
    r2_value = results_df_plot['测试R^2'].iloc[i]
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 0.1,
            f'R^2={r2_value:.3f}',
            ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

fig.tight_layout()
plt.savefig('visualization_3_model_performance_comparison_six_models.png', dpi=900)
plt.show()

# --- 模型系数对比条形图 ---
print("生成带优化数值标注的模型系数对比条形图 (仅原始特征空间)...")
coefficients = pd.DataFrame({
    '特征': final_feature_names,
    '线性回归': lr.coef_,
    '岭回归': ridge_cv.coef_,
    'Lasso回归': lasso_cv.coef_
})
top_features = coefficients.reindex(coefficients['线性回归'].abs().sort_values(ascending=False).index).head(15)

ax = top_features.plot(x='特征', y=['线性回归', '岭回归', 'Lasso回归'], kind='bar', figsize=(18, 10))
plt.title('各模型特征系数值对比 (在原始特征空间)', fontsize=18)
plt.ylabel('系数值 (Coefficient Value)')
plt.xlabel('特征 (Feature)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--')

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 12 if p.get_height() >= 0 else -12),
                textcoords='offset points',
                fontsize=6,
                color='black')

plt.tight_layout()
plt.savefig('visualization_4_coefficient_comparison_final.png', dpi=900)
plt.show()
print("结果可视化完成")
print("-" * 80)


# ================== 5. 模型保存 ==================
print("5. 模型保存")

# 取测试MSE最低的模型作为最佳模型
best_model_name = pd.DataFrame(model_results).T['测试MSE'].idxmin()
print(f"性能最佳的模型是: '{best_model_name}'")

# 模型保存
if best_model_name == '基础线性回归模型(基于标准化输入数据)':
    best_model = lr
elif best_model_name == '基础线性回归模型(基于PCA降维数据)':
    best_model = lr_pca
elif best_model_name == '岭回归(基于标准化输入数据)':
    best_model = ridge_cv
elif best_model_name == '岭回归(基于PCA降维数据)':
    best_model = ridge_cv_pca
elif best_model_name == 'Lasso回归(基于标准化输入数据)':
    best_model = lasso_cv
elif best_model_name == 'Lasso(基于PCA降维数据)':
    best_model = lasso_cv_pca

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
# 保存用于处理原始数据的scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# 若最佳模型是基于PCA的，则同时保存PCA对象
if 'PCA' in best_model_name:
    with open('pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

print(f"已将模型 '{best_model_name}' 保存至 'best_model.pkl'")
print("用于数据标准化的 'scaler.pkl' 也已保存")
if 'PCA' in best_model_name:
    print("用于PCA转换的 'pca.pkl' 也已保存")