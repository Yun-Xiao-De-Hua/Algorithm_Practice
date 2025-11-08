import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import copy
import warnings
import matplotlib.pyplot as plt

# 1. ------------ 数据加载与预处理 ------------

def load_adult_data():
    """加载、清理、划分成人收入数据集"""
    # 定义列名
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    # 定义特征类型
    # 离散特征
    categorical_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    # 连续特征
    continuous_features = [
        'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
    ]

    # 加载 adult.data
    try:
        data = pd.read_csv(
            'adult.data',
            header=None,
            names=column_names,
            sep=r',\s*',    # 确保逗号后有任意数量空格也能够正常加载
            na_values='?',  # data文件中缺失值以'?'表示
            engine='python'
        )
    except FileNotFoundError:
        print("请确保 adult.data 文件与当前脚本文件在同一目录下")
        # 文件加载失败时，返回None，数量与成功时返回的元组签名相匹配
        return None, None, None, None, None, None, None, None, None

    # 加载 adult.test
    try:
        test_data = pd.read_csv(
            'adult.test',
            header=None,
            names=column_names,
            sep=r',\s*',  # 确保逗号后有任意数量空格也能够正常加载
            na_values='?',  # test文件中缺失值以'?'表示
            engine='python',
            skiprows=1  # 跳过第一行描述信息
        )
    except FileNotFoundError:
        print("请确保 adult.test 文件与当前脚本文件在同一目录下")
        # 文件加载失败时，返回None，数量与成功时返回的元组签名相匹配
        return None, None, None, None, None, None, None, None, None

    # 数据清洗，将 data.test 中的收入列的末尾的点去除
    test_data['income'] = test_data['income'].str.replace(r'\.', '',regex=True)

    # 将数据集合测试集合并，方便后续的统一处理
    full_data = pd.concat([data, test_data], ignore_index=True)

    # 将目标变量 income 转换为 0/1
    full_data['income'] = full_data['income'].apply(lambda x: 1 if x == '>50K' else 0)

    # 划分特征和标签
    X = full_data.drop('income', axis=1)
    y = full_data['income']

    # 划分数据集
    # 原始数据集大小
    train_val_size = len(data)

    X_train_val = X.iloc[:train_val_size]
    y_train_val = y.iloc[:train_val_size]
    X_test = X.iloc[train_val_size:]
    y_test = y.iloc[train_val_size:]

    # 进一步将数据集划分为80%训练集和20%验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=66,
                                                      stratify=y_train_val)

    print("数据加载与划分完成：")
    print(f"训练集大小：{len(X_train)}")
    print(f"验证集大小：{len(X_val)}")
    print(f"测试集大小：{len(X_test)}")

    return (X_train, X_val, y_train, y_val, X_test, y_test,
            column_names[:-1], categorical_features, continuous_features)

# 2. ------------ 连续特征离散化 ------------

# (1) 二分法离散化
def find_best_split_for_continuous(feature_valus_non_missing, y_non_missing, non_missing_weights, parent_entropy, entropy_func):
    """
    处理连续特征，寻找最佳二分裂点
    entropy_func：用于接收熵值计算函数
    Returns:(最佳分裂子节点的加权熵，最佳分裂值)
    """

    best_child_entropy = float('inf')
    best_threshold = None

    values = np.unique(feature_valus_non_missing)
    thresholds = (values[:-1] + values[1:])/2   # 获得存储所有候选二分裂点的列表

    if len(thresholds) == 0:
        return parent_entropy, None # 返回父节点的熵，表示信息增益为0

    # 遍历所有的候选分裂点
    for t in thresholds:
        left_mask = (feature_valus_non_missing <= t)
        right_mask = ~left_mask

        # 计算划分后权重之和
        w_left = non_missing_weights[left_mask]
        w_right = non_missing_weights[right_mask]

        total_non_missing_weight = np.sum(non_missing_weights)
        if total_non_missing_weight == 0: continue  # 防止除零错误

        # 计算属于该特征的子节点的权值
        p_left = np.sum(w_left) / total_non_missing_weight
        p_right = 1 - p_left

        if p_left == 0 or p_right == 0: continue

        # 计算当前划分下所有子节点的熵的和
        current_child_entropy = (p_left * entropy_func(y_non_missing[left_mask], w_left)
                                 + p_right * entropy_func(y_non_missing[right_mask], w_right))

        # 更新最优分裂点：在父节点熵确定的情况下，信息增益最大化等效于子节点熵和最小化
        if current_child_entropy < best_child_entropy:
            best_child_entropy = current_child_entropy
            best_threshold = t

    return best_child_entropy, best_threshold

# (2) 基于K-Means的分箱策略(采用4分箱策略)
def discretize_features_kmeans(X_train, X_val, X_test, continuous_features, n_bins=4):
    """
    :param X_train: 训练集特征
    :param X_val: 验证集特征
    :param X_test: 测试集特征
    :param continuous_features: 待分箱的连续特征列名
    :param n_bins: 分箱数量
    :return: 经分箱处理后的新的DataFrame (X_train_binned, X_val_binned, X_test_binned)
    """

    # 创建数据副本
    X_train_binned = X_train.copy()
    X_val_binned = X_val.copy()
    X_test_binned = X_test.copy()

    print("开始对连续特征进行KMeans分箱...")

    # 忽略UserWarning，保证输出整洁
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # 遍历每一个连续特征，进行分箱操作
        for feature in continuous_features:
            print(f"处理特征{feature}...")

            # 准备模型所需训练数据
            # 选出只包含一列的DataFrame，并去除缺失值
            train_data_for_fit = X_train[[feature]].dropna()

            if train_data_for_fit.empty:
                print(f"特征{train_data_for_fit}在训练集中没有有效值，跳过")
                continue

            # 训练模型
            kmeans = KMeans(n_clusters = n_bins, random_state=66, n_init='auto')
            kmeans.fit(train_data_for_fit)

            # 创建有意义的箱体标签
            # 对聚类中心点进行排序
            sorted_centers = sorted(kmeans.cluster_centers_.flatten())

            # 计算箱体边界：取相邻两个中心点作为边界
            boundaries = [-np.inf] + [(sorted_centers[i] + sorted_centers[i+1])/2 for i in
                                      range(len(sorted_centers)-1)] + [np.inf]

            # 根据边界创建有意义的标签
            labels = [f"Bin_{i+1} ({boundaries[i]:.1f} to {boundaries[i+1]:.1f})" for i in range(len(sorted_centers))]

            # 创建原始簇索引到有序等级的映射
            # 先创建一个字典，将排序后的中心点的值映射到它们的有序等级
            center_rank_map = {center_val:rank for rank,center_val in enumerate(sorted_centers)}

            # 利用上一步的映射字典，将原始簇索引映射到等级索引
            cluster_idx_to_rank_map = {
                cluster_idx:center_rank_map[center_val]
                for cluster_idx, center_val in enumerate(kmeans.cluster_centers_.flatten())
            }

            # 将训练好的模型应用到所有数据集上
            for df in [X_train_binned, X_val_binned, X_test_binned]:
                # 记录原始缺失值位置，便于后续处理时保持不变
                nan_mask = df[feature].isnull()

                # 将数据类型又由数字转换为object，便于后续能够存入字符串类型标签
                df[feature] = df[feature].astype(object)

                # 确保当前数据集有非缺失值可供预测
                if not df.loc[~nan_mask, [feature]].empty:
                    # 获得原始簇索引
                    predicted_clusters = kmeans.predict(df.loc[~nan_mask, [feature]])

                    # 利用之前构建的映射字典，将无序簇索引转换为有序等级
                    ranked_bins = pd.Series(predicted_clusters).map(cluster_idx_to_rank_map).values

                    # 根据有序等级和标签列表，生成分类数据并赋值回DataFrame中非缺失值位置
                    df.loc[~nan_mask, feature] = pd.Categorical.from_codes(ranked_bins, categories=labels)

    print("所有连续特征分箱完成")

    # 返回处理后的数据集
    return X_train_binned, X_val_binned, X_test_binned

# 3. ------------ 决策树构建与ID3算法实现 ------------
class TreeNode:
    """决策树节点类"""

    def __init__(self):
        self.feature_idx = None
        self.feature_name = None
        self.is_continuous = False
        self.threshold = None   # 用于存储连续特征分离阈值
        self.children = {}  # 存储子节点的字典，key为分裂条件或特征取值，value为treenode
        self.label = None   # 节点的最终预测类别
        self.num_samples = 0    # 记录到达这个节点的样本权重，用于处理缺失值和剪枝
        self.class_distribution = {}    # 字典，存储到达该节点的样本中不同类别样本的权重，用于熵值计算和确定节点标签

class ID3DecisionTree:
    """ID3决策树分类器，处理连续值/缺失值，并支持后剪枝操作"""

    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split  # 一个节点允许被分裂的最小样权重
        self.max_depth = max_depth
        self.root = None
        self.feature_names = None   # 训练数据特征名称列表
        self.categorical_features = None    # 离散特征名称列表
        self.continuous_features = None

    def _calculate_entropy(self, y, weights):
        """
        信息熵计算函数
        :param y: 标签列
        :param weights: 样本的权重列
        :return: 信息熵
        """

        # 默认样本权重为1
        if weights is None: weights = np.ones(len(y))

        total_weights = np.sum(weights)
        if total_weights == 0: return 0 # 防止除零错误

        entropy = 0
        labels = np.unique(y)

        # 遍历所有类
        for label in labels:
            p_k = np.sum(weights[y == label]) / total_weights
            if p_k > 0: entropy -= p_k * np.log2(p_k)

        return entropy

    def _calculate_info_gain(self, X, y, weights, feature_idx):
        """
        计算指定特征的信息增益，处理离散/连续特征以及缺失值
        :param X: 当前节点训练数据特征
        :param y: 当前节点训练数据标签
        :param weights: 当前节点权重
        :param feature_idx: 特征索引
        :return: 信息增益以及连续特征最佳的分裂阈值
        """

        # 基本信息提取
        feature_name = self.feature_names[feature_idx]
        is_continuous = feature_name in self.continuous_features
        feature_values = X.iloc[:, feature_idx]

        # 缺失值处理
        non_missing_mask = feature_values.notna()

        # 若一个特征的所有值均缺失，则信息增益为0，直接返回
        if np.sum(non_missing_mask) == 0:
            return 0, None

        # 计算样本总权重
        total_weight = np.sum(weights)
        # 筛选非缺失值样本对应的权重
        non_missing_weights = weights[non_missing_mask]
        # 确定样本权重系数
        rho = np.sum(non_missing_weights) / total_weight

        # 计算父节点信息熵
        # 筛选非缺失样本对应的标签
        y_non_missing = y[non_missing_mask]
        # 熵值计算只在非缺失的样本子集上进行
        parent_entropy = self._calculate_entropy(y_non_missing, non_missing_weights)

        # 根据特征类型，计算子节点的加权平均信息熵
        child_entropy = 0
        best_threshold = None
        feature_values_non_missing = feature_values[non_missing_mask]

        # 处理连续特征
        if is_continuous:
            #调用二分法辅助函数
            child_entropy, best_threshold = find_best_split_for_continuous(
                feature_values_non_missing,
                y_non_missing,
                non_missing_weights,
                parent_entropy,
                entropy_func=self._calculate_entropy
            )
        else:
            #处理离散特征
            values = np.unique(feature_values[non_missing_mask])
            # 遍历该离散特征所有可能的取值，计算信息增益
            for value in values:
                # 找到当前特征值对应的样本
                mask = (feature_values[non_missing_mask] == value)
                # 获取这些样本对应的权重
                w_v = non_missing_weights[mask]

                total_non_missing_weight = np.sum(non_missing_weights)
                if total_non_missing_weight == 0: continue  # 避免除零错误

                # 计算当前特征值对应的样本的权重比例
                r_v = np.sum(w_v) / total_non_missing_weight
                # 累加信息熵
                child_entropy += r_v * self._calculate_entropy(y_non_missing[mask], w_v)

        # 计算信息增益
        info_gain = rho * (parent_entropy - child_entropy)

        return info_gain, best_threshold

    def _select_best_feature(self, X, y, weights, feature_indices):
        """
        最优划分属性选择函数
        :param X: 当前节点训练数据特征
        :param y: 当前节点训练数据标签
        :param weights: 当前节点权重
        :param feature_indices: 候选特征索引编号
        :return: 最佳特征索引，最佳连续特征阈值
        """

        best_gain = -1
        best_feature_idx = -1
        best_threshold = None

        # 遍历所有候选特征，选择信息增益最大的特征作为划分特征
        for idx in feature_indices:
            gain, threshold = self._calculate_info_gain(X, y, weights, idx)

            # 更新最大信息增益和划分阈值
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = idx
                best_threshold = threshold

        return best_feature_idx, best_threshold

    def _get_majority_class(self, y, weights):
        """确定该节点标签：按照权重计算并返回权重大的样本对应的标签"""

        # 字典，存储每个类别的总权重
        class_weights = {}

        # 遍历所有样本
        for label, weight in zip(y, weights):
            class_weights[label] = class_weights.get(label, 0) + weight

        if not class_weights: return 0  # 节点为空，返回默认类别0

        # 比较字典的值，并最终返回键，即类别标签
        return max(class_weights, key=class_weights.get)

    def fit(self, X, y, feature_names, categorical_features, continuous_features):
        """决策树模型训练函数"""
        self.feature_names = feature_names
        # 确保输入存在的特征名称
        self.categorical_features = [f for f in categorical_features if f in feature_names]
        self.continuous_features = [f for f in continuous_features if f in feature_names]

        # 初始化样本权重
        initial_weights = np.ones(len(y))
        # 初始化特征索引
        feature_indices = list(range(X.shape[1]))

        # 递归构建决策树
        self.root = self._built_tree(X, y, initial_weights, feature_indices, depth=0)

    def _built_tree(self, X, y, weights, feature_indices, depth):
        """决策树递归构建函数"""

        # 创建当前节点并填充基本信息
        node = TreeNode()
        node.num_samples = np.sum(weights)
        # 计算个类别的样本分布权重
        node.class_distribution = {label:np.sum(weights[y == label]) for label in np.unique(y)}
        # 为当前节点添加一个多数类标签
        node.label = self._get_majority_class(y, weights)

        # 递归出口
        if (len(np.unique(y)) <= 1 or  # 1.当前节点所有样本都属于同一类别
            len(feature_indices) == 0 or # 2.没有可用特征进行分裂(分支节点的数据子集为空集包含在此情形中)
            (self.max_depth is not None and depth >= self.max_depth) or # 3.达到了预设深度
            node.num_samples < self.min_samples_split): # 4.节点权重小于预设的最小权重
            return node

        # 选择最佳划分特征
        best_feature_idx, best_threshold = self._select_best_feature(X, y, weights, feature_indices)

        # 若函数返回-1，意味着所有选择所有特征均不能带来信息增益，此时也设置为停止分裂
        if best_feature_idx == -1:
            return node

        # 填充决策节点的信息
        node.feature_idx = best_feature_idx
        node.feature_name = self.feature_names[best_feature_idx]
        node.is_continuous = node.feature_name in self.continuous_features
        node.threshold = best_threshold # 如果是连续特征，填充最佳分裂阈值

        # 划分数据集
        # 处理缺失值，将训练数据划分为有/无缺失值两部分
        feature_values = X.iloc[:, best_feature_idx]
        non_missing_mask = feature_values.notna()   # 构建非缺失值掩码
        missing_mask = ~non_missing_mask    # 缺失值掩码
        X_non_missing, y_non_missing, w_non_missing = X[non_missing_mask], y[non_missing_mask], weights[non_missing_mask]
        X_missing, y_missing, w_missing = X[missing_mask], y[missing_mask], weights[missing_mask]

        # 移除已经使用过的特征，得到下一次递归使用的候选特征
        remaining_features_indices = [i for i in feature_indices if i != best_feature_idx]

        # 根据特征类型，分别创建连续/离散分裂掩码
        if node.is_continuous:
            left_mask = (X_non_missing.iloc[:, best_feature_idx] <= best_threshold)
            right_mask = ~left_mask
            # 创建分支字典，key为分支名，value为对应的数据掩码，离散特征同理
            split_masks = {'<=' + str(best_threshold):left_mask, '>' + str(best_threshold):right_mask}
        else:
            # 离散特征
            unique_values = np.unique(X_non_missing.iloc[:, best_feature_idx])
            split_masks = {value:(X_non_missing.iloc[:, best_feature_idx] == value) for value in unique_values}

        # 计算缺失值分配权重比例
        # 未缺失样本总权重
        total_w_non_missing = np.sum(w_non_missing)
        # 计算分支应得权重
        branch_proportions = {key:np.sum(w_non_missing[mask]) / total_w_non_missing
                              for key, mask in split_masks.items() if total_w_non_missing > 0}

        # 遍历所有分支，递归生成子树
        for value, mask in split_masks.items():
            # 该特征分支下没有任何样本，创建一个叶结点
            if np.sum(w_non_missing[mask]) == 0:
                leaf = TreeNode()
                leaf.label = node.label # 叶结点标签与父节点保持相同
                node.children[value] = leaf
                continue

            # 组合子节点的数据集 = 该分支非缺失样本 + 按比例分配的缺失样本
            X_child = pd.concat([X_non_missing[mask], X_missing]) if not X_missing.empty else X_non_missing[mask]
            y_child = pd.concat([y_non_missing[mask], y_missing]) if not y_missing.empty else y_non_missing[mask]
            # 组合子节点权重 = 该分支非缺失样本权重和 + 按比例分配的缺失样本权重和
            w_child_non_missing = w_non_missing[mask]
            w_child_missing = w_missing * branch_proportions.get(value, 0)
            w_child = np.concatenate([w_child_non_missing, w_child_missing])

            # 递归调用
            # 树深度+1
            node.children[value] = self._built_tree(X_child, y_child, w_child, remaining_features_indices, depth + 1)

        # 返回最终构建完成的决策树根节点
        return node

    def predict(self, x):
        """对数据集X进行预测"""
        return np.array([self._predict_single(x, self.root) for _, x in x.iterrows()])

    def _predict_single(self, x, node):
        """
        预测单个样本的标签
        :param x: 单个样本的特征数据
        :param node: 当前所在的节点
        :return: 预测的类别标签
        """
        # 递归出口
        if not node.children:
            return node.label   # 节点为叶子，直接返回标签

        # 获取样本在当前分裂特征上的值
        feature_value = x.iloc[node.feature_idx]

        # 处理缺失值
        if pd.isna(feature_value):
            # 使用加权投票，选择子节点中权重最高的预测结果
            predictions = {}    # 存储每个可能的预测结果的总权重
            # 计算所有子节点的总权重
            total_weight = sum(child.num_samples for child in node.children.values())
            if total_weight == 0:
                return node.label

            # 遍历所有子节点
            for child_node in node.children.values():
                # 递归方式预测样本标签
                pred = self._predict_single(x, child_node)
                weight = child_node.num_samples / total_weight  # 计算加权权重
                # 将权重累加到预测结果上
                predictions[pred] = predictions.get(pred, 0) + weight

            # 投票结束，返回权重比例最高的标签
            return max(predictions, key=predictions.get) if predictions else node.label

        # 根据特征值，选择下一个要进入的分支
        # 连续特征
        if node.is_continuous:
            # 得出构建的分支名称 (修正：移除空格以匹配_built_tree中的键)
            key = '<=' + str(node.threshold) if feature_value <= node.threshold else '>' + str(node.threshold)
            # 获取下一个要进入的子节点
            child_node = node.children.get(key)
        else:   # 离散特征
            child_node = node.children.get(feature_value)

        # 处理未知值
        if child_node is None:
            return node.label

        # 递归调用
        return self._predict_single(x, child_node)

    def score(self, X, y):
        """计算给定数据集上的准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_decision_path(self, x):
        """
        追踪单个样本的决策路径
        :param x: 单个样本
        :return: (预测标签, 路径描述列表)
        """
        path = []
        node = self.root

        while node.children:
            # 获取当前节点用于分裂的特征值
            feature_value = x.iloc[node.feature_idx]

            # 处理缺失值
            if pd.isna(feature_value):
                path.append(f"遇到特征 '{node.feature_name}' 的缺失值，基于当前节点多数类进行预测")
                return node.label, path

            next_node = None
            # 根据特征类型（连续/离散）确定路径
            if node.is_continuous:
                if feature_value <= node.threshold:
                    key = '<=' + str(node.threshold)
                    path.append(f"{node.feature_name} ({feature_value:.2f}) <= {node.threshold:.2f}")
                else:
                    key = '>' + str(node.threshold)
                    path.append(f"{node.feature_name} ({feature_value:.2f}) > {node.threshold:.2f}")
                next_node = node.children.get(key)
            else:  # 离散特征
                key = feature_value
                path.append(f"{node.feature_name} = '{key}'")
                next_node = node.children.get(key)

            # 处理未知特征值，即训练时未见过的分支
            if next_node is None:
                path.append(f"遇到未知值 '{feature_value}' (对于特征 '{node.feature_name}'), 基于当前节点多数类进行预测")
                return node.label, path

            node = next_node

        # 到达叶节点
        return node.label, path

# 4. ------------ 决策树剪枝与文本可视化 ------------

def prune_tree(tree, X_val, y_val):
    """决策树剪枝函数：以准确率为评估指标进行剪枝"""
    # 深拷贝构建原始决策树的副本
    pruned_tree = copy.deepcopy(tree)

    # 调用辅助递归函数
    _prune_recursive(pruned_tree.root, pruned_tree, X_val, y_val)

    # 返回剪枝后的树
    return pruned_tree

def _prune_recursive(node, tree, X_val, y_val):
    """辅助函数，递归执行剪枝操作"""
    # 若当前节点为叶结点，则直接返回
    if not node.children:
        return

    # 后续遍历所有子树进行剪枝
    for child in list(node.children.values()):
        _prune_recursive(child, tree, X_val, y_val)

    # 尝试剪枝当前节点
    # 计算剪枝前准确率
    accurancy_before = tree.score(X_val, y_val)

    # 进行剪枝
    original_children = node.children.copy()
    node.children = {}

    # 计算剪枝后准确率
    accurancy_after = tree.score(X_val, y_val)

    # 进行是否剪枝决策
    if accurancy_before > accurancy_after:
        # 撤销剪枝操作
        node.children = original_children

def visualize_tree(tree):
    """决策树可视化函数：以文本形式打印出决策树结构"""
    print("决策树可视化(文本可视化)")

    # 判断决策树是否构建
    if tree.root is None:
        print("决策树尚未构建")
        return

    _visualize_recursive(tree.root, indent="  ")

def _visualize_recursive(node, indent):
    """辅助函数，递归实现决策树可视化"""
    # 如果是叶节点，打印其信息并返回
    if not node.children:
        print(f"{indent}└── 叶节点: 预测类别 = {node.label} (样本权重: {node.num_samples:.2f})")
        return

    # 打印当前决策节点的分裂特征信息
    if node.is_continuous:
        print(f"{indent}├── 分裂特征: {node.feature_name} (阈值: {node.threshold})")
    else:
        print(f"{indent}├── 分裂特征: {node.feature_name} (离散)")

    # 遍历所有子节点
    for i, (value, child) in enumerate(node.children.items()):
        branch = "└──" if i == len(node.children) - 1 else "├──"
        print(f"{indent}{branch} 分支: {value}")
        # 递归调用，并增加下一层的缩进
        _visualize_recursive(child, indent + "  |")

def count_nodes(node):
    """递归方式计算树节点总数"""
    if not node: return 0
    count = 1   # 记录当前节点
    if node.children:
        for child in node.children.values():
            count += count_nodes(child)

    return count

def get_tree_depth(node):
    """计算树的最大深度"""
    if not node: return 0
    if not node.children: return 1
    return 1 + max((get_tree_depth(child) for child in node.children.values()), default=0)

def show_decision_paths(model, X, y, num_samples=10):
    """
    展示数据集中部分样本的决策路径
    :param model: 训练好的ID3DecisionTree模型
    :param X: 特征数据集
    :param y: 标签数据集
    :param num_samples: 要展示的样本数量
    """
    print(f"\n--- 展示 {num_samples} 个样本的决策路径 ---")
    # 确保索引对齐
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # 将 0/1 标签映射回可读的字符串
    label_map = {0: '<=50K', 1: '>50K'}

    # 随机选择样本或选择前几个样本
    sample_indices = X.head(num_samples).index

    for i, idx in enumerate(sample_indices):
        sample_x = X.loc[idx]
        sample_y_true = y.loc[idx]

        # 获取决策路径
        pred_label, path = model.get_decision_path(sample_x)

        # 格式化输出
        print(f"\n样本 {i+1} (索引: {idx}):")
        if not path:
            print("  决策路径: 直接到达根节点（叶节点）")
        else:
            print(f"  决策路径: {' -> '.join(path)}")

        print(f"  预测结果: {label_map[pred_label]}")
        print(f"  实际结果: {label_map[sample_y_true]}")

        if pred_label == sample_y_true:
            print("  预测正确")
        else:
            print("  预测错误")

# 5. ------------ 实验结果图形可视化 ------------

def visualize_comparison_results(results):
    """使用matplotlib制作对比图表"""
    # 设置全局字体，以支持中文标签
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})

    # 提取数据
    labels = ['剪枝前', '剪枝后']
    s1_accuracy = [results['test_acc_before_s1'], results['test_acc_after_s1']]
    s2_accuracy = [results['test_acc_before_s2'], results['test_acc_after_s2']]
    s1_nodes = [results['nodes_before_s1'], results['nodes_after_s1']]
    s2_nodes = [results['nodes_before_s2'], results['nodes_after_s2']]
    s1_depth = [results['depth_before_s1'], results['depth_after_s1']]
    s2_depth = [results['depth_before_s2'], results['depth_after_s2']]

    x = np.arange(len(labels))  # 标签位置
    width = 0.35  # 条形图宽度

    # 创建一个 1x3 的子图布局
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('不同连续值处理策略下决策树性能对比', fontsize=22, fontweight='bold')

    # 子图 1: 测试集准确率
    ax1 = axes[0]
    rects1_acc = ax1.bar(x - width / 2, s1_accuracy, width, label='策略1: 二分法', color='#0072B2', alpha=0.8)
    rects2_acc = ax1.bar(x + width / 2, s2_accuracy, width, label='策略2: KMeans分箱', color='#D55E00', alpha=0.8)
    ax1.set_ylabel('准确率', fontweight='bold')
    ax1.set_title('测试集准确率对比', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold')
    ax1.set_ylim(0.8, 0.88)  # 设置合适的Y轴范围
    ax1.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax1.bar_label(rects1_acc, padding=3, fmt='%.4f')
    ax1.bar_label(rects2_acc, padding=3, fmt='%.4f')

    # 子图 2: 节点总数
    ax2 = axes[1]
    rects1_nodes = ax2.bar(x - width / 2, s1_nodes, width, color='#0072B2', alpha=0.8)
    rects2_nodes = ax2.bar(x + width / 2, s2_nodes, width, color='#D55E00', alpha=0.8)
    ax2.set_ylabel('节点总数', fontweight='bold')
    ax2.set_title('模型复杂度 (节点总数)', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontweight='bold')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax2.bar_label(rects1_nodes, padding=3)
    ax2.bar_label(rects2_nodes, padding=3)

    # 子图 3: 树的深度
    ax3 = axes[2]
    rects1_depth = ax3.bar(x - width / 2, s1_depth, width, color='#0072B2', alpha=0.8)
    rects2_depth = ax3.bar(x + width / 2, s2_depth, width, color='#D55E00', alpha=0.8)
    ax3.set_ylabel('深度', fontweight='bold')
    ax3.set_title('模型复杂度 (树的深度)', fontsize=16, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontweight='bold')
    ax3.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax3.bar_label(rects1_depth, padding=3)
    ax3.bar_label(rects2_depth, padding=3)

    # 移除子图的上和右边框
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 添加统一图例
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=14, frameon=False)

    # 调整布局以防止标签重叠
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 显示图表
    plt.show()

# 6. ------------ 主执行模块 ------------

if __name__ == "__main__":
    # 1. 加载数据
    (X_train, X_val, y_train, y_val, X_test, y_test,
     feature_names, cat_features, cont_features) = load_adult_data()

    if X_train is not None:
        # 策略 1: 连续值使用二分法处理
        print("\n策略 1: 连续值使用二分法处理：")

        # 决策树构建
        tree_s1 = ID3DecisionTree(min_samples_split=20, max_depth=10)
        tree_s1.fit(X_train, y_train, feature_names, cat_features, cont_features)

        print("\n--- 评估剪枝前模型 (策略 1) ---")
        train_acc_before_s1 = tree_s1.score(X_train, y_train)
        val_acc_before_s1 = tree_s1.score(X_val, y_val)
        test_acc_before_s1 = tree_s1.score(X_test, y_test)
        nodes_before_s1 = count_nodes(tree_s1.root)
        depth_before_s1 = get_tree_depth(tree_s1.root)
        print(f"决策树复杂度: 树深度 = {depth_before_s1} , 节点总数 = {nodes_before_s1}")
        print(f"训练集准确率: {train_acc_before_s1:.4f}")
        print(f"验证集准确率: {val_acc_before_s1:.4f}")
        print(f"测试集准确率: {test_acc_before_s1:.4f}")

        # 文本可视化
        visualize_tree(tree_s1)

        print("\n--- 正在对决策树进行后剪枝 (策略 1) ---")
        pruned_tree_s1 = prune_tree(tree_s1, X_val, y_val)

        print("\n--- 评估剪枝后模型 (策略 1) ---")
        train_acc_after_s1 = pruned_tree_s1.score(X_train, y_train)
        val_acc_after_s1 = pruned_tree_s1.score(X_val, y_val)
        test_acc_after_s1 = pruned_tree_s1.score(X_test, y_test)
        nodes_after_s1 = count_nodes(pruned_tree_s1.root)
        depth_after_s1 = get_tree_depth(pruned_tree_s1.root)
        print(f"决策树复杂度: 树深度 = {depth_after_s1} , 节点总数 = {nodes_after_s1}")
        print(f"训练集准确率: {train_acc_after_s1:.4f}")
        print(f"验证集准确率: {val_acc_after_s1:.4f}")
        print(f"测试集准确率: {test_acc_after_s1:.4f}")

        # 文本可视化
        visualize_tree(pruned_tree_s1)

        # 策略 2: 连续值使用KMeans分箱处理
        print("\n策略 2: 连续值使用 KMeans 分箱处理：")

        # 连续值分箱
        X_train_binned, X_val_binned, X_test_binned = discretize_features_kmeans(
            X_train, X_val, X_test, cont_features, n_bins=4
        )
        binned_cat_features = cat_features + cont_features
        binned_cont_features = []

        # 决策树构建
        print("\n--- 正在构建和训练初始决策树 (max_depth=10, 基于分箱数据) ---")
        tree_s2 = ID3DecisionTree(min_samples_split=20, max_depth=10)
        tree_s2.fit(X_train_binned, y_train, feature_names, binned_cat_features, binned_cont_features)

        print("\n--- 评估剪枝前模型 (策略 2) ---")
        train_acc_before_s2 = tree_s2.score(X_train_binned, y_train)
        val_acc_before_s2 = tree_s2.score(X_val_binned, y_val)
        test_acc_before_s2 = tree_s2.score(X_test_binned, y_test)
        nodes_before_s2 = count_nodes(tree_s2.root)
        depth_before_s2 = get_tree_depth(tree_s2.root)
        print(f"决策树复杂度: 树深度 = {depth_before_s2} , 节点总数 = {nodes_before_s2}")
        print(f"训练集准确率: {train_acc_before_s2:.4f}")
        print(f"验证集准确率: {val_acc_before_s2:.4f}")
        print(f"测试集准确率: {test_acc_before_s2:.4f}")

        # 文本可视化
        visualize_tree(tree_s2)

        print("\n--- 正在对决策树进行后剪枝 (策略 2) ---")
        pruned_tree_s2 = prune_tree(tree_s2, X_val_binned, y_val)

        print("\n--- 评估剪枝后模型 (策略 2) ---")
        train_acc_after_s2 = pruned_tree_s2.score(X_train_binned, y_train)
        val_acc_after_s2 = pruned_tree_s2.score(X_val_binned, y_val)
        test_acc_after_s2 = pruned_tree_s2.score(X_test_binned, y_test)
        nodes_after_s2 = count_nodes(pruned_tree_s2.root)
        depth_after_s2 = get_tree_depth(pruned_tree_s2.root)
        print(f"决策树复杂度: 树深度 = {depth_after_s2} , 节点总数 = {nodes_after_s2}")
        print(f"训练集准确率: {train_acc_after_s2:.4f}")
        print(f"验证集准确率: {val_acc_after_s2:.4f}")
        print(f"测试集准确率: {test_acc_after_s2:.4f}")

        # 文本可视化
        visualize_tree(pruned_tree_s2)

        # 最终结果对比
        print("\n最终对比: 二分法 (Binary) vs. KMeans 分箱 (Binned)：")
        print(f"{'Metric':<25} | {'Binary Split (S1)':<20} | {'KMeans Binned (S2)':<20}")
        print("-" * 70)
        print("--- 剪枝前 ---")
        print(f"{'Tree Depth':<25} | {depth_before_s1:<20} | {depth_before_s2:<20}")
        print(f"{'Node Count':<25} | {nodes_before_s1:<20} | {nodes_before_s2:<20}")
        print(f"{'Test Accuracy':<25} | {test_acc_before_s1:.4f}{'':<14} | {test_acc_before_s2:.4f}{'':<14}")
        print("-" * 70)
        print("--- 剪枝后 ---")
        print(f"{'Tree Depth':<25} | {depth_after_s1:<20} | {depth_after_s2:<20}")
        print(f"{'Node Count':<25} | {nodes_after_s1:<20} | {nodes_after_s2:<20}")
        print(f"{'Test Accuracy':<25} | {test_acc_after_s1:.4f}{'':<14} | {test_acc_after_s2:.4f}{'':<14}")
        print("=" * 70)

        # 决策路径展示
        print("\n决策路径展示如下：")

        # 策略1 (二分法) 的决策路径
        print("\n---------- 策略 1 (二分法) 决策路径示例 ----------")
        show_decision_paths(pruned_tree_s1, X_test, y_test, num_samples=5)

        # 策略2 (KMeans分箱) 的决策路径
        print("\n---------- 策略 2 (KMeans分箱) 决策路径示例 ----------")
        show_decision_paths(pruned_tree_s2, X_test_binned, y_test, num_samples=5)

        # 结果对比可视化
        # 收集所有结果到一个字典中，传递给可视化函数
        results_for_viz = {
            'test_acc_before_s1': test_acc_before_s1, 'test_acc_after_s1': test_acc_after_s1,
            'nodes_before_s1': nodes_before_s1, 'nodes_after_s1': nodes_after_s1,
            'depth_before_s1': depth_before_s1, 'depth_after_s1': depth_after_s1,
            'test_acc_before_s2': test_acc_before_s2, 'test_acc_after_s2': test_acc_after_s2,
            'nodes_before_s2': nodes_before_s2, 'nodes_after_s2': nodes_after_s2,
            'depth_before_s2': depth_before_s2, 'depth_after_s2': depth_after_s2,
        }
        visualize_comparison_results(results_for_viz)