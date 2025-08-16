import os
import h5py
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from datetime import datetime
import sys
sys.dont_write_bytecode = True

voxel_t = 1.0  # 时间体素大小
voxel_z = 0.5  # 深度体素大小

# =========================
# 1. 文件读取
# =========================
def list_h5_files(root_dir, start_idx=0, end_idx=None):
    all_files = [f for f in os.listdir(root_dir) if f.endswith('.h5')]
    all_files.sort()
    selected_files = all_files[start_idx:end_idx]
    return [os.path.join(root_dir, f) for f in selected_files]

def load_data_from_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        ref_data = np.array(f['REF_DATA']).T  # Nx5
        ch3_data = np.array(f['ORG_DATA_channel_3']).T  # Nx4, 暂未用
    return ref_data, ch3_data

# =========================
# 2. 数据准备
# =========================
def prepare_train_data(ref_data):
    labels = ref_data[:, 4].astype(int)
    sea_mask = (labels == 1)
    nonsea_mask = ~sea_mask
    selected_mask = sea_mask | nonsea_mask

    # 取 T 和 Z
    points_tz = ref_data[selected_mask][:, [0, 3]]  # T, Z
    labels_bin = np.zeros(points_tz.shape[0], dtype=int)
    labels_bin[sea_mask[selected_mask]] = 1
    return points_tz, labels_bin

# =========================
# 3. 特征提取
# =========================
def extract_features_tz(points, voxel_t, voxel_z, k=10):
    """
    输入: points [T, Z]
    输出特征: [Z, center_density, ratio_center_up, flatness]
    """
    if points.shape[0] == 0:
        return np.empty((0, 4))

    # 1. 体素化
    min_tz = points.min(axis=0)
    ij = np.floor((points - min_tz) / np.array([voxel_t, voxel_z])).astype(int)
    max_ij = ij.max(axis=0) + 1
    voxel_count = np.zeros(max_ij, dtype=np.int32)

    for idx in range(points.shape[0]):
        voxel_count[ij[idx, 0], ij[idx, 1]] += 1

    # 2. 中心密度
    voxel_volume = voxel_t * voxel_z
    current_count = np.maximum(1,voxel_count[ij[:, 0], ij[:, 1]])
    center_density = current_count

    # 3. 中心比上密度
    up_j = ij[:, 1] + 1
    up_valid = up_j < max_ij[1]
    up_count = np.zeros(points.shape[0], dtype=np.int32)
    up_count[up_valid] = np.maximum(1, voxel_count[ij[up_valid, 0], up_j[up_valid]])
    up_density = up_count
    ratio_center_up = center_density / np.maximum(1, up_density)

    # 4. 平坦度
    tree = KDTree(points)
    _, idxs = tree.query(points, k=k+1)  # 包含自己
    z_neighbors = points[idxs, 1]
    flatness = np.std(z_neighbors, axis=1)

    features = np.column_stack([
        points[:, 1],  # Z
        center_density,
        ratio_center_up,
        flatness
    ])
    feature_names = ['Z', 'Center_Density', 'Ratio_Center_Up', 'Flatness']
    return features, feature_names

# =========================
# 4. 主流程
# =========================
def main():
    root_dir = r'E:/DATA_0714/flight-02/train_sig2'
    start_idx = 0
    end_idx = 4

    files = list_h5_files(root_dir, start_idx, end_idx)
    print(f"将处理 {len(files)} 个文件...")

    all_features = []
    all_labels = []
    

    for file in files:
        print(f"处理文件: {file}")
        ref_data, _ = load_data_from_h5(file)
        points, labels = prepare_train_data(ref_data)
        feats, feature_names = extract_features_tz(points, voxel_t, voxel_z, k=10)
        if feats.shape[0] == 0:
            print("该文件无训练数据，跳过")
            continue
        all_features.append(feats)
        all_labels.append(labels)

    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)

    print(f"总训练样本数: {all_features.shape[0]}")
    print("特征列名:", feature_names)

    # 划分数据
    X_train, X_val, y_train, y_val = train_test_split(
        all_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # 转换为 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist'
    }

    evallist = [(dtrain, 'train'), (dval, 'eval')]

    # 重新训练（不会加载旧模型）
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evallist,
        early_stopping_rounds=20,
        verbose_eval=True
    )

    # 保存新模型（带时间戳）
    model_name = f"sea_classifier_tz.model"
    model.save_model(model_name)
    print(f"模型已保存为: {model_name}")

    # 验证集评估
    val_preds = (model.predict(dval) > 0.5).astype(int)
    val_acc = (val_preds == y_val).mean()
    print(f"验证集准确率: {val_acc:.4f}")

    # 特征重要性
    importance_scores = model.get_score(importance_type='gain')
    importance_dict = {fname: importance_scores.get(fname, 0) for fname in feature_names}
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(6,4))
    features_sorted, scores = zip(*sorted_features)
    plt.bar(features_sorted, scores)
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.show()

    print("\nFeature Importance Ranking:")
    for feat, score in sorted_features:
        print(f"{feat:15s}: {score:.4f}")

    # 训练数据统计
    print("\n=== 训练数据统计 ===")
    train_stats = {
        'mean': np.mean(X_train, axis=0),
        'std': np.std(X_train, axis=0),
        'min': np.min(X_train, axis=0),
        'max': np.max(X_train, axis=0)
    }
    for i, fname in enumerate(feature_names):
        print(f"{fname:15s}: mean={train_stats['mean'][i]:.3f}, std={train_stats['std'][i]:.3f}, "
              f"range=[{train_stats['min'][i]:.3f}, {train_stats['max'][i]:.3f}]")

    # 测试第4个文件
    print("\n=== 预测第4个文件 ===")
    test_files = list_h5_files(root_dir, start_idx=5, end_idx=6)
    test_file = test_files[0]
    print(f"预测文件: {test_file}")

    ref_data, _ = load_data_from_h5(test_file)
    points_tz = ref_data[:, [0, 3]]  # 全部点
    true_labels = (ref_data[:, 4].astype(int) == 1).astype(int)

    features,feature_names = extract_features_tz(points_tz, voxel_t, voxel_z, k=10)
    dmat = xgb.DMatrix(features, feature_names=feature_names)
    preds = (model.predict(dmat) > 0.5).astype(int)

    test_acc = (preds == true_labels).mean()
    print(f"测试准确率: {test_acc:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, preds, target_names=['Non-sea', 'Sea Surface']))

    # 测试数据分布统计
    print("\n测试数据统计:")
    for i, fname in enumerate(feature_names):
        test_mean = np.mean(features[:, i])
        test_std = np.std(features[:, i])
        test_min = np.min(features[:, i])
        test_max = np.max(features[:, i])
        print(f"{fname:15s}: mean={test_mean:.3f}, std={test_std:.3f}, range=[{test_min:.3f}, {test_max:.3f}]")
        mean_shift = abs(test_mean - train_stats['mean'][i]) / (train_stats['std'][i] + 1e-6)
        scale_ratio = test_std / (train_stats['std'][i] + 1e-6)
        if mean_shift > 0.5 or scale_ratio < 0.5 or scale_ratio > 2:
            print(f"警告: {fname} 特征分布偏移显著! 均值偏移={mean_shift:.2f}σ, 标准差比={scale_ratio:.2f}")

    # 绘制预测与真实对比 (T-Z 投影)
    
    z_mean = np.mean(ref_data[:,3])
    z_min = z_mean - 2
    z_max = z_mean + 2

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.scatter(points_tz[:, 0], points_tz[:, 1], c=true_labels, cmap='coolwarm', s=1, alpha=0.6)
    plt.colorbar(ticks=[0,1])
    plt.title('Ground Truth (Blue:Non-sea Red:Sea)')
    plt.xlabel('T')
    plt.ylabel('Z')
    plt.ylim(z_min, z_max)

    plt.subplot(122)
    plt.scatter(points_tz[:, 0], points_tz[:, 1], c=preds, cmap='coolwarm', s=1, alpha=0.6)
    plt.colorbar(ticks=[0,1])
    plt.title('Predictions (Blue:Non-sea Red:Sea)')
    plt.xlabel('T')
    plt.ylabel('Z')
    plt.ylim(z_min, z_max)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    print("\n" + "="*40)
    print("=== 当前运行时间戳:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*40 + "\n")
    main()
