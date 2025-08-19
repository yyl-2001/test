import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
def evaluate_model(model, X_val, y_val, label_names, val_preds, feature_names, reverse_map, points_val=None, title_prefix="Model"):

    # 映射回原始标签
    y_val_original = np.array([reverse_map[int(y)] for y in y_val])
    val_preds_original = np.array([reverse_map[int(p)] for p in val_preds])

    # 唯一标签
    unique_labels = sorted(set(np.unique(y_val_original)) | set(np.unique(val_preds_original)))
    
    target_names = [f"{label_names[label]} ({label})" for label in unique_labels]

    # --- 1. 分类报告 ---
    print(f"\n=== {title_prefix} Classification Report ===")
    print(classification_report(y_val_original, val_preds_original,
                                labels=unique_labels,
                                target_names=target_names))
    
    # --- 2. 混淆矩阵 ---
    cm = confusion_matrix(y_val_original, val_preds_original, labels=unique_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10,8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.title(f'{title_prefix} Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, target_names, rotation=45, ha='right')
    plt.yticks(tick_marks, target_names)
    
    thresh = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm_normalized.shape):
        plt.text(j, i, f'{cm_normalized[i,j]:.2%}',
                 horizontalalignment='center',
                 color='white' if cm_normalized[i,j] > thresh else 'black')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # --- 3. 特征重要性 ---
    importance_scores = model.get_score(importance_type='gain')
    importance_dict = {feature_names[i]: v for i, v in enumerate(importance_scores.values())}
    sorted_features = sorted(importance_dict.items(), key=lambda x:x[1], reverse=True)
    print(f"\n=== {title_prefix} Feature Importance ===")
    for feat, score in sorted_features:
        print(f"{feat:20s}: {score:.4f}")
    # 绘制特征重要性条形图
    plt.figure(figsize=(12,6))
    feats, scores = zip(*sorted_features)
    plt.bar(feats, scores)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{title_prefix} Feature Importance')
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== {title_prefix} Feature Importance Ranking ===")
    for feat, score in sorted_features:
        print(f"{feat:20s}: {score:.4f}")

    # --- 4. 特征相关性 ---
    plt.figure(figsize=(12,10))
    corr_matrix = np.corrcoef(X_val.T)
    plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title(f'{title_prefix} Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_tz_projection(T, Z, y_true, y_pred, title_prefix="", max_points=50000):
    """
    T-Z 投影图 (时间-深度)，显示 Ground Truth 和 Predictions
    自动抽稀以避免绘图卡死
    输入：
        T: 时间坐标，一维数组，shape=(N,)
        Z: 深度坐标，一维数组，shape=(N,)
        y_true: 真实标签，一维数组，shape=(N,)
        y_pred: 预测标签，一维数组，shape=(N,)
        title_prefix: 图标题前缀
        max_points: 最大绘图点数，超过则随机抽稀
    """
    if T is None or Z is None or y_true is None or y_pred is None:
        print("⚠️ 输入为空，跳过绘图")
        return
    
    N = len(T)
    # --- 抽稀 ---
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        T = T[idx]
        Z = Z[idx]
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        print(f"⚠️ 数据量 {N} 太大，抽稀到 {max_points} 个点进行绘图")
    
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))

    plt.figure(figsize=(15,5))
    # --- Ground Truth ---
    plt.subplot(121)
    scatter = plt.scatter(T, Z, c=y_true, cmap='tab10', s=1, alpha=0.6)
    plt.colorbar(scatter, ticks=unique_labels)
    plt.title(f'{title_prefix} Ground Truth (T-Z)')
    plt.xlabel('Time')
    plt.ylabel('Depth')

    # --- Predictions ---
    plt.subplot(122)
    scatter = plt.scatter(T, Z, c=y_pred, cmap='tab10', s=1, alpha=0.6)
    plt.colorbar(scatter, ticks=unique_labels)
    plt.title(f'{title_prefix} Predictions (T-Z)')
    plt.xlabel('Time')
    plt.ylabel('Depth')

    plt.tight_layout()
    plt.show()


from mpl_toolkits.mplot3d import Axes3D

def plot_xyz_points(X, Y, Z, labels, title_prefix="", max_points=50000):
    """
    绘制三维点云 (X,Y,Z)，按标签着色
    自动抽稀以避免绘图卡死
    输入：
        X, Y, Z: 坐标，一维数组
        labels: 对应标签，一维数组
        title_prefix: 图标题前缀
        max_points: 最大绘图点数
    """
    if X is None or Y is None or Z is None or labels is None:
        print("⚠️ 输入为空，跳过绘图")
        return

    N = len(X)
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        X = X[idx]
        Y = Y[idx]
        Z = Z[idx]
        labels = labels[idx]
        print(f"⚠️ 数据量 {N} 太大，抽稀到 {max_points} 个点进行绘图")

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X, Y, Z, c=labels, cmap='tab10', s=1, alpha=0.6)
    fig.colorbar(scatter, ticks=np.unique(labels))
    ax.set_title(f"{title_prefix} 3D Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
