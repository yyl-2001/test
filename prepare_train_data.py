import os
import h5py
import numpy as np

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

def prepare_seasurface_land_data(ref_data,sea_height):
    # 筛选Z大于sea_height的
    labels = ref_data[:, 4].astype(int)
    labels = labels[ref_data[:, 3] > sea_height]
    ref_data = ref_data[ref_data[:, 3] > sea_height] # 海面以上

    # 把labels == 3的替换为2
    labels[labels == 4] = 0
    labels[labels == 2] = 0
    labels[labels == 3] = 2  # 将标签3替换为2 代表陆地
    # 取 T 和 Z
    points_xyz = ref_data[:, 1:4]  # T,X,Y,Z
    return points_xyz, labels

def balanced_class_split(X, y, points=None, test_size=0.2, random_state=42):
    """
    对每个类别抽取相同数量的样本，保证训练集和验证集均衡。
    
    X: 特征 (N, d)
    y: 标签 (N,)
    points: 可选原始坐标 (N, ?)
    test_size: 验证集比例
    """
    rng = np.random.default_rng(random_state)
    unique_labels = np.unique(y)
    
    # 找每类最少样本数
    min_count = min([np.sum(y == lbl) for lbl in unique_labels])
    
    train_idx = []
    val_idx = []
    
    for lbl in unique_labels:
        idx = np.where(y == lbl)[0]
        rng.shuffle(idx)
        idx = idx[:min_count]  # 每类只取 min_count 个样本
        n_val = max(1, int(min_count * test_size))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    if points is not None:
        pts_train, pts_val = points[train_idx], points[val_idx]
        return X_train, X_val, y_train, y_val, pts_train, pts_val
    else:
        return X_train, X_val, y_train, y_val