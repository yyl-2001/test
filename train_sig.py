import os
import h5py
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
import matplotlib.pyplot as plt
from train_sea import extract_features_tz
from eva import evaluate_model  # 复用评估函数
from get_features_seabed import extract_underwater_features  # 复用特征提取函数


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

def prepare_train_data(ref_data):
    labels = ref_data[:, 4].astype(int)
    sea_mask = (labels == 1)
    nonsea_mask = ~sea_mask
    selected_mask = sea_mask | nonsea_mask
    points = ref_data[selected_mask, 1:4]  # X,Y,Z
    # 注意这里labels_bin要对应selected_mask过滤后
    labels_bin = np.zeros(points.shape[0], dtype=int)
    # sea_mask 和 selected_mask 均为原始点云长度掩码，需要重构
    sea_mask_filtered = sea_mask[selected_mask]
    labels_bin[sea_mask_filtered] = 1
    return points, labels_bin

def main():
    root_dir = r'E:/DATA_0714/flight-02/train_sig2'
    start_idx = 0
    end_idx = 3

    # 1. 加载已训练的海面分类器
    surface_model = xgb.Booster()
    surface_model.load_model('sea_classifier_tz.model')
    
    # 2. 处理数据
    all_underwater_features = []
    all_underwater_labels = []
    all_underwater_points = []  # 保存原始坐标
    
    files = list_h5_files(root_dir, start_idx, end_idx)
    print(f"将处理 {len(files)} 个文件...")

    for file in files:
        print(f"处理文件: {file}")
        ref_data, ch3_data = load_data_from_h5(file)
        
        points_sea = ref_data[:, [0,3]]  # 时间 + 深度
        points = ref_data[:, 1:4]        # XYZ
        
        # 海面分类器预测
        features_sea, feature_names_sea = extract_features_tz(points_sea, voxel_t=1.0, voxel_z=0.5, k=10)
        dmat = xgb.DMatrix(features_sea, feature_names=feature_names_sea)
        surface_preds = (surface_model.predict(dmat) > 0.5).astype(int)
        
        surface_points = points[surface_preds == 1]
        mean_surface_height = np.mean(surface_points[:, 2]) if len(surface_points) > 0 else 0
        
        # 1. 主通道（points）筛选海面以下点
        underwater_mask_main = points[:, 2] < (mean_surface_height - 0.5)
        underwater_points = points[underwater_mask_main]
        underwater_labels = ref_data[underwater_mask_main, 4].astype(int)

        # 2. 辅助通道（ch3_data）筛选海面以下点
        underwater_mask_ch3 = ch3_data[:, 3] < (mean_surface_height - 0.5)
        underwater_ch3_points = ch3_data[underwater_mask_ch3]
        
        # 提取水下特征
        underwater_features, feature_names_seabed = extract_underwater_features(
            underwater_points, 
            underwater_ch3_points, 
            voxel_scales=(0.25,0.5,1.0),
            neigh_radius=1,
            depth_bins=np.arange(-40,-7,1)  # 水下特征提取时的水面高度
        )
        
        all_underwater_features.append(underwater_features)
        all_underwater_labels.append(underwater_labels)
        all_underwater_points.append(underwater_points)

    # 拼接所有文件
    X = np.vstack(all_underwater_features)
    y = np.hstack(all_underwater_labels)
    points_all = np.vstack(all_underwater_points)

    # 标签映射
    label_map = {0: 0, 1: 1, 2: 2, 4: 0}
    y = np.array([label_map[int(label)] for label in y])

    # 训练/验证集分割
    X_train, X_val, y_train, y_val, pts_train, pts_val = train_test_split(
        X, y, points_all, test_size=0.2, random_state=42, stratify=y
    )

    # 构建 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 训练模型
    params = {
        'max_depth': 8,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'num_class': 3,  # 0: Other, 1: Sea Surface, 2: Seabed
        'eval_metric': ['mlogloss','merror'],
        'tree_method': 'hist'
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain,'train'), (dval,'eval')],
        early_stopping_rounds=30,
        verbose_eval=True
    )
    model.save_model('underwater_classifier.model')

    # 预测
    val_preds = model.predict(dval).argmax(axis=1)
    reverse_map = {0:0,1:1,2:2,3:0}
    y_val_original = np.array([reverse_map[int(y)] for y in y_val])
    val_preds_original = np.array([reverse_map[int(p)] for p in val_preds])

    # 调用统一评估函数
    evaluate_model(
        points_val = pts_val,
        model = model,
        X_val = X_val,
        y_val = y_val,
        val_preds = val_preds,
        feature_names = feature_names_seabed,
        reverse_map = reverse_map,
        title_prefix = "Underwater"
    )

    
if __name__ == "__main__":
    main()
