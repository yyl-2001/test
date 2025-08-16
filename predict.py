import h5py
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from get_features_seabed import  extract_underwater_features # 复用函数
from train_sea import extract_features_tz
from eva import evaluate_model  # 复用评估函数

def load_data_from_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        ref_data = np.array(f['REF_DATA']).T  # Nx5
        ch3_data = np.array(f['ORG_DATA_channel_3']).T  # Nx4
    return ref_data, ch3_data


def main():
    # ====== 1. 加载模型 ======
    surface_model = xgb.Booster()
    surface_model.load_model('sea_classifier_tz.model')
    underwater_model = xgb.Booster()
    underwater_model.load_model('underwater_classifier.model')
    
    # ====== 2. 加载测试数据 ======
    test_file = r'E:/DATA_0714/flight-02/train_sig2/L3_20250714190602.h5'
    print(f"\n处理测试文件: {test_file}")
    ref_data, ch3_data = load_data_from_h5(test_file)
    true_labels = ref_data[:, 4].astype(int)
    points = ref_data[:, 1:4]  # XYZ
    points_sea = ref_data[:, [0,3]]  # 时间 + 深度

    # ====== 3. 海面特征 & 预测 ======
    features_sea, feature_names_sea = extract_features_tz(points_sea, voxel_t=1.0, voxel_z=0.5, k=10)
    
    
    dmat_sea = xgb.DMatrix(features_sea, feature_names=feature_names_sea)
    surface_preds = (surface_model.predict(dmat_sea) > 0.5).astype(int)

    surface_points = points[surface_preds == 1]
    mean_surface_height = np.mean(surface_points[:, 2]) if len(surface_points) > 0 else 0

    # 3. 主通道（points）筛选海面以下点
    underwater_mask_main = points[:, 2] < (mean_surface_height - 0.5)
    underwater_points = points[underwater_mask_main]
    underwater_labels = ref_data[underwater_mask_main, 4].astype(int)

    # 4. 辅助通道（ch3_data）筛选海面以下点
    underwater_mask_ch3 = ch3_data[:, 3] < (mean_surface_height - 0.5)
    underwater_ch3_points = ch3_data[underwater_mask_ch3]

    # ====== 5. 水下特征 & 预测 ======
    underwater_preds = np.array([], dtype=int)
    if len(underwater_points) > 0:
        underwater_features, feature_names_underwater = extract_underwater_features(
            underwater_points, 
            underwater_ch3_points, 
            voxel_scales=(0.25, 0.5, 1.0),  # 多尺度体素大小（对应不同“卷积核”）
            neigh_radius=1,         # 3D 邻域半径（=1 即 3×3×3）, 
            depth_bins=np.arange(-40, -7, 1)  # 水下特征提取时的水面高度
        )
    
        dmat_underwater = xgb.DMatrix(underwater_features, feature_names=feature_names_underwater)
        underwater_preds_idx = underwater_model.predict(dmat_underwater).argmax(axis=1)
        label_map = {0: 0, 1: 1, 2: 2, 4: 0}
        underwater_preds = np.array([label_map[p] for p in underwater_preds_idx])

    # ====== 6. 合并预测结果 ======
    final_preds = np.zeros_like(true_labels)
    final_preds[surface_preds == 1] = 1  # 海面
    if len(underwater_preds) > 0:
        final_preds[underwater_mask_main] = underwater_preds

    # ====== 7. 评估 ======
    print("\n=== 评估结果 ===")
    print("实际标签分布:", np.unique(true_labels, return_counts=True))
    print("预测标签分布:", np.unique(final_preds, return_counts=True))

    reverse_map = {0:0,1:1,2:2,4:0}  # 映射回原始标签

    label_names = {0: 'Other', 1: 'Sea Surface', 2: 'Seabed',4: 'Water Body'}
    unique_labels = sorted(set(np.unique(true_labels)) | set(np.unique(final_preds)))
    target_names = [f"{label_names[label]} ({label})" for label in unique_labels]

    print("\n分类报告:")
    print(classification_report(true_labels, final_preds,
                                labels=unique_labels,
                                target_names=target_names))

    evaluate_model(
        points_val = underwater_points,
        model = underwater_model,
        X_val = underwater_features,
        y_val = underwater_labels,
        val_preds = underwater_preds,
        feature_names = feature_names_underwater,
        reverse_map = reverse_map,
        title_prefix = "Underwater"
    )


if __name__ == "__main__":
    main()
