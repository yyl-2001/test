
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from eva import evaluate_model  # 复用评估函数
from get_features_seabed import extract_underwater_features  # 复用特征提取函数
from prepare_train_data import list_h5_files, load_data_from_h5, prepare_seasurface_land_data


def main():
    root_dir = r'E:/DATA_0714/flight-02/train_sig3'
    start_idx = 0
    end_idx = 12

    # 1. 加载已训练的海面分类器
    surface_model = xgb.Booster()
    surface_model.load_model('land_sea_classifier_tz.model')
    
    # 2. 处理数据
    all_underwater_features = []
    all_underwater_labels = []
    all_underwater_points = []  # 保存原始坐标
    
    files = list_h5_files(root_dir, start_idx, end_idx)
    print(f"将处理 {len(files)} 个文件...")

    for file in files:
        print(f"处理文件: {file}")
        ref_data, ch3_data = load_data_from_h5(file)
        
        # ====== 2. 准备海面数据 ======
        sea_height_str = surface_model.attr('sea_height')
        sea_height = float(sea_height_str) if sea_height_str is not None else -10.0
        print("从模型读取 sea_height:", sea_height)

        points = ref_data[:, 1:4]        # XYZ
        mean_surface_height = sea_height  # 使用模型中的海面高度
        # 1. 主通道（points）筛选海面以下点
        underwater_mask_main = points[:, 2] < (mean_surface_height + 0.5)
        underwater_points = points[underwater_mask_main]
        underwater_labels = ref_data[underwater_mask_main, 4].astype(int)

        # 2. 辅助通道（ch3_data）筛选海面以下点
        underwater_mask_ch3 = ch3_data[:, 3] < (mean_surface_height + 0.5)
        underwater_ch3_points = ch3_data[underwater_mask_ch3,1:4]
        print('提取水下特征...')
        # 提取水下特征
        underwater_features, feature_names_seabed = extract_underwater_features(
            underwater_points, 
            underwater_ch3_points, 
            H_surface = mean_surface_height - 1,  # 水下特征提取时的水面高度
            voxel_scales=(0.1,0.5,1.0),
            neigh_radius=1,
        )
        
        all_underwater_features.append(underwater_features)
        all_underwater_labels.append(underwater_labels)
        all_underwater_points.append(underwater_points)

    # 拼接所有文件
    X = np.vstack(all_underwater_features)
    y = np.hstack(all_underwater_labels)
    points_all = np.vstack(all_underwater_points)

    # 标签映射
    label_map = {0: 0, 1: 1, 2: 2, 4: 3}
    y = np.array([label_map[int(label)] for label in y])

    # 训练/验证集分割
    X_train, X_val, y_train, y_val, pts_train, pts_val = train_test_split(
        X, y, points_all, test_size=0.1, random_state=42, stratify=y
    )

    # 构建 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 训练模型
    params = {
        'max_depth': 8,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'num_class': 4,  # 0: Other, 1: Sea Surface, 2: Seabed
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
    model.save_model('underwater_classifier3.model')

    # 预测
    val_preds = model.predict(dval).argmax(axis=1)
    reverse_map = {0:0,1:1,2:2,3:4}
    y_val_original = np.array([reverse_map[int(y)] for y in y_val])
    val_preds_original = np.array([reverse_map[int(p)] for p in val_preds])

    # 调用统一评估函数
    evaluate_model(
        points_val = pts_val,
        model = model,
        X_val = X_val,
        y_val = y_val,
        label_names = {0:'Other', 1:'Sea Surface', 2:'Seabed', 4:'Water body'},
        val_preds = val_preds,
        feature_names = feature_names_seabed, 
        reverse_map = reverse_map,
        title_prefix = "Underwater"
    )

    
if __name__ == "__main__":
    main()
