import os
import h5py
import numpy as np
from sklearn.neighbors import KDTree
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from eva import evaluate_model
from prepare_train_data import list_h5_files, load_data_from_h5, prepare_seasurface_land_data, balanced_class_split
from get_features_sea_land import extract_features_xyz
from xgboost.callback import TrainingCallback

class CustomEarlyStop(TrainingCallback):
    def __init__(self, logloss_thresh=0.2, error_thresh=0.05):
        self.logloss_thresh = logloss_thresh
        self.error_thresh = error_thresh

    def after_iteration(self, model, epoch, evals_log):
        # 取验证集(eval)的 logloss 和 merror
        logloss = evals_log['eval']['mlogloss'][-1]
        merror  = evals_log['eval']['merror'][-1]

        # 如果达到目标精度则提前停止
        if logloss <= self.logloss_thresh or merror <= self.error_thresh:
            print(f"✅ 达到目标精度 (logloss={logloss:.4f}, merror={merror:.4f})，提前停止")
            return True
        return False

sys.dont_write_bytecode = True

def main():
    root_dir = r'E:/DATA_0714/flight-02/train_sea_land'
    start_idx = 0
    end_idx = 4

    files = list_h5_files(root_dir, start_idx, end_idx)
    print(f"将处理 {len(files)} 个文件...")

    all_features = []
    all_labels = []
    all_points = []  # 保存原始坐标

    for file in files:
        print(f"处理文件: {file}")
        ref_data, _ = load_data_from_h5(file)
        points, labels = prepare_seasurface_land_data(ref_data,sea_height=-10.0)
        feats, feature_names = extract_features_xyz(points, voxel_size=(0.25,0.5,1), k=10, mean_surface_height=-10.0)
        if feats.shape[0] == 0:
            print("该文件无训练数据，跳过")
            continue
        all_features.append(feats)
        all_labels.append(labels)
        all_points.append(points)

    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)
    all_points = np.vstack(all_points)
    print(f"总训练样本数: {all_features.shape[0]}")
    print("特征列名:", feature_names)

    # 划分数据
    X_train, X_val, y_train, y_val, pts_train, pts_val = balanced_class_split(
    all_features, all_labels, all_points,
    test_size=0.2,
    random_state=42
    )


    # 转换为 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    params = {
        'max_depth': 8,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'num_class': 3,  # 0: Other, 1: Sea Surface, 2: Land
        'eval_metric': ['mlogloss','merror'],
        'tree_method': 'hist'
    }

    evallist = [(dtrain, 'train'), (dval, 'eval')]



    # ====== 训练 ======
    model = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dtrain,'train'), (dval,'eval')],
    callbacks=[CustomEarlyStop(logloss_thresh=0.2, error_thresh=0.05)]
)


    # 保存新模型（带时间戳）
    model_name = f"land_sea_classifier_tz.model"

    val_preds = model.predict(dval).argmax(axis=1)
    reverse_map = {0:0,1:1,2:2}
    y_val_original = np.array([reverse_map[int(y)] for y in y_val])
    val_preds_original = np.array([reverse_map[int(p)] for p in val_preds])

    sea_height = np.min(pts_val[val_preds_original==1, 2]) if np.any(val_preds_original==1) else -10.0
    print(f"sea_height = {sea_height}")
    # 保存到模型
    model.set_attr(sea_height=str(sea_height))
    model.save_model(model_name)
    print(f"模型已保存为: {model_name}")

    
    label_names = {0:'Other', 1:'Sea Surface', 2:'land'}
    # 调用统一评估函数
    evaluate_model(
        points_val = pts_val,
        model = model,
        X_val = X_val,
        y_val = y_val,
        label_names = label_names,
        val_preds = val_preds,
        feature_names = feature_names,
        reverse_map = reverse_map,
        title_prefix = "land and water"
    )

if __name__ == "__main__":
    print("\n" + "="*40)
    print("=== 当前运行时间戳:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*40 + "\n")
    main()
