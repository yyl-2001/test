import h5py
import numpy as np
import xgboost as xgb
from get_features_seabed import extract_underwater_features
from prepare_train_data import list_h5_files, prepare_seasurface_land_data
from get_features_sea_land import extract_features_xyz
from eva import plot_tz_projection

def prepare_data(h5_file):
    import os
    import h5py
    import numpy as np

    fname = os.path.basename(h5_file)
    print(f"处理文件: {fname}")

    with h5py.File(h5_file, "r") as f:
        # CH1 数据
        ref_data = np.vstack([
            f["GNSS_SEC_CH1"][:].ravel(), # type: ignore
            f["POINT_X_CH1"][:].ravel(), # type: ignore
            f["POINT_Y_CH1"][:].ravel(), # type: ignore
            f["POINT_Z_CH1"][:].ravel(), # type: ignore
        ]).T

        ref_data = np.hstack([
            ref_data, 
            np.zeros((ref_data.shape[0], 1), dtype=int)  # 占位标签
        ])

        # CH3 数据
        ch3_data = np.vstack([
            f["GNSS_SEC_CH3"][:].ravel(), # type: ignore
            f["POINT_X_CH3"][:].ravel(), # type: ignore
            f["POINT_Y_CH3"][:].ravel(), # type: ignore
            f["POINT_Z_CH3"][:].ravel(), # type: ignore
        ]).T

    return ref_data, ch3_data


def predict_and_save(root_dir, out_dir, start_idx=0, end_idx=None):
    # ====== 1. 加载模型 ======
    surface_model = xgb.Booster()
    surface_model.load_model('land_sea_classifier_tz.model')
    underwater_model = xgb.Booster()
    underwater_model.load_model('underwater_classifier3.model')

    files = list_h5_files(root_dir, start_idx, end_idx)

    for idx, file_path in enumerate(files, start=1):
        print(f"\n处理文件: {file_path}")
        ref_data, ch3_data = prepare_data(file_path)
        N = ref_data.shape[0]
        final_preds = np.zeros(N, dtype=int)

        # ====== 2. 读取水面高度 ======
        sea_height_str = surface_model.attr('sea_height')
        sea_height = float(sea_height_str) if sea_height_str is not None else -10.0

        # ====== 3. 海面数据准备 ======
        points_sea, labels_sea = prepare_seasurface_land_data(ref_data, sea_height)
        labels_sea[labels_sea == 2] = 3  # 陆地
        idx_sea = np.arange(N)[ref_data[:,3] > sea_height]

        if len(points_sea) > 0:
            features_sea, feature_names_sea = extract_features_xyz(points_sea, voxel_size=(0.25,0.5,1), k=10, mean_surface_height=sea_height)
            dmat_sea = xgb.DMatrix(features_sea, feature_names=feature_names_sea)
            pred_labels_sea = surface_model.predict(dmat_sea).argmax(axis=1)
            sea_label_map = {0:0, 1:1, 2:3}
            final_preds[idx_sea] = np.array([sea_label_map[p] for p in pred_labels_sea])
            surface_points = points_sea[pred_labels_sea == 1]
            mean_surface_height = np.mean(surface_points[:,2]) if len(surface_points) > 0 else sea_height
        else:
            mean_surface_height = sea_height

        # ====== 4. 水下切分 ======
        underwater_mask_main = ref_data[:,3] < (mean_surface_height - 0.5)
        underwater_points = ref_data[underwater_mask_main,1:4]
        underwater_mask_ch3 = ch3_data[:,3] < (mean_surface_height - 0.5)
        underwater_ch3_points = ch3_data[underwater_mask_ch3,1:4]
        flag = 'no_water'
        # ====== 5. 水下特征 & 预测 ======
        if len(underwater_points) > 0:
            flag = 'water'
            underwater_features, feature_names_underwater = extract_underwater_features(
                underwater_points,
                underwater_ch3_points,
                H_surface = mean_surface_height - 1,
                voxel_scales=(0.1,0.5,1.0),
                neigh_radius=1
            )
            dmat_underwater = xgb.DMatrix(underwater_features, feature_names=feature_names_underwater)
            underwater_preds_idx = underwater_model.predict(dmat_underwater).argmax(axis=1)
            label_map = {0:0, 1:1, 2:2, 3:4}
            final_preds[underwater_mask_main] = np.array([label_map[p] for p in underwater_preds_idx])

        # ====== 6. 保存到新 H5 文件 ======
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.basename(file_path)
        out_file = os.path.join(out_dir, f"{(idx+start_idx):03d}{'_'}{flag}{'_'}{base_name.replace('.h5','_pred.h5')}")

        with h5py.File(out_file, 'w') as hf:
            hf.create_dataset('ref_data', data=ref_data)
            hf.create_dataset('ch3_data', data=ch3_data)
            hf.create_dataset('pred_labels', data=final_preds)

        out_txt = os.path.join(out_dir, f"{(idx+start_idx):03d}{'_'}{flag}{'_'}{base_name.replace('.h5','_pred.txt')}")
        # 拼接前四列 + 预测值
        txt_data = np.hstack([ref_data[:, :4], final_preds.reshape(-1, 1)])
        np.savetxt(out_txt, txt_data, fmt="%.6f %.6f %.6f %.6f %d")
        print(f"保存 TXT 完成: {out_txt}")

        print(f"保存h5文件完成: {out_file}")
        # plot_tz_projection(ref_data[:,1], ref_data[:,3], ref_data[:,4], final_preds, title_prefix="All Points", max_points=30000)

if __name__ == "__main__":
    root_dir = r'E:/0716/f1/L2_XYZ_DATA'
    out_dir = r'E:/0716/f1/train_sig1_pred'
    import os
    os.makedirs(out_dir, exist_ok=True)
    predict_and_save(root_dir, out_dir, start_idx=120, end_idx=150)
