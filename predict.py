import h5py
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from get_features_seabed import extract_underwater_features
from sklearn.metrics import classification_report
from prepare_train_data import list_h5_files, load_data_from_h5, prepare_seasurface_land_data
from get_features_sea_land import extract_features_xyz
from eva import evaluate_model, plot_tz_projection

def main():
    # ====== 1. 加载模型 ======
    surface_model = xgb.Booster()
    surface_model.load_model('land_sea_classifier_tz.model')
    underwater_model = xgb.Booster()
    underwater_model.load_model('underwater_classifier3.model')

    root_dir = r'E:/DATA_0714/flight-02/train_sig3'
    start_idx = 5
    end_idx = 6
    files = list_h5_files(root_dir, start_idx, end_idx)

    for test_file in files:
        print(f"\n处理测试文件: {test_file}")
        ref_data, ch3_data = load_data_from_h5(test_file)
        true_labels = ref_data[:,4].astype(int)

        # ====== 2. 准备海面数据 ======
        sea_height_str = surface_model.attr('sea_height')
        sea_height = float(sea_height_str) if sea_height_str is not None else -10.0
        print("从模型读取 sea_height:", sea_height)

        points_sea, labels_sea = prepare_seasurface_land_data(ref_data, sea_height)
        labels_sea[labels_sea == 2] = 3  # 将标签2替换为3 代表陆地
        idx_sea = np.arange(ref_data.shape[0])[ref_data[:,3] > sea_height]  # 原始索引
        final_preds = np.zeros_like(true_labels)

        # ====== 3. 海面特征 & 预测 ======
        if len(points_sea) > 0:
            print("提取水上特征...")
            features_sea, feature_names_sea = extract_features_xyz(points_sea, voxel_size=(0.25,0.5,1), k=10, mean_surface_height=sea_height)
            dmat_sea = xgb.DMatrix(features_sea, feature_names=feature_names_sea)
            pred_labels_sea = surface_model.predict(dmat_sea).argmax(axis=1)

            # 映射回原始标签: 0->0, 1->1(海面), 2->3(陆地)
            sea_label_map = {0:0, 1:1, 2:3}
            pred_labels_sea_mapped = np.array([sea_label_map[p] for p in pred_labels_sea])
            final_preds[idx_sea] = pred_labels_sea_mapped

            # 平均海面高度 Z
            surface_points = points_sea[pred_labels_sea == 1]
            mean_surface_height = np.mean(surface_points[:,2]) if len(surface_points) > 0 else sea_height

            # ====== 海面类别检查 ======
            print("\n[检查] 海面部分类别")
            print("原始海面标签类别:", np.unique(labels_sea))
            print("预测海面标签类别:", np.unique(pred_labels_sea_mapped))

            # 海面分类报告
            unique_labels_sea = sorted(set(labels_sea) | set(pred_labels_sea_mapped))
            label_names_sea_eval = {0:'Other', 1:'Sea Surface', 3:'Land'}
            target_names_sea = [f"{label_names_sea_eval[l]} ({l})" for l in unique_labels_sea]
            print("\n=== 海面部分评估 ===")
            print(classification_report(labels_sea, pred_labels_sea_mapped,
                                        labels=unique_labels_sea,
                                        target_names=target_names_sea))

        if len(points_sea) > 0 and len(surface_points) > 1000: # type: ignore
            print("\n=== 海底点预测 ===")
            # ====== 4. 水下切分 ======
            # 主通道（水下点）
            underwater_mask_main = ref_data[:,3] < (mean_surface_height - 0.5) # type: ignore
            underwater_points = ref_data[underwater_mask_main,1:4]
            underwater_labels = true_labels[underwater_mask_main]

            # 辅助通道（水下点）
            underwater_mask_ch3 = ch3_data[:,3] < (mean_surface_height - 0.5) # type: ignore
            underwater_ch3_points = ch3_data[underwater_mask_ch3,1:4]

            # ====== 5. 水下特征 & 预测 ======
            underwater_preds = np.array([], dtype=int)
            if len(underwater_points) > 0:
                print("提取水下特征...")
                underwater_features, feature_names_underwater = extract_underwater_features(
                    underwater_points, 
                    underwater_ch3_points, 
                    H_surface = mean_surface_height + 1,  # 水下特征提取时的水面高度 # type: ignore
                    voxel_scales=(0.25,0.5,1.0),
                    neigh_radius=1,
                )
                
                dmat_underwater = xgb.DMatrix(underwater_features, feature_names=feature_names_underwater)
                underwater_preds_idx = underwater_model.predict(dmat_underwater).argmax(axis=1)
                label_map = {0:0, 1:1, 2:2, 3:4}  # 统一原始标签
                underwater_preds = np.array([label_map[p] for p in underwater_preds_idx])
                final_preds[underwater_mask_main] = underwater_preds

                # ====== 水下类别检查 ======
                print("\n[检查] 水下部分类别")
                print("原始水下标签类别:", np.unique(underwater_labels))
                print("预测水下标签类别:", np.unique(underwater_preds))

                # # 水下评估
                # reverse_map = {0:0,1:1,2:2,4:4}
                # evaluate_model(
                #     model=underwater_model,
                #     X_val=underwater_features,
                #     y_val=underwater_labels,
                #     label_names={0:'Other',1:'Sea Surface',2:'Seabed',4:'Water Body'},
                #     val_preds=underwater_preds,
                #     feature_names=feature_names_underwater,
                #     reverse_map=reverse_map,
                #     title_prefix="Underwater"
                # )

        # ====== 6. 全局评估 ======
        print("\n[检查] 全部点类别")
        print("原始全局标签类别:", np.unique(true_labels))
        print("预测全局标签类别:", np.unique(final_preds))

        label_names_all = {0:'Other',1:'Sea Surface',2:'Seabed',3:'Land',4:'Water Body'}
        unique_labels_all = sorted(set(true_labels) | set(final_preds))
        target_names_all = [f"{label_names_all[l]} ({l})" for l in unique_labels_all]
        print("\n=== 全部点分类报告 ===")
        print(classification_report(true_labels, final_preds,
                                    labels=unique_labels_all,
                                    target_names=target_names_all))

        # ====== 7. 全部点 X-Z 投影可视化 ======
        plot_tz_projection(ref_data[:,1], ref_data[:,3], true_labels, final_preds, title_prefix="All Points", max_points=30000)

if __name__ == "__main__":
    main()
