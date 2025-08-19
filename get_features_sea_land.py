import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

def extract_features_xyz(points_xyz, voxel_size=(0.25,0.5,1), k=10, mean_surface_height=None):
    """
    高效版特征提取（使用 XYZ）：
    - Z
    - 相对海平面高度 DeltaZ
    - 中心密度 / 上下层密度比
    - 平坦度 (std of k neighbors)
    - 局部高度范围 (max-min)
    - KNN密度
    """
    import time
    from sklearn.neighbors import NearestNeighbors

    t0 = time.time()
    N = points_xyz.shape[0]

    X,Y,Z = points_xyz[:,0], points_xyz[:,1], points_xyz[:,2]

    # --- 体素化统计 ---
    voxel_t, voxel_z, voxel_xy = voxel_size
    ijk = np.floor((points_xyz - points_xyz.min(axis=0)) / np.array([voxel_xy, voxel_z, voxel_xy])).astype(int)
    max_ijk = ijk.max(axis=0) + 1
    voxel_count = np.zeros(max_ijk, dtype=np.int32)
    np.add.at(voxel_count, (ijk[:,0], ijk[:,1], ijk[:,2]), 1)
    center_density = voxel_count[ijk[:,0], ijk[:,1], ijk[:,2]]

    # --- 上下层密度比 (仅 Z方向) ---
    up_j = np.clip(ijk[:,1]+1, 0, max_ijk[1]-1)
    down_j = np.clip(ijk[:,1]-1, 0, max_ijk[1]-1)
    up_count = voxel_count[ijk[:,0], up_j, ijk[:,2]]
    down_count = voxel_count[ijk[:,0], down_j, ijk[:,2]]
    ratio_center_up = center_density / np.maximum(1, up_count)
    ratio_center_down = center_density / np.maximum(1, down_count)

    # --- KNN ---
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points_xyz)
    dists, idxs = nn.kneighbors(points_xyz)
    z_neighbors = Z[idxs]
    flatness = z_neighbors.std(axis=1)
    height_range = np.ptp(z_neighbors, axis=1)
    knn_density = 1.0 / (dists[:,k] + 1e-8)

    # --- DeltaZ 相对海面高度 ---
    DeltaZ = Z - (mean_surface_height if mean_surface_height is not None else Z.mean())

    features = np.column_stack([Z,DeltaZ,center_density,ratio_center_up,flatness,height_range,knn_density])
    feature_names = ['Z','DeltaZ','Center_Density','Ratio_Center_Up',
                     'Flatness','Height_Range','KNN_Density']

    print(f"特征提取完成，用时 {time.time()-t0:.2f}s, 样本数 {N}")
    return features, feature_names

