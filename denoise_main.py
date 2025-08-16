import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import open3d as o3d

# ========== 参数区 ==========
ORGPathName = r'E:\DATA_0714\flight-02'  # TODO: 修改为实际路径
PathName = os.path.join(ORGPathName, 'L2_XYZ_DATA')
dstPathName = os.path.join(ORGPathName, 'train_sig2')
os.makedirs(dstPathName, exist_ok=True)

num = 10000
channel_num = 3
f = 500000
min_points_threshold = 500
Sea_surface_KNN_thelshold = 0.005
Sea_surface_KNN_thelshold2 = 0.1
land_sea_flag_height = 10
ocean_height_up = 2
Z_max = 100
Z_RES = 5
KNN_notseabed_thelshold = 0.1
RATIO_channel_select_thelshold = [40, 15, 5]
PRE_bottom_echo = [0, 1, 1, 4]
channel_select = [[0,1],[0,1],[1],[1]]  # Python索引从0开始
width = 0.5
D_sparse = 20
echo_p = 150
width_seabed = [5,9]
echo_p_land = 300

# ========== 关键函数区 ==========
def surface_find(LIDAR_TIME, LIDAR_Z, Thelshold, f, num, land_sea_flag_height, Z_max):
    Xedges = np.arange(LIDAR_TIME.min(), LIDAR_TIME.max() + 1/f*num, 1/f*num)
    Zedges = np.arange(LIDAR_Z.min(), Z_max + 1, 1)
    counts, _, _ = np.histogram2d(LIDAR_TIME, LIDAR_Z, bins=[Xedges, Zedges])
    surface_depth = np.full(counts.shape[0], np.nan)
    surface_echo = np.full(counts.shape[0], np.nan)
    for i in range(counts.shape[0]):
        idxs = np.where(counts[i, :] > Thelshold)[0]
        if idxs.size > 0:
            idx = idxs[-1]
            surface_depth[i] = Zedges[idx]
            surface_echo[i] = counts[i, idx] / (1/f*num*f) * 100
    mask = np.isnan(surface_depth)
    if np.any(mask):
        surface_depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), surface_depth[~mask])
    for i in range(1, len(surface_depth)):
        if np.isnan(surface_depth[i]):
            surface_depth[i] = surface_depth[i-1]
    surface_time = Xedges[:-1]
    depth_variation = np.abs(np.nanmin(surface_depth) - np.nanmax(surface_depth))
    flag = 1 if depth_variation > land_sea_flag_height else 0
    plt.figure()
    plt.scatter(LIDAR_TIME, LIDAR_Z, s=1)
    plt.plot(surface_time, surface_depth, color='r')
    plt.xlabel('LIDAR TIME')
    plt.ylabel('LIDAR DEPTH')
    plt.title('2D Histogram and Surface')
    plt.show()
    return surface_time, surface_depth, surface_echo, flag

def plot_ratio_dep(channel_num, surface_time, surface_echo, surface_height, GNSS_SEC, POINT_Z):
    plt.figure()
    for i in [2, 0, 1]:
        plt.bar(surface_time, surface_echo[i], label=f'Channel {i+1}', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Echo count (%)')
    plt.legend()
    plt.title('Surface Echo per Channel')
    D = 1
    for i in range(channel_num):
        plt.figure()
        plt.scatter(GNSS_SEC[i][::D], POINT_Z[i][::D], s=1, label=f'Channel {i+1}')
        plt.plot(surface_time, surface_height, linewidth=2, label='Surface')
        plt.xlabel('Time (s)')
        plt.ylabel('Elevation (m)')
        plt.legend()
        plt.title(f'Channel {i+1} Elevation Profile')
    plt.figure()
    RATIO = surface_echo[2] / surface_echo[0]
    plt.bar(surface_time, RATIO)
    plt.xlabel('Time (s)')
    plt.ylabel('Depolarization Ratio')
    plt.title('Depolarization Ratio (Channel 3 / Channel 1)')
    valid = (~np.isinf(RATIO)) & (RATIO > 1)
    mean_ratio = np.mean(RATIO[valid]) if np.any(valid) else np.nan
    print(f'表面下1米区域的退偏比 R_dep = {mean_ratio:.3f}')
    #plt.show()
    return RATIO, mean_ratio

def surface_echo_caculate(LIDAR_TIME, LIDAR_Z, surface_height, f, num, Z_RES, Z_max):
    Xedges = np.arange(LIDAR_TIME.min(), LIDAR_TIME.max() + 1/f*num, 1/f*num)
    Yedges = np.arange(LIDAR_Z.min(), Z_max + Z_RES, Z_RES)
    counts, _, _ = np.histogram2d(LIDAR_TIME, LIDAR_Z, bins=[Xedges, Yedges])
    surface_echo = np.full(counts.shape[0], np.nan)
    for i in range(len(surface_height) - 1):
        idx = np.argmin(np.abs(Yedges - surface_height[i]))
        if 10 <= idx < counts.shape[1]:
            surface_echo[i] = np.sum(counts[i, idx-10:idx]) / (1/f*num*f) * 100
    return surface_echo

def knn_and_zhist_filter1(T, X, Y, Z, SNR_Seabed, knn_K, mean_height, height_range, flag):
    if flag == 1:
        mask_z = Z > -30
    elif flag == 2:
        mask_z = (Z < (mean_height + 2.2)) & (Z > (mean_height - height_range[1]))
    else:
        mask_z = Z <= mean_height
    T_sub = T[mask_z]
    X_sub = X[mask_z]
    Y_sub = Y[mask_z]
    Z_sub = Z[mask_z]
    idx_sub = np.where(mask_z)[0]
    if X_sub.size < knn_K:
        return np.array([], dtype=int)
    PointCloud_sub = np.column_stack((X_sub, Y_sub, Z_sub))
    tree = KDTree(PointCloud_sub)
    d_knn = tree.query(PointCloud_sub, k=knn_K)[0][:, -1]
    selected_sub_idx = []
    if flag == 2:
        Z_RES = 0.1
        z_min = Z_sub.min()
        z_max = Z_sub.max()
        for z0 in np.arange(z_min, z_min+2, Z_RES):
            bin_mask = (Z_sub >= z0) & (Z_sub < z0 + Z_RES)
            bin_idx = np.where(bin_mask)[0]
            if bin_idx.size == 0:
                continue
            p = 0.05
            d_knn_bin = d_knn[bin_idx]
            sort_idx = np.argsort(d_knn_bin)
            num_keep = max(1, int(p * SNR_Seabed / 100 * bin_idx.size))
            selected_bin_idx = bin_idx[sort_idx[:num_keep]]
            selected_sub_idx.extend(selected_bin_idx.tolist())
        for z0 in np.arange(z_min+2, z_max+0.2, 0.1):
            bin_mask = (Z_sub >= z0) & (Z_sub < z0 + 0.1)
            bin_idx = np.where(bin_mask)[0]
            if bin_idx.size == 0:
                continue
            p = 0.35
            d_knn_bin = d_knn[bin_idx]
            sort_idx = np.argsort(d_knn_bin)
            num_keep = max(1, int(p * SNR_Seabed / 100 * bin_idx.size))
            selected_bin_idx = bin_idx[sort_idx[:num_keep]]
            selected_sub_idx.extend(selected_bin_idx.tolist())
    else:
        Z_RES = 5
        for z0 in np.arange(-19, Z_sub.max()-4, Z_RES):
            bin_mask = (Z_sub >= z0) & (Z_sub < z0 + Z_RES)
            bin_idx = np.where(bin_mask)[0]
            if bin_idx.size == 0:
                continue
            p = 0.5
            d_knn_bin = d_knn[bin_idx]
            sort_idx = np.argsort(d_knn_bin)
            num_keep = max(1, int(p * SNR_Seabed / 100 * bin_idx.size))
            selected_bin_idx = bin_idx[sort_idx[:num_keep]]
            selected_sub_idx.extend(selected_bin_idx.tolist())
    selected_idx = idx_sub[selected_sub_idx]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_sub, Y_sub, Z_sub, s=1)
    ax.scatter(X[selected_idx], Y[selected_idx], Z[selected_idx], s=2, c='yellow')
    plt.show()
    return selected_idx

# ========== 主流程 ==========
if __name__ == "__main__":
    FileList = [f for f in os.listdir(PathName) if f.endswith('.h5')]

    for FileIndex in [59]:  # 可批量处理
        NameString = FileList[FileIndex]
        print(NameString)
        FileName = os.path.join(PathName, NameString)
        dstNameString = NameString.replace('L2_', 'L3_')
        dstFileName = os.path.join(dstPathName, dstNameString)

        if not os.path.exists(dstFileName):
            POINT_X = []
            POINT_Y = []
            POINT_Z = []
            GNSS_SEC = []
            with h5py.File(FileName, 'r') as h5f:
                for ch in range(1, channel_num+1):
                    POINT_X.append(h5f[f'/POINT_X_CH{ch}'][:])
                    POINT_Y.append(h5f[f'/POINT_Y_CH{ch}'][:])
                    POINT_Z.append(h5f[f'/POINT_Z_CH{ch}'][:])
                    GNSS_SEC.append(h5f[f'/GNSS_SEC_CH{ch}'][:])

            # 可视化（open3d 替代 pcshow）
            for idx in [0, 2]:
                pts = np.column_stack((POINT_X[idx], POINT_Y[idx], POINT_Z[idx]))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                o3d.visualization.draw_geometries([pcd], window_name=f"Channel {idx+1}")

            # 1. 用强通道找到表面
            strong_p_channel = 2  # Python索引
            surface_time, surface_height, surface_echo, flag = surface_find(
                GNSS_SEC[strong_p_channel], POINT_Z[strong_p_channel], min_points_threshold, f, num, land_sea_flag_height, Z_max)

            # 2. 计算表面退偏比
            surface_echo_list = []
            for i in range(channel_num):
                surface_echo_list.append(surface_echo_caculate(GNSS_SEC[i], POINT_Z[i], surface_height, f, num, Z_RES, Z_max))
            RATIO, mean_ratio = plot_ratio_dep(channel_num, surface_time, surface_echo_list, surface_height, GNSS_SEC, POINT_Z)

            # ...后续点云筛选、标签、保存等处理，参考前述迁移方案...
            # 例如 knn_and_zhist_filter1、标签赋值、h5写入等

            # 保存部分数据
            with h5py.File(dstFileName, 'a') as h5f:
                if '/ORG_DATA_channel_1' in h5f:
                    del h5f['/ORG_DATA_channel_1']
                h5f.create_dataset('/ORG_DATA_channel_1', data=np.column_stack((GNSS_SEC[0], POINT_X[0], POINT_Y[0], POINT_Z[0])))
                if '/ORG_DATA_channel_2' in h5f:
                    del h5f['/ORG_DATA_channel_2']
                h5f.create_dataset('/ORG_DATA_channel_2', data=np.column_stack((GNSS_SEC[1], POINT_X[1], POINT_Y[1], POINT_Z[1])))
                if '/ORG_DATA_channel_3' in h5f:
                    del h5f['/ORG_DATA_channel_3']
                h5f.create_dataset('/ORG_DATA_channel_3', data=np.column_stack((GNSS_SEC[2], POINT_X[2], POINT_Y[2], POINT_Z[2])))
                
        else:
            print('skip')
            continue