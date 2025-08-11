import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# ORGPathName 需要在运行前定义，例如：
# ORGPathName = r'E:\01 02 2025\数据处理流程\TEST_3q\'
ORGPathName = r'E:\DATA_0714\flight-02'  # 修改为实际路径
PathName = os.path.join(ORGPathName, 'L2_XYZ_DATA')
FileList = [f for f in os.listdir(PathName) if f.endswith('.h5')]

POINT_X_ALL = []
POINT_Y_ALL = []
POINT_Z_ALL = []
FileIndex_ALL = []
POINT_TIME_ALL = []

for FileIndex in range(10, 31):
    NameString = FileList[FileIndex]
    print(f'处理文件名：{NameString}')
    FileName = os.path.join(PathName, NameString)
    with h5py.File(FileName, 'r') as f:
        if len(f.keys()) == 16 or len(f) == 16:  # 兼容不同h5py版本
            print('OK')
            POINT_TIME = f['/GNSS_SEC_CH1'][:]
            POINT_X = f['/POINT_X_CH1'][:]
            POINT_Y = f['/POINT_Y_CH1'][:]
            POINT_Z = f['/POINT_Z_CH1'][:]
            POINT_X_ALL.append(POINT_X)
            POINT_Y_ALL.append(POINT_Y)
            POINT_Z_ALL.append(POINT_Z)
            POINT_TIME_ALL.append(POINT_TIME)
            FileIndex_ALL.append(np.full_like(POINT_Z, FileIndex))

# 合并所有数据
POINT_X_ALL = np.concatenate(POINT_X_ALL)
POINT_Y_ALL = np.concatenate(POINT_Y_ALL)
POINT_Z_ALL = np.concatenate(POINT_Z_ALL)
POINT_TIME_ALL = np.concatenate(POINT_TIME_ALL)
FileIndex_ALL = np.concatenate(FileIndex_ALL)

ind = np.where((POINT_Z_ALL >= -100) & (POINT_Z_ALL <= 200))[0]
POINT_X1 = POINT_X_ALL[ind]
POINT_Y1 = POINT_Y_ALL[ind]
POINT_Z1 = POINT_Z_ALL[ind]
POINT_TIME1 = POINT_TIME_ALL[ind]

D = 10
depth_xyz = np.column_stack((POINT_X1[::D], POINT_Y1[::D], POINT_Z1[::D]))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(depth_xyz)
o3d.visualization.draw_geometries([pcd], window_name="pcshow-like point cloud")