import numpy as np
import time
from numba import njit

# ---------------- Numba 加速函数 ----------------
@njit
def _points_to_bins_numba(xyz, edges0, edges1, edges2):
    N = xyz.shape[0]
    ix = np.empty(N, dtype=np.int32)
    iy = np.empty(N, dtype=np.int32)
    iz = np.empty(N, dtype=np.int32)
    for i in range(N):
        # x
        xi = 0
        while xi < len(edges0)-1 and xyz[i,0] >= edges0[xi+1]:
            xi += 1
        ix[i] = xi
        # y
        yi = 0
        while yi < len(edges1)-1 and xyz[i,1] >= edges1[yi+1]:
            yi += 1
        iy[i] = yi
        # z
        zi = 0
        while zi < len(edges2)-1 and xyz[i,2] >= edges2[zi+1]:
            zi += 1
        iz[i] = zi
    return ix, iy, iz

@njit
def _gather_from_grid_numba(grid, ix, iy, iz):
    N = ix.shape[0]
    vals = np.empty(N, dtype=np.float32)
    for i in range(N):
        vals[i] = grid[ix[i], iy[i], iz[i]]
    return vals

@njit
def _box_sum3d_numba(arr, r=1):
    X,Y,Z = arr.shape
    A = np.zeros((X+1,Y+1,Z+1), dtype=np.float32)
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                A[i+1,j+1,k+1] = arr[i,j,k] + A[i,j+1,k+1] + A[i+1,j,k+1] + A[i+1,j+1,k] \
                                 - A[i,j,k+1] - A[i,j+1,k] - A[i+1,j,k] + A[i,j,k]
    out = np.empty_like(arr)
    for i in range(X):
        x0 = max(i-r,0)
        x1 = min(i+r,X-1)
        for j in range(Y):
            y0 = max(j-r,0)
            y1 = min(j+r,Y-1)
            for k in range(Z):
                z0 = max(k-r,0)
                z1 = min(k+r,Z-1)
                out[i,j,k] = A[x1+1,y1+1,z1+1] - A[x0,y1+1,z1+1] - A[x1+1,y0,z1+1] - A[x1+1,y1+1,z0] \
                             + A[x0,y0,z1+1] + A[x0,y1+1,z0] + A[x1+1,y0,z0] - A[x0,y0,z0]
    return out

def _local_mean_var_from_counts_numba(counts,r=1):
    nbh_sum = _box_sum3d_numba(counts,r)
    nbh_sum2 = _box_sum3d_numba(counts**2,r)
    ones = np.ones_like(counts,dtype=np.float32)
    nbh_voxels = _box_sum3d_numba(ones,r)
    mean = nbh_sum / np.maximum(nbh_voxels,1)
    var = nbh_sum2 / np.maximum(nbh_voxels,1) - mean**2
    var = np.clip(var,0,None)
    return mean, var

@njit
def _depth_layer_norm_numba(values, depths, bin_edges):
    eps = 1e-6
    N = values.shape[0]
    nbins = len(bin_edges)-1
    b = np.empty(N,dtype=np.int32)
    for i in range(N):
        bi = 0
        while bi < nbins-1 and depths[i] >= bin_edges[bi+1]:
            bi += 1
        b[i] = bi
    layer_sum = np.zeros(nbins,dtype=np.float32)
    layer_cnt = np.zeros(nbins,dtype=np.int32)
    for i in range(N):
        layer_sum[b[i]] += values[i]
        layer_cnt[b[i]] += 1
    layer_mean = np.empty(nbins,dtype=np.float32)
    for i in range(nbins):
        if layer_cnt[i] > 0:
            layer_mean[i] = layer_sum[i]/layer_cnt[i]
        else:
            layer_mean[i] = eps
    out = np.empty(N,dtype=np.float32)
    for i in range(N):
        out[i] = values[i]/(layer_mean[b[i]]+eps)
    return out

# ---------------- 工具函数 ----------------
def _make_edges(min_v, max_v, step):
    return np.arange(min_v, max_v + step*1.0001, step)

def _hist3d_counts(xyz, edges):
    return np.histogramdd(xyz, bins=edges)[0].astype(np.float32)

# ---------------- 主特征提取 ----------------
def extract_underwater_features(points, ch3_data,
                                voxel_scales=(0.25,0.5,1.0),
                                neigh_radius=1,
                                depth_bins=np.arange(-40,-7,1)):
    t0 = time.time()
    if points.shape[0]==0:
        fns = []
        for s in voxel_scales:
            fns += [f'LocalVar_s{s}', f'Depol_s{s}', f'LocalMean_s{s}',
                    f'DensityNorm_s{s}', f'DepolNorm_s{s}']
        return np.empty((0,len(fns))), fns

    eps = 1e-6
    xyz = points[:,:3].astype(np.float32)
    ch3_xyz = ch3_data[:,1:4].astype(np.float32)
    depths = xyz[:,2]

    lo = np.minimum(xyz.min(axis=0), ch3_xyz.min(axis=0))
    hi = np.maximum(xyz.max(axis=0), ch3_xyz.max(axis=0))

    features_per_scale = []
    feature_names = []

    for s in voxel_scales:
        edges = [_make_edges(lo[d], hi[d], s) for d in range(3)]
        # 主通道 / ch3 的 3D 密度网格
        main_counts = _hist3d_counts(xyz, edges)
        ch3_counts  = _hist3d_counts(ch3_xyz, edges)

        # 邻域统计
        local_mean, local_var = _local_mean_var_from_counts_numba(main_counts, r=neigh_radius)

        # 体素索引一次性计算
        ix, iy, iz = _points_to_bins_numba(xyz, edges[0], edges[1], edges[2])

        # 取体素值
        center_density = _gather_from_grid_numba(main_counts, ix, iy, iz)
        depol = _gather_from_grid_numba(ch3_counts, ix, iy, iz)/(center_density+eps)
        loc_mean_pt = _gather_from_grid_numba(local_mean, ix, iy, iz)
        loc_var_pt  = _gather_from_grid_numba(local_var,  ix, iy, iz)

        # 深度分层归一化
        dens_norm  = _depth_layer_norm_numba(center_density, depths, depth_bins)
        depol_norm = _depth_layer_norm_numba(depol, depths, depth_bins)

        # 拼接该尺度特征
        feats_s = np.column_stack([loc_var_pt, depol, loc_mean_pt, dens_norm, depol_norm])
        features_per_scale.append(feats_s)

        feature_names += [f'LocalVar_s{s}', f'Depol_s{s}', f'LocalMean_s{s}',
                          f'DensityNorm_s{s}', f'DepolNorm_s{s}']

    features = np.hstack(features_per_scale).astype(np.float32)
    print(f"[Feature extraction] {points.shape[0]} points, {features.shape[1]} features, time={time.time()-t0:.3f}s")
    return features, feature_names
