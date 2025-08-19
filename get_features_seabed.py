import numpy as np
from numba import njit
from sklearn.neighbors import KDTree
import time

# ---------------- Numba 工具 ----------------
from numba import njit
import numpy as np

@njit
def _points_to_bins_numba(xyz, edges0, edges1, edges2):
    """
    将点映射到体素索引，自动防越界
    """
    N = xyz.shape[0]
    ix = np.empty(N, dtype=np.int32)
    iy = np.empty(N, dtype=np.int32)
    iz = np.empty(N, dtype=np.int32)
    
    max_x = len(edges0) - 2
    max_y = len(edges1) - 2
    max_z = len(edges2) - 2
    
    for i in range(N):
        xi = 0
        while xi < max_x and xyz[i,0] >= edges0[xi+1]:
            xi += 1
        ix[i] = xi
        
        yi = 0
        while yi < max_y and xyz[i,1] >= edges1[yi+1]:
            yi += 1
        iy[i] = yi
        
        zi = 0
        while zi < max_z and xyz[i,2] >= edges2[zi+1]:
            zi += 1
        iz[i] = zi
    return ix, iy, iz

@njit
def _gather_from_grid_numba(grid, ix, iy, iz):
    """
    从三维网格提取值，自动防越界
    """
    N = ix.shape[0]
    vals = np.zeros(N, dtype=np.float32)
    
    if grid.size == 0:
        # 空网格，返回全0
        return vals
    
    X, Y, Z = grid.shape
    for i in range(N):
        xi = min(max(ix[i],0), X-1)
        yi = min(max(iy[i],0), Y-1)
        zi = min(max(iz[i],0), Z-1)
        vals[i] = grid[xi, yi, zi]
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

def _make_edges(min_v, max_v, step):
    return np.arange(min_v, max_v + step*1.0001, step)

def _hist3d_counts(xyz, edges):
    if xyz.shape[0]==0:
        return np.zeros((len(edges[0])-1,len(edges[1])-1,len(edges[2])-1), dtype=np.float32)
    return np.histogramdd(xyz, bins=edges)[0].astype(np.float32)

# ---------------- 水下特征提取 ----------------
def extract_underwater_features(points, ch3_data,
                                H_surface,
                                voxel_scales,
                                neigh_radius=1,
                                knn_k=10):
    """
    多尺度水下特征提取：
    - 小/中尺度: DensityWeighted, Density, KNNDensity weighted & unweighted
    - 最大尺度: LocalVar, Depol
    """
    t0 = time.time()
    N = points.shape[0]
    if N==0:
        fns = []
        for s in voxel_scales[:-1]:  # 小/中尺度
            fns += [f'DensityWeighted_s{s}', f'Density_s{s}',
                    f'KNNDensity_s{s}', f'KNNDensity_unweighted_s{s}']
        # 最大尺度特征
        fns += [f'LocalVar_s{voxel_scales[-1]}', f'Depol_s{voxel_scales[-1]}']
        return np.empty((0,len(fns)), dtype=np.float32), fns

    eps = 1e-6
    xyz = np.ascontiguousarray(points[:,:3], dtype=np.float32)
    ch3_xyz = np.ascontiguousarray(ch3_data[:,:3], dtype=np.float32)
    Z = xyz[:,2]

    lo = np.minimum(xyz.min(axis=0), ch3_xyz.min(axis=0))
    hi = np.maximum(xyz.max(axis=0), ch3_xyz.max(axis=0))

    features_per_scale = []
    feature_names = []

    depth_weight = np.maximum((H_surface - Z)**2, 1.0).astype(np.float32)
    tree = KDTree(xyz) if N>1 else None

    # 小/中尺度处理
    for s in voxel_scales[:-1]:
        if tree is not None:
            dists, _ = tree.query(xyz, k=min(knn_k+1,N))
            knn_dist = dists[:,-1] * s
            knn_density_weighted = depth_weight / (knn_dist + eps)
            knn_density_unweighted = 1.0 / (knn_dist + eps)
        else:
            knn_density_weighted = depth_weight.copy()
            knn_density_unweighted = np.ones_like(depth_weight)

        edges = [np.array(_make_edges(lo[d], hi[d], s), dtype=np.float32) for d in range(3)]
        main_counts = _hist3d_counts(xyz, edges)
        ix, iy, iz = _points_to_bins_numba(xyz, edges[0], edges[1], edges[2])
        center_density = _gather_from_grid_numba(main_counts, ix, iy, iz)

        density_weighted = center_density * depth_weight
        density_unweighted = center_density

        feats_s = [density_weighted, density_unweighted,
                   knn_density_weighted, knn_density_unweighted]
        features_per_scale.append(np.column_stack(feats_s))
        feature_names += [f'DensityWeighted_s{s}', f'Density_s{s}',
                          f'KNNDensity_s{s}', f'KNNDensity_unweighted_s{s}']

    # 最大尺度 LocalVar / Depol
    max_s = voxel_scales[-1]
    edges_max = [np.array(_make_edges(lo[d], hi[d], max_s), dtype=np.float32) for d in range(3)]
    counts_max = _hist3d_counts(xyz, edges_max)
    ch3_counts_max = _hist3d_counts(ch3_xyz, edges_max)
    local_mean, local_var = _local_mean_var_from_counts_numba(counts_max, r=neigh_radius)
    ix_max, iy_max, iz_max = _points_to_bins_numba(xyz, edges_max[0], edges_max[1], edges_max[2])
    loc_var_pt = _gather_from_grid_numba(local_var, ix_max, iy_max, iz_max)
    depol_pt = _gather_from_grid_numba(ch3_counts_max, ix_max, iy_max, iz_max) / (_gather_from_grid_numba(counts_max, ix_max, iy_max, iz_max)+eps)

    features_per_scale.append(np.column_stack([loc_var_pt, depol_pt]))
    feature_names += [f'LocalVar_s{max_s}', f'Depol_s{max_s}']

    features = np.hstack(features_per_scale).astype(np.float32)
    print(f"[Feature extraction] {N} points, {features.shape[1]} features, time={time.time()-t0:.3f}s")
    return features, feature_names




