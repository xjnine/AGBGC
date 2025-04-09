import itertools
import time
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import pdb
import itertools
import math
import warnings
from sklearn.neighbors import NearestNeighbors
from GBC_2.wkmeans_no_random import WKMeans
from sklearn.cluster import k_means
from GBC_2.lwkmeans import lwkmeans
warnings.filterwarnings('ignore')

def get_dm(hb, w):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center - hb
    w_mat = np.tile(w, (num, 1))
    sq_diff_mat = diff_mat ** 2 * w_mat
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sum_radius = sum(distances)
    if num > 1:
        return sum_radius / num
    else:
        return 1

def division(hb_list, hb_list_not, division_num, K, parent_w):
    gb_list_new = []
    i = 0
    K = 2
    for hb in hb_list:
        hb_no_w = hb
        # Caltech101-7>=8
        if len(hb_no_w) >= 2 :
            i = i + 1
            ball, child_w = spilt_ball_by_k(hb_no_w, parent_w, K, division_num) # k分裂球簇
            if all(len(b) > 1 for b in ball):
                dm_child_ball = []
                child_ball_length = []
                dm_child_divide_len = []
                for i in range(K):
                    temp_dm = get_dm(np.delete(ball[i], -1,axis = 1), parent_w)
                    temp_len = len(ball[i])
                    dm_child_ball.append(temp_dm)
                    child_ball_length.append(temp_len)
                    dm_child_divide_len.append(temp_dm * temp_len)
                w0 = np.array(child_ball_length).sum()
                dm_child = np.array(dm_child_divide_len).sum() / w0
                dm_parent = get_dm(np.delete(hb_no_w,-1,axis = 1), parent_w)
                t2 = (dm_child < dm_parent)
                if t2:
                    for i in range(K):
                        gb_list_new.append(ball[i])
                else:
                    hb_list_not.append(hb)
            else:
                hb_list_not.append(hb)
        else:
            hb_list_not.append(hb)
    return gb_list_new, hb_list_not

def KNN_ball_outer_2(ball_center_num, k, distances, ball_list_w):
    hb_graph = []
    hb_graph_w = []
    result_temp = []
    result = []
    for row in distances:
        sorted_indices = np.argsort(row)
        k_smallest_indices = sorted_indices[1:k]
        result_temp.append(k_smallest_indices)
    for re in result_temp:
        result.append([int(ball_center_num[i]) for i in re])
    for i in range(len(result)):
        perms = [(int(ball_center_num[i]), num) for num in result[i]]
        hb_graph.extend(perms)
    hb_graph = sorted(list(set(hb_graph)), key=lambda x: (x[0], x[1]))
    for i in range(len(hb_graph)):
        w = ball_list_w[0]
        hb_graph_w.extend([w])
    return hb_graph, hb_graph_w
def KNN_ball_outer(hb_list_number,ball_center_list, ball_weights, ball_data):
    """
    ball_center_list: 粒球中心样本的索引列表 (如 [array([139.]), array([128.])]
    ball_weights: 粒球权重列表
    ball_data: 所有粒球的数据 (如 [array([[0.3942722,...,0.5842721]]), ...])
    """
    hb_graph = []
    hb_graph_w = []
    min_k = 7# k_in的最小值
    max_k = 15
    ball_props = []
    sample_counts = [len(hb) for hb in hb_list_number]
    for i, data in enumerate(ball_data):
        features = data  # 每个粒球的特征
        center = np.mean(features, axis=0)  # 计算粒球的中心特征
        ball_props.append({
            "center": center,
            "samples": ball_center_list[i][0],
                # 使用粒球中心的索引
        })
    
    # ========== 计算粒球中心间距离 ==========
    centers = np.array([b["center"] for b in ball_props])
    center_dists = np.sqrt(np.sum((centers[:, None] - centers) ** 2, axis=2))
    
    # ========== 自适应k值计算 ==========
    avg_distances = np.mean(center_dists, axis=1)  # 每个粒球中心到其他中心的平均距离
    global_ddistances = np.median(avg_distances)  # 使用中位数更鲁棒
    k_out_values = []
    for i,dist in enumerate(avg_distances):
        k_base = int(dist / (global_ddistances+ 1e-8))
        size_factor = np.log1p(sample_counts[i])  # 粒球大小影响因子
        k_adj = int(k_base * size_factor * sample_counts[i])
        k_adj = min(max(k_adj, min_k),max_k) 
        k_out_values.append(k_adj)
    
    # ========== 构建粒球间连接 ==========
    for i, b in enumerate(ball_props):
        k_curr = k_out_values[i]
        if len(ball_props) <= k_curr:  # 全连接
            pairs = list(itertools.permutations([b["samples"]], 2))
            hb_graph.extend(pairs)
            hb_graph_w.extend([(ball_weights[i] + ball_weights[j]) / 2 
                              for j in range(len(ball_props))])
        else:  # KNN连接
            sorted_indices = np.argsort(center_dists[i])[1:k_curr + 1]  # 排除自身
            
            for j in sorted_indices:
                src_samples = b["samples"]  # 这里是粒球中心索引
                tgt_samples = ball_props[j]["samples"]  # 这里是粒球中心索引
                
                # 添加双向连接
                hb_graph.append((int(src_samples), int(tgt_samples)))
                hb_graph_w.append(
                    (ball_weights[i] + ball_weights[j]) / 2 * 
                    np.exp(-center_dists[i, j])  # 距离衰减因子
                )
    
    return hb_graph, hb_graph_w
def KNN_ball_inner(hb_list_number, K , ball_list_w):
    hb_graph = []
    hb_graph_w = []
    min_k = 7  # k_in的最小值
    max_k = 15
    # 预计算全局密度统计量
    density_list = []
    for b in hb_list_number:
        features = b[:, :-1]
        center = np.mean(features, axis=0)
        dists = np.linalg.norm(features - center, axis=1)
        density_list.append(1/(np.median(dists) + 1e-8))
    global_density = np.median(density_list)  # 使用中位数更鲁棒

    for idx, hb_ball in enumerate(hb_list_number):
        features = hb_ball[:, :-1]
        labels = hb_ball[:, -1].astype(int)
        n_samples = len(features)
        
        # 动态属性计算
        center = np.mean(features, axis=0)
        dists = np.linalg.norm(features - center, axis=1)
        local_density = 1/(np.median(dists) + 1e-8)
        
        # k_in计算公式
        density_ratio =  local_density/ (global_density  + 1e-8)
        size_factor = np.log1p(n_samples)  # 粒球大小影响因子
        k_in = int(density_ratio * size_factor * n_samples)
        k_in = min(max(k_in, min_k),max_k)  # 确保k_in不小于min_k
        
        # 优化距离计算
        diff = features[:, None] - features
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        
        # 构建邻接关系
        if n_samples <= k_in :  # 全连接+自连接排除
            mask = ~np.eye(n_samples, dtype=bool)
            rows, cols = np.where(mask)
            hb_graph.extend(zip(labels[rows], labels[cols]))
            hb_graph_w.extend([ball_list_w[idx]] * len(rows))
        else:
            # 获取k_in最近邻（排除自身）
            knn_indices = np.argpartition(dist_matrix, k_in, axis=1)[:, 1:k_in+1]
            
            # 构建连接
            for i in range(n_samples):
                for j in knn_indices[i]:
                    hb_graph.append((labels[i], labels[j]))
                    hb_graph_w.append(dist_matrix[i,j] * ball_list_w[idx])
    
    return hb_graph, hb_graph_w

def load_graph_w_natural_v2(hb_list,ball_list_w, K, origin_data,prunning_one,prunning_two, common_neighbors):
    start_time = time.time()
    graph_label = []
    hb_list_number = hb_list
    # 用于计算图中每个节点的K近邻关系，并返回两个值：邻接节点列表 hb_graph_1 和相应的权重 hb_graph_w1
    hb_graph_1, hb_graph_w1 = KNN_ball_inner(hb_list_number,K, ball_list_w)
    # 用于计算图中节点的中心及其相关信息
    ball_center_num, ball_center, ball_center_no_idx = get_center_and_num(hb_list_number, ball_list_w)
    # 外部球之间的邻居关系
    hb_graph_2, hb_graph_w2 = KNN_ball_outer(hb_list_number,ball_center_num, ball_list_w, ball_center_no_idx)
    # hb_graph_2, hb_graph_w2 = KNN_ball_outer(ball_center_num, K, max_num, ball_list_w, ball_center_no_idx)
    # 内部和外部邻居关系组合起来
    hb_graph = hb_graph_1 + hb_graph_2
    hb_graph_w = hb_graph_w1 + hb_graph_w2
    graph_data = origin_data
    # 有向图矩阵
    directed_graph_w_matrix = np.zeros((len(graph_data), len(graph_data)))
    for index, (i, j) in enumerate(hb_graph):
        w = hb_graph_w[index]
        dis = np.sum(np.linalg.norm(graph_data[i] - graph_data[j]) * w) # 默认二范数求距离
        directed_graph_w_matrix[i, j] = dis
    for i in range(len(directed_graph_w_matrix)):
        # 对于矩阵的对角线元素，更新为该节点所有连接权重的平均值。这是为了调整对角线的值，使得节点的自连接权重等于其所有连接权重的平均值
        directed_graph_w_matrix[i, i] = np.sum(directed_graph_w_matrix[i, :]) / len(directed_graph_w_matrix[i, :])
    # 创建零一图矩阵
    graph_w_dis_numpy = (directed_graph_w_matrix > 0).astype(int)

    # 确保对角线元素为 0
    np.fill_diagonal(graph_w_dis_numpy, 0)
    if prunning_one:
        # Pruning strategy 1
        adj_wave = graph_w_dis_numpy
        original_adj_wave = adj_wave
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)  #将结果转为稀疏矩阵
    else: 
        # transform the matrix to be symmetric (Instead of Pruning strategy 1)
        adj_wave = graph_w_dis_numpy
        np_adj_wave = construct_symmetric_matrix(adj_wave)
        adj_wave = sp.csc_matrix(np_adj_wave)

    # obtain the adjacency matrix without self-connection
    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.A
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()

    # construct the adjacency hat matrix
    adj_hat = construct_adjacency_hat(adj)
    print("The construction of adjacency matrix is finished!")
    print("The time cost of construction: ", time.time() - start_time)
    return adj, adj_wave, adj_hat,directed_graph_w_matrix, graph_data, graph_label

def get_center_and_num(hb_list_number, ball_list_w):
    ball_center_num = []
    ball_center = []
    ball_center_no_idx = []
    for i, hb_ball in enumerate(hb_list_number):
        center = np.mean(hb_ball, axis=0)
        min = 100000000000
        for data in hb_ball:   # 寻找最接近平均质心的点:
            dis = np.sum(np.power((np.delete(data, -1, axis=0) - np.delete(center, -1, axis=0)), 2) * ball_list_w[i])
            if dis < min:
                data = np.array([data])
                idx = data[:, -1]
                center1 = data
                min = dis
        ball_center_num.append(idx)
        ball_center.append(center1)
        ball_center_no_idx.append(np.delete(center1, -1, axis=1))
    return ball_center_num, ball_center, ball_center_no_idx

def get_distance_w_2(ball_center_num, ball_list_w, ball_center):
    distances = np.zeros((len(ball_center_num), len(ball_center_num)))
    for i in range(len(ball_center_num)):
        for j in range(len(ball_center_num)):
            w = ball_list_w[0]
            distances[i, j] = np.sum(
                np.linalg.norm(ball_center[i] - ball_center[j]) * w)
    return distances

def spilt_ball_by_k(data, w, k, division_num):
    centroids = []
    data_no_label = np.delete(data, -1, axis=1)
    k = 2
    center = data_no_label.mean(0)
    p_max1 = np.argmax(((data_no_label - center) ** 2).sum(axis=1) ** 0.5)
    p_max2 = np.argmax(((data_no_label - data_no_label[p_max1]) ** 2).sum(axis=1) ** 0.5)
    c1 = (data_no_label[p_max1] + center) / 2
    c2 = (data_no_label[p_max2] + center) / 2
    centroids.append(c1)
    centroids.append(c2)
    idx = np.ones(len(data_no_label))
    for i in range(len(data_no_label)):
        subs = centroids - data_no_label[i, :]
        w_dimension2 = np.power(subs, 2)
        w_distance2 = np.sum(w_dimension2, axis=1)
        if math.isnan(w_distance2.sum()) or math.isinf(w_distance2.sum()):
            w_distance2 = np.zeros(k)
        idx[i] = np.where(w_distance2 == w_distance2.min())[0][0]
    ball = []
    for i in range(k):
        ball.append(data[idx == i, :])
    return ball, w

def load_wkmeans_weight_and_ball(K, data):
    data_no_label = np.delete(data, -1, axis=1)
    centers = k_means(data_no_label, K, init="k-means++", n_init=10)[0]
    model = WKMeans(n_clusters=K, max_iter=10, belta=4, centers=centers)
    cluster = model.fit_predict(data_no_label)
    w = model.w
    ball = []
    for i in range(K):
        ball.append(data[cluster == i, :])
    return ball, w

def load_lwkmeans_weight_and_ball(K, data, m):
    if m <= 50:
        return load_wkmeans_weight_and_ball(K, data)   # 使用k-means++ 预测球簇
    data_no_label = np.delete(data, -1, axis=1)
    cluster, w, _, _ = lwkmeans(data_no_label, K, alpha=0.005)  # 加权K均值算法 算法返回簇和权重
    ball = []
    for i in range(K):
        ball.append(data[cluster == i, :])
    return ball, w


def construct_adjacency_hat(adj):
    """
    :param adj: original adjacency matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def construct_symmetric_matrix(original_matrix):
    """
        transform a matrix (n*n) to be symmetric
    :param np_matrix: <class 'numpy.ndarray'>
    :return: result_matrix: <class 'numpy.ndarray'>
    """
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
                pdb.set_trace()
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")
        pdb.set_trace()

    return result_matrix

# 执行某种形式的聚类或划分，最后生成一个图的加权邻接矩阵及相关数据    WGBS形成
def hbc_v2(data_all, y, K, origin_data,prunning_one,prunning_two, common_neighbors):
    n, m = data_all.shape
    hb_list_not_temp = []
    division_num = 1
    # 加载初始的球簇（ball）和权重（all_w）
    ball, all_w = load_lwkmeans_weight_and_ball(K, data_all, m)
    hb_list_temp = ball
    while 1:
        # 球簇分裂
        ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
        division_num = division_num + 1
        # not_temp是球簇未分裂的,temp就是球簇分裂出来的
        hb_list_temp, hb_list_not_temp = division(hb_list_temp, hb_list_not_temp, division_num, K, all_w)
        ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
        if ball_number_new == ball_number_old:  # 判断球簇数量是否增加，每增加结束循环
            hb_list_temp = hb_list_not_temp
            break
    ball_list = []
    # 构造邻接矩阵
    for index, hb_ball in enumerate(hb_list_temp):  # 新分裂的球簇组成的hb_list_temp簇
        if len(hb_ball) != 0:
            ball_list.append(hb_ball)
    hb_list_temp = ball_list
    dic_w = {}
    for index, hb_all in enumerate(hb_list_temp):
        dic_w[index] = all_w
    # 加载图数据和计算距离矩阵
    # 用于构建加权有向图的邻接矩阵，并返回图的相关信息
    adj, adj_wave, adj_hat,directed_graph_w_matrix, graph_data, graph_label = load_graph_w_natural_v2(hb_list_temp,dic_w, K, origin_data,prunning_one,prunning_two, common_neighbors)
    graph_w_dis_numpy = (directed_graph_w_matrix + directed_graph_w_matrix.T) / 2
    return adj, adj_wave, adj_hat, graph_w_dis_numpy, graph_data,graph_label, hb_list_temp, directed_graph_w_matrix