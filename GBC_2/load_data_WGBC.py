import numpy as np
# from GBC_2.W_GBC_create_graph import hbc_v2
from GBC_2.create_graphMVAK import hbc_v2

def load_data_WGBC_v2(data,y, k,prunning_one,prunning_two, common_neighbors):
    origin_data = data
    n, m = data.shape
    y = np.arange(n).reshape((n, 1))
    data = np.hstack((data, y))
    adj, adj_wave, adj_hat, graph_w_dis_numpy, graph_data,graph_label, hb_list_temp, directed_graph_w_matrix = hbc_v2(data , y, k, origin_data,prunning_one,prunning_two, common_neighbors)
    graph_label = y
    return adj, adj_wave, adj_hat,graph_data, graph_label, graph_w_dis_numpy, hb_list_temp, directed_graph_w_matrix


