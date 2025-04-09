# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:35:51 2022

@author: xiejiang
"""
from matplotlib.ticker import FixedLocator
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



class GB:
    def __init__(self, data, label):
        self.data = data
        # self.center = self.data[:, :-1].mean(0)
        self.center = self.data.mean(0)

        self.radius = self.get_radius()
        self.label = label
        self.num = len(data)

    def get_radius(self):
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)
    # def get_radius(self):
    #     return max(((self.data[:, :-1] - self.center) ** 2).sum(axis=1) ** 0.5)


class UF:
    def __init__(self, len):
        self.parent = [0] * len
        self.size = [0] * len
        self.count = len

        for i in range(0, len):
            self.parent[i] = i
            self.size[i] = 1

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        if self.size[rootP] > self.size[rootQ]:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        self.count = self.count - 1

    def connected(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ

    def count(self):
        return self.count


# 粒球划分
def division(hb_list, n):
    gb_list_new = []
    for hb in hb_list:
        if len(hb) >= 8:
            ball_1, ball_2 = spilt_ball(hb)  # 粒球分裂
            DM_parent = get_DM(hb)
            DM_child_1 = get_DM(ball_1)
            DM_child_2 = get_DM(ball_2)
            t1 = ((DM_child_1 > DM_parent) & (DM_child_2 > DM_parent))
            if t1:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(hb)
        else:
            gb_list_new.append(hb)

    return gb_list_new

# 根据最远的两个点划分粒球
def spilt_ball_2(data,data_index):
    ball1 = []
    ball2 = []
    index1 = []
    index2 = []
    # n行数、 m列数
    n, m = data.shape
    X = data.T
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))
    D = np.sqrt(np.abs(H + H.T - G * 2))
    # D中最大的元素所在的行r 列c
    r, c = np.where(D == np.max(D))
    r1 = r[1]
    c1 = c[1]
    # 遍历球内的每个样本
    for j in range(0, len(data)):
        if D[j, r1] < D[j, c1]:
            ball1.extend([data[j, :]])
            index1.append(data_index[j])
        else:
            ball2.extend([data[j, :]])
            index2.append(data_index[j])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2,index1,index2]

# 选择中心最远的点划分粒球
def spilt_ball_3(data,data_index):
    ball1 = []
    ball2 = []
    index1 = []
    index2 = []
    # n行数、 m列数
    n, m = data.shape
    # X = data.T
    # G = np.dot(X.T, X)
    # H = np.tile(np.diag(G), (n, 1))
    # D = np.sqrt(np.abs(H + H.T - G * 2))
    center = data.mean(0)
    p_max1 = np.argmax(((data - center) ** 2).sum(axis=1) ** 0.5)
    p_max2 = np.argmax(((data - data[p_max1]) ** 2).sum(axis=1) ** 0.5)
    c1 = (data[p_max1] + center) / 2
    c2 = (data[p_max2] + center) / 2

    # 遍历球内的每个样本
    for j in range(0, len(data)):
        D1=((data[j] - c1) ** 2).sum(axis=0) ** 0.5
        D2=((data[j] - c2) ** 2).sum(axis=0) ** 0.5

        if D2 > D1:
            ball1.extend([data[j, :]])
            index1.append(data_index[j])
        else:
            ball2.extend([data[j, :]])
            index2.append(data_index[j])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2,index1,index2]


# 获取球内的密度
def get_density_volume(gb):
    num = len(gb)

    # 计算gb中所有点的均值
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    # 每个点到中心点的距离
    distances = sqDistances ** 0.5
    sum_radius = 0
    if len(distances) == 0:
        print("0")
    # radius = max(distances)
    for i in distances:
        sum_radius = sum_radius + i
    # 平均距离
    mean_radius = sum_radius / num
    dimension = len(gb[0])
    # print('*******dimension********',dimension)
    if mean_radius != 0:
        # density_volume = num/(radius**dimension)
        # density_volume = num/((radius**dimension)*sum_radius)
        density_volume = num / sum_radius
        # density_volume = num/(sum_radius)
    else:
        density_volume = num

    return density_volume


# 无参遍历粒球是否需要分裂，根据子球和父球的比较，不带断裂判断的分裂,1分2
def division_2_2(gb_list,gb_data_index):
    gb_list_new = []
    gb_list_index_new=[]

    for i,gb_data in enumerate (gb_list):
        # 粒球内样本数大于等于8的粒球进行处理
        if len(gb_data) >= 8:
            ball_1, ball_2,index_1,index_2 = spilt_ball_3(gb_data,gb_data_index[i])  # 无模糊
            # ball_1, ball_2 = spilt_ball_fuzzy(gb_data)  # 有模糊
            # # ?
            # if len(ball_1) * len(ball_2) == 0:
            #     return gb_list
            # # ?
            # (zt)6.9
            # 如果划分的两个球中 其中一个球的内的样本数小于等于1 则该球不该划分
            if len(ball_1) == 1 or len(ball_2) == 1:
                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])
                continue
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])
                continue
            # if len(ball_1)*len(ball_2) == 0:
            #     gb_list_new.append(gb_data)
            #     continue
            # print("p")
            # print(len(gb_data[:, :-1]))
            # print("b1")
            # print(len(ball_1[:, :-1]))
            # print("b2")
            # print(len(ball_2[:, :-1]))
            parent_dm = get_density_volume(gb_data[:, :])
            child_1_dm = get_density_volume(ball_1[:, :])
            child_2_dm = get_density_volume(ball_2[:, :])
            w1 = len(ball_1) / (len(ball_1) + len(ball_2))
            w2 = len(ball_2) / (len(ball_1) + len(ball_2))
            # sep = get_separation(ball_1,ball_2)
            # print('this is sep test',sep)
            # t = (w1*density_child_1+w2*density_child_2)- 1/(w1*w2)*(w/n)
            w_child_dm = (w1 * child_1_dm + w2 * child_2_dm)  # 加权子粒球DM
            # if w > 20:
            # print("_______________________")
            # print("父亲数量", len(gb_data))
            # print("子球1数量", len(ball_1))
            # print("子球2数量", len(ball_2))
            # print("父球质量", parent_dm)
            # print("子球1质量", child_1_dm)
            # print("子球2质量", child_2_dm)
            # print("子球加权质量", w_child_dm)
            t1 = ((child_1_dm > parent_dm) & (child_2_dm > parent_dm))
            t2 = (w_child_dm > parent_dm)  # 加权DM上升
            t3 = ((len(ball_1) > 0) & (len(ball_2) > 0))  # 球中数据个数低于4个的情况不能分裂
            if t2:
                gb_list_new.extend([ball_1, ball_2])
                gb_list_index_new.extend([index_1,index_2])
            else:
                # print("父亲数量",w)
                # print("子球1数量",len(ball_1))
                # print("子球2数量",len(ball_2))
                # print("父球质量",density_parent)
                # print("子球1质量",density_child_1)
                # print("子球2质量",density_child_2)
                # print("分裂后效果不好")
                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])

        else:
            gb_list_new.append(gb_data)
            gb_list_index_new.append(gb_data_index[i])



    return gb_list_new,gb_list_index_new


# 粒球分裂
def spilt_ball(data,data_index):
    ball1 = []
    ball2 = []
    index1=[]
    index2=[]
    n, m = data.shape
    X = data.T
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))
    D = np.sqrt(np.abs(H + H.T - G * 2))  # 计算欧式距离
    r, c = np.where(D == np.max(D))  # 查找距离最远的两个点坐标
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):  # 对于距离最远的两个点之外的点，根据到两点距离不同划分到不同的簇中
        if D[j, r1] < D[j, c1]:
            ball1.extend([data[j, :]])
            index1.append(data_index[j])
        else:
            ball2.extend([data[j, :]])
            index2.append(data_index[j])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2,index1,index2]


# fuzzy粒球分裂
def spilt_ball_fuzzy(data):
    cluster = FCM_no_random(data[:, :-1], 2)
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]


def get_DM(hb):
    num = len(hb)
    center = hb.mean(0)
    diffMat = np.tile(center, (num, 1)) - hb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5  # 欧式距离
    sum_radius = 0
    radius = max(distances)
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius / num
    dimension = len(hb[0])
    if mean_radius != 0:
        DM = num / sum_radius
    else:
        DM = num
    return DM


def get_radius(gb_data):
    # 通过计算每个样本点与中心点之间的距离，并取最大值作为半径。
    # origin get_radius 7*O(n)
    sample_num = len(gb_data)
    center = gb_data.mean(0)
    diffMat = np.tile(center, (sample_num, 1)) - gb_data
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)

    # (zt)new get_radius *O(nd)
    # center = gb_data.mean(0)
    # radius = 0
    # for data in gb_data:
    #     temp = 0
    #     index = 0
    #     while index != len(data):
    #         temp += (data[index] - center[index]) ** 2
    #         index += 1
    #     radius_temp = temp ** 0.5
    #     if radius_temp > radius:
    #         radius = radius_temp
    return radius


def plot_dot(data):
    # plt.axes([0.4, 0.3, 0.3, 0.3])
    fig = plt.subplot(121)
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=5, alpha=0.6, marker='o', label='data point')
    try:
        fig.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=5, alpha=0.6, marker='o', label='data point')
    except:
        print()
    # plt.legend()
    fig.legend()
    return fig.findobj()


def gb_plot_test2(mb_list):
    color = {
        0: '#707afa',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',

        25: '#ff8444',
        26: '#a6dceb',
        27: '#fdd3a2',
        28: '#e6b1c2',
        29: '#9bb7d4',
        30: '#fedb5c',
        31: '#b2e1e0',
        32: '#f8c0b6',
        33: '#c8bfe7',
        34: '#f4af81',
        35: '#a3a3a3',
        36: '#bce784',
        37: '#8d6e63',
        38: '#e9e3c9',
        39: '#f5e9b2',
        40: '#ffba49',
        41: '#c0c0c0',
        42: '#d3a7b5',
        43: '#f2c2e0',
        44: '#b7dd29',
        45: '#dcf7c1',
    }
    label_c = {
        0: 'cluster-1',
        1: 'cluster-2',
        2: 'cluster-3',
        3: 'cluster-4',
        4: 'cluster-5',
        5: 'cluster-6',
        6: 'cluster-7',
        7: 'cluster-8',
        8: 'cluster-9',
        9: 'cluster-10',
        10: 'cluster-11',
        11: 'cluster-12',
        12: 'cluster-13',
        13: 'cluster-14',
        14: 'cluster-15',
        15: 'cluster-16',
        16: 'cluster-17',
        17: 'cluster-18',
        18: 'cluster-19',
        19: 'cluster-20',
        20: 'cluster-21',
        21: 'cluster-22',
        22: 'cluster-23',
        23: 'cluster-24',
        24: 'cluster-25',
        25: 'cluster-26',
        26: 'cluster-27',
        27: 'cluster-28',
        28: 'cluster-29',
        29: 'cluster-30',
        30: 'cluster-31',
        31: 'cluster-32',
        32: 'cluster-33',
        33: 'cluster-34',
        34: 'cluster-35',
        35: 'cluster-36',
        36: 'cluster-37',
        37: 'cluster-38',
        38: 'cluster-39',
        39: 'cluster-40',
        40: 'cluster-41',
        41: 'cluster-42',
        42: 'cluster-43',
        43: 'cluster-44',
        44: 'cluster-45',
        45: 'cluster-46',
    }
    plt.figure(figsize=(15, 15))
    theta = np.arange(0, 2 * np.pi, 0.01)
    flag = True
    for mb in mb_list:
        if flag:
            plt.scatter(mb.data[:, 0], mb.data[:, 1], s=4, c="blue", linewidths=5, alpha=0.9,
                        marker='o', label='connect')
            flag = False
        else:
            plt.scatter(mb.data[:, 0], mb.data[:, 1], s=4, c="blue", linewidths=5, alpha=0.9,
                        marker='o')
        center = mb.data[:, :-1].mean(0)
        r = mb.get_radius()
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        plt.plot(x, y, c="blue")
    plt.title("合并后")
    plt.legend(loc=1, fontsize=12)
    plt.show()


def gb_plot_test1(gb_list, mb_list):
    color = {
        0: '#707afa',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',

        25: '#ff8444',
        26: '#a6dceb',
        27: '#fdd3a2',
        28: '#e6b1c2',
        29: '#9bb7d4',
        30: '#fedb5c',
        31: '#b2e1e0',
        32: '#f8c0b6',
        33: '#c8bfe7',
        34: '#f4af81',
        35: '#a3a3a3',
        36: '#bce784',
        37: '#8d6e63',
        38: '#e9e3c9',
        39: '#f5e9b2',
        40: '#ffba49',
        41: '#c0c0c0',
        42: '#d3a7b5',
        43: '#f2c2e0',
        44: '#b7dd29',
        45: '#dcf7c1',
    }
    label_c = {
        0: 'cluster-1',
        1: 'cluster-2',
        2: 'cluster-3',
        3: 'cluster-4',
        4: 'cluster-5',
        5: 'cluster-6',
        6: 'cluster-7',
        7: 'cluster-8',
        8: 'cluster-9',
        9: 'cluster-10',
        10: 'cluster-11',
        11: 'cluster-12',
        12: 'cluster-13',
        13: 'cluster-14',
        14: 'cluster-15',
        15: 'cluster-16',
        16: 'cluster-17',
        17: 'cluster-18',
        18: 'cluster-19',
        19: 'cluster-20',
        20: 'cluster-21',
        21: 'cluster-22',
        22: 'cluster-23',
        23: 'cluster-24',
        24: 'cluster-25',
        25: 'cluster-26',
        26: 'cluster-27',
        27: 'cluster-28',
        28: 'cluster-29',
        29: 'cluster-30',
        30: 'cluster-31',
        31: 'cluster-32',
        32: 'cluster-33',
        33: 'cluster-34',
        34: 'cluster-35',
        35: 'cluster-36',
        36: 'cluster-37',
        37: 'cluster-38',
        38: 'cluster-39',
        39: 'cluster-40',
        40: 'cluster-41',
        41: 'cluster-42',
        42: 'cluster-43',
        43: 'cluster-44',
        44: 'cluster-45',
        45: 'cluster-46',
    }
    plt.figure(figsize=(15, 15))
    theta = np.arange(0, 2 * np.pi, 0.01)
    flag = True
    for mb in mb_list:
        if flag:
            plt.scatter(mb.data[:, 0], mb.data[:, 1], s=4, c="blue", linewidths=5, alpha=0.9,
                        marker='o', label='origin')
            flag = False
        else:
            plt.scatter(mb.data[:, 0], mb.data[:, 1], s=4, c="blue", linewidths=5, alpha=0.9,
                        marker='o')
        center = mb.data[:, :-1].mean(0)
        r = mb.get_radius()
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        plt.plot(x, y, c="blue")
    flag = True
    for gb in gb_list:
        if flag:
            plt.scatter(gb.data[:, 0], gb.data[:, 1], s=4, c="red", linewidths=5, alpha=0.9,
                        marker='o', label='new')
            flag = False
        else:
            plt.scatter(gb.data[:, 0], gb.data[:, 1], s=4, c="red", linewidths=5, alpha=0.9,
                        marker='o')
        center = gb.data[:, :-1].mean(0)
        r = gb.get_radius()
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        plt.plot(x, y, c="red")
    plt.title("合并前")
    plt.legend(loc=1, fontsize=12)
    plt.show()


def gb_plot_in_split(gb_list, split_count, lab):
    sum = 0
    for gb in gb_list:
        sum += len(gb)
    print(str(split_count) + " sum: " + str(sum))
    color = {
        0: '#707afa',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',
        25: '#ff8444',
        26: '#a6dceb',
        27: '#fdd3a2',
        28: '#e6b1c2',
        29: '#9bb7d4',
        30: '#fedb5c',
        31: '#b2e1e0',
        32: '#f8c0b6',
        33: '#c8bfe7',
        34: '#f4af81',
        35: '#a3a3a3',
        36: '#bce784',
        37: '#8d6e63',
        38: '#e9e3c9',
        39: '#f5e9b2',
        40: '#ffba49',
        41: '#c0c0c0',
        42: '#d3a7b5',
        43: '#f2c2e0',
        44: '#b7dd29',
        45: '#dcf7c1',
        46: '#62b1a0',
        47: '#eb5c2b',
        48: '#95774d',
        49: '#fad7e7',
        50: '#69a2b2',
        51: '#ffdf6b',
        52: '#9d8fae',
        53: '#f7a798',
        54: '#c4e96e',
        55: '#c49e7d',
        56: '#fdb0b3',
        57: '#6d6968',
        58: '#e1f394',
        59: '#f5b2e1',
        60: '#bdeed6',
        61: '#fad3c0',
        62: '#888c46',
        63: '#bebebe',
        64: '#b3b3c1',
        65: '#fbd8cc',
        66: '#788476',
        67: '#edbea3',
        68: '#aad5a1',
        69: '#e8e57f',
        70: '#b8a1d6',
        71: '#c7d0db',
        72: '#b4dcd9',
        73: '#f7d792',
        74: '#a7b5d6',
        75: '#b4dab7',
        76: '#fde38e',
        77: '#c1d1c7',
        78: '#ffb7c5',
        79: '#a3a3a3',
        80: '#b8d4e5',
        81: '#f4cfae',
        82: '#a8c0a0',
        83: '#edf9c3',
        84: '#df9eaf',
        85: '#b6e2ed',
        86: '#ffef88',
        87: '#c5c8a2',
        88: '#d8d8d8',
        89: '#c1c6dd',
        90: '#eb8a77',
        91: '#e2d3ae',
        92: '#b2ddcd',
        93: '#f1c4d1',
        94: '#b9b9b9',
        95: '#d1eeff',
        96: '#ffd93f',
        97: '#c7bfb7',
        98: '#f1f0d7',
        99: '#d2d2d2',
        100: '#f8e8d3',
    }
    plt.figure(figsize=(20, 20), dpi=80)
    theta = np.arange(0, 2 * np.pi, 0.01)
    flag = True
    i = 0
    for gb in gb_list:
        if flag:
            plt.scatter(gb[:, 0], gb[:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
                        marker='o', label='new')
            flag = False
        else:
            plt.scatter(gb[:, 0], gb[:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
                        marker='o')
        center = gb[:, :-1].mean(0)
        plt.scatter(center[0], center[1], s=4, c="blue", linewidths=10, alpha=0.9,
                    marker='x')
        r = max(((gb[:, :-1] - center) ** 2).sum(axis=1) ** 0.5)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        plt.plot(x, y, c=color[i])
        i += 1
    plt.title(str(split_count))
    plt.legend(loc=1, fontsize=12)
    plt.savefig("E:\pythonProject\GB-Stream\\result\split\\" + lab + str(split_count) + ".png")
    plt.show()


def gb_plot(gb_dict, noise, t, data, trueLabel):
    datalist = data.tolist()
    # plt.axes([0.7, 0.3, 0.3, 0.3])
    color = {
        0: '#707afa',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',

        25: '#ff8444',
        26: '#a6dceb',
        27: '#fdd3a2',
        28: '#e6b1c2',
        29: '#9bb7d4',
        30: '#fedb5c',
        31: '#b2e1e0',
        32: '#f8c0b6',
        33: '#c8bfe7',
        34: '#f4af81',
        35: '#a3a3a3',
        36: '#bce784',
        37: '#8d6e63',
        38: '#e9e3c9',
        39: '#f5e9b2',
        40: '#ffba49',
        41: '#c0c0c0',
        42: '#d3a7b5',
        43: '#f2c2e0',
        44: '#b7dd29',
        45: '#dcf7c1',
        46: '#6f9ed7',
        47: '#d8a8c3',
        48: '#76c57f',
        49: '#f6e9cd',
        50: '#a16fd8',
        51: '#c5e6a7',
        52: '#f98f76',
        53: '#b3d6e3',
        54: '#efc8a5',
        55: '#5c9aa1',
        56: '#d3e1b6',
        57: '#a87ac8',
        58: '#e2d095',
        59: '#c95a3b',
        60: '#7fb4d1',
        61: '#f7d28e',
        62: '#b9c9b0',
        63: '#e994b9',
        64: '#8bc9e4',
        65: '#e6b48a',
        66: '#acd4d8',
        67: '#f3e0b0',
        68: '#57a773',
        69: '#d9bb7b',
        70: '#8e73e5',
        71: '#f4c4e3',
        72: '#75a88b',
        73: '#c0d4eb',
        74: '#a46c9b',
        75: '#d7e3a0',
        76: '#bd5f36',
        77: '#77c5b8',
        78: '#e8b7d5',
        79: '#4e8746',
        80: '#f0d695',
        81: '#9b75cc',
        82: '#c2e68a',
        83: '#f56e5c',
        84: '#a9ced0',
        85: '#e18a6d',
        86: '#6291b1',
        87: '#d1dbab',
        88: '#c376c5',
        89: '#8fc9b5',
        90: '#f7e39e',
        91: '#6d96b8',
        92: '#f9c0a6',
        93: '#63a77d',
        94: '#dbb8e9',
        95: '#9aa3d6',
        96: '#e3ca7f',
        97: '#b15d95',
        98: '#88c2e0',
        99: '#f4c995',
        100: '#507c94',
    }
    label_c = {
        0: 'clu-1',
        1: 'clu-2',
        2: 'clu-3',
        3: 'clu-4',
        4: 'clu-5',
        5: 'clu-6',
        6: 'clu-7',
        7: 'clu-8',
        8: 'clu-9',
        9: 'clu-10',
        10: 'cluster-11',
        11: 'cluster-12',
        12: 'cluster-13',
        13: 'cluster-14',
        14: 'cluster-15',
        15: 'cluster-16',
        16: 'cluster-17',
        17: 'cluster-18',
        18: 'cluster-19',
        19: 'cluster-20',
        20: 'cluster-21',
        21: 'cluster-22',
        22: 'cluster-23',
        23: 'cluster-24',
        24: 'cluster-25',
        25: 'cluster-26',
        26: 'cluster-27',
        27: 'cluster-28',
        28: 'cluster-29',
        29: 'cluster-30',
        30: 'cluster-31',
        31: 'cluster-32',
        32: 'cluster-33',
        33: 'cluster-34',
        34: 'cluster-35',
        35: 'cluster-36',
        36: 'cluster-37',
        37: 'cluster-38',
        38: 'cluster-39',
        39: 'cluster-40',
        40: 'cluster-41',
        41: 'cluster-42',
        42: 'cluster-43',
        43: 'cluster-44',
        44: 'cluster-45',
        45: 'cluster-46',
        46: 'cluster-47',
        47: 'cluster-48',
        48: 'cluster-49',
        49: 'cluster-50',
        50: 'cluster-51',
        51: 'cluster-52',
        52: 'cluster-53',
        53: 'cluster-54',
        54: 'cluster-55',
        55: 'cluster-56',
        56: 'cluster-57',
        57: 'cluster-58',
        58: 'cluster-59',
        59: 'cluster-60',
        60: 'cluster-61',
        61: 'cluster-62',
        62: 'cluster-63',
        63: 'cluster-64',
        64: 'cluster-65',
        65: 'cluster-66',
        66: 'cluster-67',
        67: 'cluster-68',
        68: 'cluster-69',
        69: 'cluster-70',
        70: 'cluster-71',
        71: 'cluster-72',
        72: 'cluster-73',
        73: 'cluster-74',
        74: 'cluster-75',
        75: 'cluster-76',
        76: 'cluster-77',
        77: 'cluster-78',
        78: 'cluster-79',
        79: 'cluster-80',
        80: 'cluster-81',
        81: 'cluster-82',
        82: 'cluster-83',
        83: 'cluster-84',
        84: 'cluster-85',
        85: 'cluster-86',
        86: 'cluster-87',
        87: 'cluster-88',
        88: 'cluster-89',
        89: 'cluster-90',
        90: 'cluster-91',
        91: 'cluster-92',
        92: 'cluster-93',
        93: 'cluster-94',
        94: 'cluster-95',
        95: 'cluster-96',
        96: 'cluster-97',
        97: 'cluster-98',
        98: 'cluster-99',
        99: 'cluster-100',
        100: 'cluster-101',
    }
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # fig = plt.subplot(122)
    x_min = 1
    x_max = 0
    cluster_label_list = []
    for i in range(0, len(gb_dict)):
        if gb_dict[i].label not in cluster_label_list:
            cluster_label_list.append(gb_dict[i].label)
    plt.figure(figsize=(12, 12))
    # 设置坐标轴标签的字体大小
    plt.rc('xtick', labelsize=25)  # x轴刻度标签的字体大小
    plt.rc('ytick', labelsize=25)  # y轴刻度标签的字体大小

    for i in range(0, len(cluster_label_list)):
        if cluster_label_list[i] == -1:
            cluster_label_list.remove(-1)
            break
    cluster = {}
    for label in cluster_label_list:
        for key in gb_dict.keys():
            if gb_dict[key].label == label:
                if label not in cluster.keys():
                    cluster[label] = gb_dict[key].data
                else:
                    cluster[label] = np.append(cluster[label], gb_dict[key].data, axis=0)

    theta = np.arange(0, 2 * np.pi, 0.01)

    for i, key in enumerate(cluster.keys()):
        # plt.scatter(cluster[key][:, 0], cluster[key][:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
        #             marker='o', label=label_c[i])
        plt.scatter(cluster[key][:, 0], cluster[key][:, 1], c=color[i],
                    label=label_c[i])


    # 画圆
    # for gb in gb_dict.values():
    #     if gb.label == key:
    #         center = gb.data[:, :-1].mean(0)
    #         r = gb.get_radius()
    #         x = center[0] + r * np.cos(theta)
    #         y = center[1] + r * np.sin(theta)
    #         plt.plot(x, y, c=color[i])

    # for i in range(0, len(cluster_label_list)):
    #     if i >= 25:
    #         c = "black"
    #         label = "too many cluster"
    #     else:
    #         c = color[i]
    #         label = label_c[i]
    #     for key in gb_dict.keys():
    #         if gb_dict[key].label == cluster_label_list[i]:
    #             plt.scatter(gb_dict[key].data[:, 0], gb_dict[key].data[:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
    #                         marker='o', label=label_c[i])
    #             # fig.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=4, c=c, linewidths=5, alpha=0.9,
    #             #             marker='o', label=label)
    #             break
    #
    # for key in gb_dict.keys():
    #     for i in range(0, len(cluster_label_list)):
    #         if i >= 25:
    #             c = "black"
    #             label = "too many cluster"
    #         else:
    #             c = color[i]
    #             label = label_c[i]
    #         if gb_dict[key].label == cluster_label_list[i]:
    #             plt.scatter(gb_dict[key].data[:, 0], gb_dict[key].data[:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
    #                         marker='o')
    #             # fig.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=4, c=c, linewidths=5, alpha=0.9,
    #             #             marker='o')

    if len(noise) > 0:
        plt.scatter(noise[:, 0], noise[:, 1], s=40, c='black', linewidths=2, alpha=1, marker='x', label='noise')
        # fig.scatter(noise[:, 0], noise[:, 1], s=40, c='black', linewidths=2, alpha=1, marker='x', label='noise')

    for key in gb_dict.keys():
        for i in range(0, len(cluster_label_list)):
            if gb_dict[key].label == -1:
                plt.scatter(gb_dict[key].data[:, 0], gb_dict[key].data[:, 1], s=40, c='black', linewidths=2, alpha=1,
                            marker='x')
                # fig.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=40, c='black', linewidths=2, alpha=1,
                #             marker='x')

    # plt.legend(loc=3, fontsize=25)

    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # fig.legend(loc=1, fontsize=12)
    plt.title("t = " + str(int(t)), size=35, loc='center')
    plt.savefig("../notebooks/fig/RBF3/" + str(t) + ".jpg")

    # plt.show()
    # for i in range(0, len(part_sample)):
    #     plt.scatter(part_sample[i][0], part_sample[i][1], c=colors[part_cla[i]])
    #     cluster.add(part_cla[i])
    # plt.title("cluser num = " + str(len(cluster)))
    # plt.savefig(save_path + "/" + str(n) + ".png")
    truth_dict = {}
    plt.figure(figsize=(12, 12))
    # 设置坐标轴标签的字体大小
    plt.rc('xtick', labelsize=25)  # x轴刻度标签的字体大小
    plt.rc('ytick', labelsize=25)  # y轴刻度标签的字体大小
    flag = True
    for i in range(0, len(gb_dict)):
        for data in gb_dict[i].data.tolist():
            if trueLabel[datalist.index(data)] == "nan":
                if flag:
                    plt.scatter(data[0], data[1], c='b', s=4, linewidths=6, marker='|', label='noise')  # 绘制噪声图例
                    flag = False
                else:
                    plt.scatter(data[0], data[1], c='b', s=4, linewidths=6, marker='|')
            else:
                plt.scatter(data[0], data[1], c=color[int(float((trueLabel[datalist.index(data)])))])
                truth_dict[int(float((trueLabel[datalist.index(data)])))] = data
    for i, key in enumerate(truth_dict.keys()):
        # plt.scatter(cluster[key][:, 0], cluster[key][:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
        #             marker='o', label=label_c[i])
        plt.scatter(np.array(truth_dict[key])[0], np.array(truth_dict[key])[1], s=10, c=color[i],
                    label=label_c[i])
        # print(np.array(truth_dict[key])[0])

    plt.legend(loc=3, fontsize=25)

    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.title("t = " + str(int(t)), size=35, loc='center')
    plt.savefig("../notebooks/fig/RBF3_groundtruth/" + str(t) + ".jpg")  # RBF3_groundtruth
    # plt.show()

    # return fig.findobj()


def draw_ball(hb_list):
    fig = plt.subplot(121)
    for data in hb_list:
        if len(data) > 1:
            center = data.mean(0)
            radius = np.max((((data - center) ** 2).sum(axis=1) ** 0.5))
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            # plt.plot(x, y, ls='-', color='black', lw=0.7)
            fig.plot(x, y, ls='-', color='black', lw=0.7)
        else:
            # plt.plot(data[0][0], data[0][1], marker='*', color='#0000EF', markersize=3)
            fig.plot(data[0][0], data[0][1], marker='*', color='#0000EF', markersize=3)
    # plt.plot(x, y, ls='-', color='black', lw=1.2, label='hyper-ball boundary')
    # plt.legend(loc=1)
    fig.legend(loc=1)
    # plt.show()
    return fig.findobj()


def connect_ball0(gb_list, noise, c_count):
    gb_cluster = {}
    for i in range(0, len(gb_list)):  # 生成每个粒球对象
        gb = GB(gb_list[i], i)
        gb_cluster[i] = gb

    radius_sum = 0  # 总半径
    num_sum = 0  # 总样本数
    hb_len = 0  # 总粒球数
    radius_sum = 0
    num_sum = 0
    for i in range(0, len(gb_cluster)):  # 计算总粒球数、总样本数、总半径
        if gb_cluster[i].out == 0:
            hb_len = hb_len + 1
            radius_sum = radius_sum + gb_cluster[i].radius
            num_sum = num_sum + gb_cluster[i].num

    for i in range(0, len(gb_cluster) - 1):
        if gb_cluster[i].out != 1:
            center_i = gb_cluster[i].center
            radius_i = gb_cluster[i].radius
            for j in range(i + 1, len(gb_cluster)):
                if gb_cluster[j].out != 1:
                    center_j = gb_cluster[j].center
                    radius_j = gb_cluster[j].radius
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5  # 计算第i个粒球和第j个粒球的欧式距离
                    if (dis <= radius_i + radius_j) & ((gb_cluster[i].hardlapcount == 0) & (  # 重叠统计
                            gb_cluster[j].hardlapcount == 0)):  # 由于两个的点是噪声球，所以纳入重叠统计
                        gb_cluster[i].overlap = 1
                        gb_cluster[j].overlap = 1
                        gb_cluster[i].hardlapcount = gb_cluster[i].hardlapcount + 1
                        gb_cluster[j].hardlapcount = gb_cluster[j].hardlapcount + 1

    hb_uf = UF(len(gb_list))
    for i in range(0, len(gb_cluster) - 1):
        if gb_cluster[i].out != 1:
            center_i = gb_cluster[i].center
            radius_i = gb_cluster[i].radius
            for j in range(i + 1, len(gb_cluster)):
                if gb_cluster[j].out != 1:
                    center_j = gb_cluster[j].center
                    radius_j = gb_cluster[j].radius
                    max_radius = max(radius_i, radius_j)
                    min_radius = min(radius_i, radius_j)
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if c_count == 1:
                        dynamic_overlap = dis < radius_i + radius_j + 1 * min_radius / (
                                min(gb_cluster[i].hardlapcount, gb_cluster[j].hardlapcount) + 1)
                    if c_count == 2:
                        dynamic_overlap = dis <= radius_i + radius_j + 1 * max_radius / (
                                min(gb_cluster[i].hardlapcount, gb_cluster[j].hardlapcount) + 1)
                    num_limit = ((gb_cluster[i].num > 2) & (gb_cluster[j].num > 2))
                    if dynamic_overlap & num_limit:
                        gb_cluster[i].flag = 1
                        gb_cluster[j].flag = 1
                        hb_uf.union(i, j)
                    if dis <= radius_i + radius_j + max_radius:
                        gb_cluster[i].softlapcount = 1
                        gb_cluster[j].softlapcount = 1

    for i in range(0, len(gb_cluster)):
        k = i
        if hb_uf.parent[i] != i:
            while hb_uf.parent[k] != k:
                k = hb_uf.parent[k]
        hb_uf.parent[i] = k

    for i in range(0, len(gb_cluster)):
        gb_cluster[i].label = hb_uf.parent[i]
        gb_cluster[i].size = hb_uf.size[i]

    label_num = set()
    for i in range(0, len(gb_cluster)):
        label_num.add(gb_cluster[i].label)

    list = []
    for i in range(0, len(label_num)):
        list.append(label_num.pop())

    for i in range(0, len(gb_cluster)):
        if (gb_cluster[i].hardlapcount == 0) & (gb_cluster[i].softlapcount == 0):
            gb_cluster[i].flag = 0

    for i in range(0, len(list)):
        count_ball = 0
        count_data = 0
        list1 = []
        for key in range(0, len(gb_cluster)):
            if gb_cluster[key].label == list[i]:
                count_ball += 1
                count_data += gb_cluster[key].num
                list1.append(key)
        while count_ball < 6:
            for j in range(0, len(list1)):
                gb_cluster[list1[j]].flag = 0
            break

    for i in range(0, len(gb_cluster)):
        distance = np.sqrt(2)
        if gb_cluster[i].flag == 0:
            for j in range(0, len(gb_cluster)):
                if gb_cluster[j].flag == 1:
                    center = gb_cluster[i].center
                    center2 = gb_cluster[j].center
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5 - (gb_cluster[i].radius + gb_cluster[j].radius)
                    if dis < distance:
                        distance = dis
                        gb_cluster[i].label = gb_cluster[j].label
                        gb_cluster[i].flag = 2
            for k in range(0, len(noise)):
                center = gb_cluster[i].center
                dis = ((center - noise[k]) ** 2).sum(axis=0) ** 0.5
                if dis < distance:
                    distance = dis
                    gb_cluster[i].label = -1
                    gb_cluster[i].flag = 2

    label_num = set()
    for i in range(0, len(gb_cluster)):
        label_num.add(gb_cluster[i].label)
    return gb_cluster


def connect_ball0(gb_list, noise, c_count):
    gb_dist = {}
    for i in range(0, len(gb_list)):
        gb = GB(gb_list[i], i)
        gb_dist[i] = gb

    radius_sum = 0
    num_sum = 0

    gblen = 0
    radius_sum = 0
    num_sum = 0
    for i in range(0, len(gb_dist)):
        if gb_dist[i].out == 0:
            gblen = gblen + 1
            radius_sum = radius_sum + gb_dist[i].radius
            num_sum = num_sum + gb_dist[i].num

    for i in range(0, len(gb_dist) - 1):
        if gb_dist[i].out != 1:
            center_i = gb_dist[i].center
            radius_i = gb_dist[i].radius
            for j in range(i + 1, len(gb_dist)):
                if gb_dist[j].out != 1:
                    center_j = gb_dist[j].center
                    radius_j = gb_dist[j].radius
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if (dis <= radius_i + radius_j) & (
                            (gb_dist[i].hardlapcount == 0) & (gb_dist[j].hardlapcount == 0)):  # 由于两个的点是噪声球，所以纳入重叠统计
                        gb_dist[i].overlap = 1
                        gb_dist[j].overlap = 1
                        gb_dist[i].hardlapcount = gb_dist[i].hardlapcount + 1
                        gb_dist[j].hardlapcount = gb_dist[j].hardlapcount + 1

    # for i in range(0, len(gb_dist)):
    #     if(gb_dist[i].num > 20):
    #         print('this is hardlapount test:',gb_dist[i].hardlapcount)
    #         print('this is num test:',gb_dist[i].num)

    gb_uf = UF(len(gb_list))
    for i in range(0, len(gb_dist) - 1):
        if gb_dist[i].out != 1:
            center_i = gb_dist[i].center
            radius_i = gb_dist[i].radius
            for j in range(i + 1, len(gb_dist)):
                if gb_dist[j].out != 1:
                    center_j = gb_dist[j].center
                    radius_j = gb_dist[j].radius
                    max_radius = max(radius_i, radius_j)
                    min_radius = min(radius_i, radius_j)
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if c_count == 1:
                        dynamic_overlap = dis < radius_i + radius_j + 1 * min_radius / (
                                max(gb_dist[i].hardlapcount, gb_dist[j].hardlapcount) + 1)
                    if c_count == 2:
                        dynamic_overlap = dis <= radius_i + radius_j + 1 * min_radius / (
                                max(gb_dist[i].hardlapcount, gb_dist[j].hardlapcount) + 1)
                    # dynamic_overlap = dis <= radius_i + radius_j + min_radius
                    num_limit = ((gb_dist[i].num > 2) & (gb_dist[j].num > 2))
                    if dynamic_overlap:
                        gb_dist[i].flag = 1
                        gb_dist[j].flag = 1
                        gb_uf.union(i, j)
                    # 原来是1.5
                    if dis <= radius_i + radius_j + 3 * max_radius:
                        gb_dist[i].softlapcount += 1
                        gb_dist[j].softlapcount += 1
                        gb_uf.union(i, j)

    for i in range(0, len(gb_dist)):
        k = i
        if gb_uf.parent[i] != i:
            while gb_uf.parent[k] != k:
                k = gb_uf.parent[k]
        gb_uf.parent[i] = k

    for i in range(0, len(gb_dist)):
        gb_dist[i].label = gb_uf.parent[i]
        gb_dist[i].size = gb_uf.size[i]

    label_num = set()
    for i in range(0, len(gb_dist)):
        label_num.add(gb_dist[i].label)

    list = []
    for i in range(0, len(label_num)):
        list.append(label_num.pop())

    # 对孤立球flag重置为0，再就近分配
    for i in range(0, len(gb_dist)):
        if (gb_dist[i].hardlapcount == 0) and (gb_dist[i].softlapcount == 0):
            # gb_dist[i].label = -1
            gb_dist[i].flag = 0
        # if(gb_dist[i].num == 2):
        #     gb_dist[i].flag = 0
    # # 对噪声簇进行筛选，重置flag
    # for i in range(0, len(list)):
    #     count_ball = 0
    #     count_data = 0  # 噪声簇不但要考虑簇中球的数量还要考虑数据个数
    #     list1 = []
    #     for key in range(0, len(gb_dist)):
    #         if gb_dist[key].label == list[i]:
    #             count_ball += 1
    #             count_data += gb_dist[key].num
    #             list1.append(key)
    #     while count_ball < 6:
    #         for j in range(0, len(list1)):
    #             gb_dist[list1[j]].flag = 0
    #         break

    # #test cluster
    # label_num = set()
    # for i in range(0, len(gb_dist)):
    #     label_num.add(gb_dist[i].label)
    # gb_plot(gb_dist,noise)

    for i in range(0, len(gb_dist)):
        distance = np.sqrt(2)  # 数据归一化到1，数据点对的距离最大不会超过根号2
        if gb_dist[i].flag == 0:
            for j in range(0, len(gb_dist)):
                if gb_dist[j].flag == 1:
                    center = gb_dist[i].center
                    center2 = gb_dist[j].center
                    # dis = ((center - center2) ** 2).sum(axis=0) ** 0.5
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5 - (gb_dist[i].radius + gb_dist[j].radius)
                    if dis < distance:
                        distance = dis
                        gb_dist[i].label = gb_dist[j].label
                        gb_dist[i].flag = 2
            for k in range(0, len(noise)):
                center = gb_dist[i].center
                dis = ((center - noise[k]) ** 2).sum(axis=0) ** 0.5
                if dis < distance:
                    distance = dis
                    gb_dist[i].label = -1
                    gb_dist[i].flag = 2

    label_num = set()
    for i in range(0, len(gb_dist)):
        label_num.add(gb_dist[i].label)
    # print("聚类簇数为", len(label_num))
    # gb_plot(gb_dist, noise)
    return gb_dist



def connect_ball(gb_list, noise):
    gb_dist = {}
    for i in range(0, len(gb_list)):
        gb = GB(gb_list[i], i)
        gb_dist[i] = gb

    # gblen = 0
    # radius_sum = 0
    # num_sum = 0
    # for i in range(0, len(gb_dist)):
    #     if gb_dist[i].out == 0:
    #         # gblen = gblen + 1
    #         # radius_sum = radius_sum + gb_dist[i].radius
    #         num_sum = num_sum + gb_dist[i].num

    for i in range(0, len(gb_dist) - 1):

        center_i = gb_dist[i].center
        radius_i = gb_dist[i].radius
        for j in range(i + 1, len(gb_dist)):

            center_j = gb_dist[j].center
            radius_j = gb_dist[j].radius
            dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5 #这段代码计算了两个点center_i和center_j之间的欧氏距离

            # (gb_dist[i].hardlapcount == 0) & (gb_dist[j].hardlapcount == 0)

            if (dis <= radius_i + radius_j) & (
                    (gb_dist[i].hardlapcount == 0) & (gb_dist[j].hardlapcount == 0)):  # 由于两个的点是噪声球，所以纳入重叠统计
                # if dis <= radius_i + radius_j:
                gb_dist[i].overlap = 1
                gb_dist[j].overlap = 1
                gb_dist[i].hardlapcount = gb_dist[i].hardlapcount + 1
                gb_dist[j].hardlapcount = gb_dist[j].hardlapcount + 1

    # for i in range(0, len(gb_dist)):
    #     if(gb_dist[i].num > 20):
    #         print('this is hardlapount test:',gb_dist[i].hardlapcount)
    #         print('this is num test:',gb_dist[i].num)

    gb_uf = UF(len(gb_list))
    for i in range(0, len(gb_dist) - 1):

        center_i = gb_dist[i].center
        radius_i = gb_dist[i].radius
        for j in range(i + 1, len(gb_dist)):

            center_j = gb_dist[j].center
            radius_j = gb_dist[j].radius
            max_radius = max(radius_i, radius_j)
            min_radius = min(radius_i, radius_j)
            dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
            # if c_count == 1:
            #     dynamic_overlap = dis < radius_i + radius_j + 1 * min_radius / (
            #             min(gb_dist[i].hardlapcount, gb_dist[j].hardlapcount) + 1)
            # if c_count == 2:
            #     dynamic_overlap = dis <= radius_i + radius_j + 1 * min_radius / (
            #             min(gb_dist[i].hardlapcount, gb_dist[j].hardlapcount) + 1)
            dynamic_overlap = dis <= radius_i + radius_j + 1 * min_radius / (
                    min(gb_dist[i].hardlapcount, gb_dist[j].hardlapcount) + 1)
            # dynamic_overlap = dis <= radius_i + radius_j + min_radius
            # num_limit = ((gb_dist[i].num > 2) & (gb_dist[j].num > 2))
            if dynamic_overlap:
                gb_dist[i].flag = 1
                gb_dist[j].flag = 1
                gb_uf.union(i, j)
            if dis <= radius_i + radius_j + 3 * max_radius:
                gb_dist[i].softlapcount += 1
                gb_dist[j].softlapcount += 1
                gb_uf.union(i, j)

    for i in range(0, len(gb_dist)):
        k = i
        if gb_uf.parent[i] != i:
            while (gb_uf.parent[k] != k):
                k = gb_uf.parent[k]
        gb_uf.parent[i] = k

    for i in range(0, len(gb_dist)):
        gb_dist[i].label = gb_uf.parent[i]
        gb_dist[i].size = gb_uf.size[i]

    # label_num = set()
    # for i in range(0, len(gb_dist)):
    #     label_num.add(gb_dist[i].label)
    #
    # list = []
    # # list = list(label_num)
    # for i in range(0, len(label_num)):
    #     list.append(label_num.pop())

    # # 对孤立球flag重置为0，再就近分配
    # for i in range(0, len(gb_dist)):
    #     if (gb_dist[i].hardlapcount == 0) & (gb_dist[i].softlapcount == 0):
    #         # gb_dist[i].label = -1
    #         gb_dist[i].flag = 0
    #     # if(gb_dist[i].num == 2):
    #     #     gb_dist[i].flag = 0
    # # 对噪声簇进行筛选，重置flag
    # for i in range(0, len(list)):
    #     count_ball = 0
    #     count_data = 0  # 噪声簇不但要考虑簇中球的数量还要考虑数据个数
    #     list1 = []
    #     for key in range(0, len(gb_dist)):
    #         if gb_dist[key].label == list[i]:
    #             count_ball += 1
    #             count_data += gb_dist[key].num
    #             list1.append(key)
    #     while count_ball < 6:
    #         for j in range(0, len(list1)):
    #             gb_dist[list1[j]].flag = 0
    #         break

    # #test cluster
    # label_num = set()
    # for i in range(0, len(gb_dist)):
    #     label_num.add(gb_dist[i].label)
    # gb_plot(gb_dist,noise)
    # 没有重叠的粒球，
    for i in range(0, len(gb_dist)):
        distance = np.sqrt(2)  # 数据归一化到1，数据点对的距离最大不会超过根号2
        if gb_dist[i].flag == 0:
            for j in range(0, len(gb_dist)):
                if gb_dist[j].flag == 1:
                    center = gb_dist[i].center
                    center2 = gb_dist[j].center
                    # dis = ((center - center2) ** 2).sum(axis=0) ** 0.5
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5 - (gb_dist[i].radius + gb_dist[j].radius)
                    if dis < distance:
                        distance = dis
                        gb_dist[i].label = gb_dist[j].label
                        gb_dist[i].flag = 2
            for k in range(0, len(noise)):
                center = gb_dist[i].center
                dis = ((center - noise[k]) ** 2).sum(axis=0) ** 0.5
                if dis < distance:
                    distance = dis
                    gb_dist[i].label = -1
                    gb_dist[i].flag = 2

    # label_num = set()
    # for i in range(0, len(gb_dist)):
    #     label_num.add(gb_dist[i].label)
    # print("聚类簇数为", len(label_num))
    # gb_plot(gb_dist, noise)
    return gb_dist


# 缩小粒球
def  minimum_ball(gb_list, radius_detect,index):
    gb_list_temp = []
    gb_list_temp_index=[]
    for i,gb_data in enumerate(gb_list):
        # if len(hb) < 2: stream
        if len(gb_data) <= 2:
            # gb_lis t_temp.append(gb_data)

            if (len(gb_data) == 2) and (get_radius(gb_data) > 1.0 * radius_detect):
                # print(get_radius(gb_data))
                gb_list_temp.append(np.array([gb_data[0], ]))
                gb_list_temp.append(np.array([gb_data[1], ]))
                gb_list_temp_index.append(index[i][0])
                gb_list_temp_index.append(index[i][1])


            else:
                gb_list_temp.append(gb_data)
                gb_list_temp_index.append(index[i])
        else:
            # if get_radius(gb_data) <= radius_detect:
            if get_radius(gb_data) <= 1.0 * radius_detect:
                gb_list_temp.append(gb_data)
                gb_list_temp_index.append(index[i])
            else:
                ball_1, ball_2,index_1,index_2 = spilt_ball_3(gb_data,index[i])  # 无模糊
                if len(ball_1)==0:
                    gb_list_temp.append(ball_2)
                    gb_list_temp_index.append(index_2)
                    continue
                if len(ball_2)==0:
                    gb_list_temp.append(ball_1)
                    gb_list_temp_index.append(index_1)
                    continue
                # ball_1, ball_2 = spilt_ball_fuzzy(gb_data)  # 有模糊
                if len(ball_1) == 1 or len(ball_2) == 1:
                    if get_radius(gb_data) > radius_detect:
                        gb_list_temp.extend([ball_1, ball_2])
                        gb_list_temp_index.extend([index_1,index_2])
                    else:
                        gb_list_temp.append(gb_data)
                        gb_list_temp_index.append(index)
                else:
                    gb_list_temp.extend([ball_1, ball_2])
                    gb_list_temp_index.extend([index_1, index_2])

    return gb_list_temp,gb_list_temp_index


# 归一化
def normalized_ball(gb_list, radius_detect):
    gb_list_temp = []
    for gb_data in gb_list:
        # if len(gb_data) < 2:  # stream
        if len(gb_data) <= 2:
            if (len(gb_data) == 2) and (get_radius(gb_data) > 1.5 * radius_detect):
                gb_list_temp.append(np.array([gb_data[0], ]))
                gb_list_temp.append(np.array([gb_data[1], ]))
            else:
                gb_list_temp.append(gb_data)
        else:
            if get_radius(gb_data[:, :-1]) <= 1.5 * radius_detect:
                # if get_radius(gb_data[:, :-1]) <= 0.8 * radius_detect:
                gb_list_temp.append(gb_data)
            else:
                # ball_1, ball_2 = spilt_ball(hb) # 无模糊
                ball_1, ball_2 = spilt_ball_fuzzy(gb_data)  # 有模糊
                # (zt)6.19
                # if len(ball_1) == 1 or len(ball_2) == 1:
                #     gb_list_temp.append(gb_data)
                # else:
                gb_list_temp.extend([ball_1, ball_2])

    return gb_list_temp


def normalized_ball_2(gb_dict, radius_mean, cluster_label_list):
    gb_list_temp = []
    for i in range(0, len(radius_mean)):
        for key in gb_dict.keys():
            if gb_dict[key].label == cluster_label_list[i]:
                # if hb_cluster[key].num < 2:
                if gb_dict[key].num < 2:
                    gb_list_temp.append(gb_dict[key].data)
                else:
                    # ball_1, ball_2 = spilt_ball(hb_cluster[key].data) # 无模糊
                    ball_1, ball_2 = spilt_ball_fuzzy(gb_dict[key].data)  # fuzzy
                    if gb_dict[key].radius <= 1.5 * radius_mean[i] or len(ball_1) * len(ball_2) == 0:
                        gb_list_temp.append(gb_dict[key].data)
                    else:
                        gb_list_temp.extend([ball_1, ball_2])
    return gb_list_temp


# 加载数据集
def load_data(key):
    dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(dir + "/synthetic/" + key + ".csv", header=None)
    data = df.values
    return data


def main(data):
    gb_list_temp = [data]  # 粒球集合,初始只有一个粒球[ [[data1],[data2],...], [[data1],[data2],...],... ]
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division_2_2(gb_list_temp)  # 粒球划分
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    radius = []  # 汇总所有粒球半径
    for gb_data in gb_list_temp:
        if len(gb_data) >= 2:  # 粒球中样本多于2个时，认定为合法粒球，并收集其粒球半径
            radius.append(get_radius(gb_data[:, :-1]))

    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius = max(radius_median, radius_mean)

    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = normalized_ball(gb_list_temp, radius)  # 归一化
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break
    noise = []
    gb_dict = connect_ball(gb_list_temp, noise)  # gb_dict key为标记粒球，并没有其他含意。

    label_data_num = {}  # {簇标签：样本个数}
    for i in range(0, len(gb_dict)):
        label_data_num.setdefault(gb_dict[i].label, 0)
        label_data_num[gb_dict[i].label] = label_data_num.get(gb_dict[i].label) + len(gb_dict[i].data)
    cluster_label_list = []  # 获取簇中样本数大于2的簇标签
    for key in label_data_num.keys():
        if label_data_num[key] > 2:
            cluster_label_list.append(key)
    # label = set()
    # for key in label_num.keys():
    #     if label_num[key] > 2:
    #         label.add(key)
    # list1 = []
    # for i in range(0, len(label)):
    #     list1.append(label.pop())

    # radius列表用于存储每个聚类的半径值。
    # gb_num_in_cluster列表用于存储每个聚类中的数据点数量。
    # radius_mean列表用于存储每个聚类的半径均值。
    radius = [0] * len(cluster_label_list)
    gb_num_in_cluster = [0] * len(cluster_label_list)
    radius_mean = [0] * len(cluster_label_list)
    for key in gb_dict.keys():
        for i in range(0, len(cluster_label_list)):
            if gb_dict[key].label == cluster_label_list[i]:
                radius[i] = radius[i] + gb_dict[key].radius
                gb_num_in_cluster[i] = gb_num_in_cluster[i] + 1

    for i in range(0, len(cluster_label_list)):
        radius_mean[i] = radius[i] / gb_num_in_cluster[i]

    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = normalized_ball_2(gb_dict, radius_mean, cluster_label_list)  # 归一化
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break
    # plot_dot(data)  # 画点
    # draw_ball(hb_list_temp)  # 画球
    gb_final_list = gb_list_temp
    noise = []
    gb_dict = connect_ball(gb_final_list, noise)  # 最终聚类结果，第5次画图
    # gb_plot(gb_dict, noise)
    # print("===============================", (len(gb_list_cluster) == len(gb_list_final)), "===========")
    # print("clustering complete")

    # labels = set()
    # for i in gb_list_cluster:
    #     labels.add(gb_list_cluster[i].label)
    # print(labels)
    return gb_final_list, gb_dict

#
# if __name__ == '__main__':
#     csv = pd.read_csv("../data/window/cth.csv", header=None).values[:, :-1]
#     main(data=csv)
