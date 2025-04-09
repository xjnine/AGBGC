
# 判断粒球是都需要二分裂
import numpy as np
import torch


def division_2_2(gb_list,gb_data_index):
    gb_list_new = []
    gb_list_index_new=[]

    for i,gb_data in enumerate (gb_list):
        # 粒球内样本数大于等于8的粒球进行处理
        if len(gb_data[0]) >= 8:
            ball_1, ball_2,index_1,index_2 = spilt_ball_2(gb_data,gb_data_index[i])  # 无模糊

            # 如果划分的两个球中 其中一个球的内的样本数小于等于1 则该球不该划分
            if len(ball_1[0]) == 1 or len(ball_2[0]) == 1:
                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])
                continue
            if len(ball_1[0]) == 0 or len(ball_2[0]) == 0:
                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])
                continue

            parent_dm = get_density_volume(gb_data)
            child_1_dm = get_density_volume(ball_1)
            child_2_dm = get_density_volume(ball_2)
            w1 = len(ball_1[0]) / (len(ball_1[0]) + len(ball_2[0]))

            w2 = len(ball_2[0]) / (len(ball_1[0]) + len(ball_2[0]))

            w_child_dm = (w1 * child_1_dm + w2 * child_2_dm)  # 加权子粒球DM

            t1 = ((child_1_dm > parent_dm) & (child_2_dm > parent_dm))
            t2 = (w_child_dm > parent_dm)  # 加权DM上升
            t3 = ((len(ball_1) > 0) & (len(ball_2) > 0))  # 球中数据个数低于1个的情况不能分裂
            if t2:
                gb_list_new.extend([ball_1, ball_2])
                gb_list_index_new.extend([index_1,index_2])
            else:

                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])

        else:
            gb_list_new.append(gb_data)
            gb_list_index_new.append(gb_data_index[i])
    return gb_list_new, gb_list_index_new

def get_radius(gb_datas):
    num_view=len(gb_datas)
    radius=0
    for i in range(num_view):
        gb_data=gb_datas[i]
        # 通过计算每个样本点与中心点之间的距离，并取最大值作为半径。
        # origin get_radius 7*O(n)
        sample_num = len(gb_data)
        center = gb_data.mean(0)
        diffMat = np.tile(center, (sample_num, 1)) - gb_data
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        radius =radius+ max(distances)
    radius=radius/num_view
    return radius

# 缩小粒球
def minimum_ball(gb_list, radius_detect,index):
    gb_list_temp = []
    gb_list_temp_index=[]
    num_view = len(gb_list[0])
    for i,gb_data in enumerate(gb_list):
        # if len(hb) < 2: stream

        if len(gb_data[0]) <= 2:

            # gb_lis t_temp.append(gb_data)

            # 如果样本点等于2 并且半径大于探测半径 则直接分裂
            if (len(gb_data[0]) == 2) and (get_radius(gb_data) > 1.0* radius_detect):
                # print(get_radius(gb_data))
                # for j in range(num_view):
                temp0=[[] for _ in range(num_view)]
                temp1=[[] for _ in range(num_view)]

                for j in range(num_view):
                    temp0[j]=gb_data[j][0].reshape(1, -1)
                    temp1[j] =gb_data[j][1].reshape(1, -1)

                gb_list_temp.append(temp0)
                gb_list_temp.append(temp1)
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
                ball_1, ball_2,index_1,index_2 = spilt_ball_2(gb_data,index[i])  # 无模糊
                if len(ball_1[0])==0:
                    gb_list_temp.append(ball_2)
                    gb_list_temp_index.append(index_2)
                    continue
                if len(ball_2[0])==0:
                    gb_list_temp.append(ball_1)
                    gb_list_temp_index.append(index_1)
                    continue
                # ball_1, ball_2 = spilt_ball_fuzzy(gb_data)  # 有模糊
                if len(ball_1[0]) == 1 or len(ball_2[0]) == 1:
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


# 选择中心最远的点划分粒球
def spilt_ball_2(data,data_index):
    ball1 = []
    ball2 = []
    index1 = []
    index2 = []
    num_view=len(data)
    # 遍历每个视图求样本离中心的距离
    distances=[]
    center=[]
    for i in range(num_view):
        center.append(data[i].mean(axis=0))
        # distances.append(((data[i]-center[i])**2).sum(axis=1)**0.5)

        data[i] = torch.tensor(data[i])
        center[i] = torch.tensor(center[i])
        data[i] = data[i].detach().cpu().numpy()
        center[i] = center[i].detach().cpu().numpy()

        test = data[i] - center[i]
        # print(test.dtype)
        res = np.linalg.norm(test,axis=1)
        distances.append(res)
        # distances.append(np.linalg.norm(data[i] - center[i], axis=1))

    distance=np.mean(distances, axis=0)
    # p1_max_index=np.argmax(distance)
    p1_max_value = max(distance)
    p1_max_index = next((idx for idx, value in enumerate(distance) if value == p1_max_value), None)
    # p1_max_index = np.where(distance == p1_max_value)
    # p1_max_index=p1_max_index[0]
    # 遍历每个视图 求离p1最远的样本点 p2
    distances = []
    for i in range(num_view):
        # distances.append(((data[i]-data[i][p1_max_index])**2).sum(axis=1)**0.5)
        # distances.append(np.linalg.norm(data[i].detach().cpu().numpy() - data[i][p1_max_index].detach().cpu().numpy(), axis=1))
        distances.append(
            np.linalg.norm(data[i] - data[i][p1_max_index], axis=1))

    distance = np.mean(distances, axis=0)
    #
    p2_max_value = max(distance)
    p2_max_index = next((idx for idx, value in enumerate(distance) if value == p2_max_value), None)
    # p2_max_index=np.argmax(distance)
    # p2_max_index = np.where(distance == p2_max_value)
    # p2_max_index=p2_max_index[0]
    # p2_max_index = distance.index(p2_max_value)
    # 遍历每个视图 求c1 与c2
    c1=[]
    c2=[]
    for i in range(num_view):
        c1.append((data[i][p1_max_index]+center[i])/2)
        # c1.append(data[i][p1_max_index])
        # c2.append(data[i][p2_max_index])
        c2.append((data[i][p2_max_index]+center[i])/2)
    # 遍历球内的每个样本

    ball1 = [[] for _ in range(num_view)]
    ball2 = [[] for _ in range(num_view)]
    # D1 = 0
    # D2 = 0
    for j in range(0, len(data[0])):
        D1 = 0
        D2 = 0
        for i in range(num_view):
            D1=D1+(((data[i][j] - c1[i]) ** 2).sum() ** 0.5)
            D2=D2+(((data[i][j] - c2[i]) ** 2).sum() ** 0.5)

        D1=D1/num_view
        D2=D2/num_view
        if D2 > D1:
            for i in range(num_view):
                ball1[i].append(data[i][j])
            # ball1.extend([data[j, :]])
            index1.append(data_index[j])
        else:
            for i in range(num_view):
                ball2[i].append(data[i][j])
            # ball2.extend([data[j, :]])
            index2.append(data_index[j])
    for i in range(num_view):
        ball1[i]=np.array(ball1[i])
        ball2[i]=np.array(ball2[i])

    return [ball1, ball2,index1,index2]

# 获取球内平均的密度
def get_density_volume(gbs):
    num_view=len(gbs)
    density_volume=0
    # DM=[]
    for i in range(num_view):
        gb=gbs[i]
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

        for j in distances:
            sum_radius = sum_radius + j
        # 平均距离
        mean_radius = sum_radius / num

        if mean_radius != 0:
            density_volume =density_volume+ num / sum_radius

        else:
            density_volume =density_volume+ num
        # DM.append(density_volume)
    density_volume=density_volume/num_view
    return density_volume