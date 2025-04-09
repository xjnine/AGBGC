# -*- coding: utf-8 -*-
"""
@Time ： 2023/5/13 12:05
@Auth ： daiminggao
@File ：granular_ball.py
@IDE ：PyCharm
@Motto:咕咕咕
"""
import numpy as np


class GranularBall:
    def __init__(self, data, gb_index):
        self.data = data
        self.center = self.get_center()
        self.radius = self.get_radius()
        self.overlap = 0
        self.label = -1
        self.index=gb_index



    def get_radius(self):
        return
        # if len(self.data) == 1:
        #     return 0
        # return max(((self.data[:, :] - self.center) ** 2).sum(axis=1) ** 0.5)
    
    def get_center(self):
        num_view=len(self.data)

        center = []
        for i in range(num_view):
            center.append(self.data[i].mean(0))
        return center

    def get_w_raduis(self):
        if len(self.data) == 1:
            return 0
        return max(((self.data[:, :] * self.w - self.w_center) ** 2).sum(axis=1) ** 0.5)
