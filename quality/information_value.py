# -*- coding: utf-8 -*-
# Author: lkm
# date: 2023/8/21 23:41

import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class WOETransform:
    def __init__(self, ndigits=4, latex=True):
        self.ndigits = ndigits

        if latex:
            print("\033[93m{}\033[0m".format('WOE计算公式：woe=ln(正例占比/负例占比)'))

    def transform(self, feature, target):
        # 将特征和目标变量转换为NumPy数组
        feature = np.array(feature)
        target = np.array(target)

        if self.is_integer(feature) and self.is_integer(target):
            # 创建一个包含特征和目标变量的DataFrame
            data = pd.DataFrame({'feature': feature, 'target': target})

            # 计算每个单元的正例数、负例数和总例数
            grouped = data.groupby('feature')['target'].agg(['sum', 'count'])
            grouped.columns = ['positive', 'total']

            # 计算总的正例数和总的负例数
            total_positive = grouped['positive'].sum()
            total_negative = grouped['total'].sum() - total_positive

            # 计算每个单元的正例占比、负例占比和IV值
            grouped['positive_rate'] = grouped['positive'] / total_positive
            grouped['negative_rate'] = (grouped['total'] - grouped['positive']) / total_negative
            grouped['woe'] = np.log(grouped['positive_rate'] / grouped['negative_rate'])
        else:
            raise TypeError('\033[91m{}\033[1m'.format('数据类型错误，请转换为整数型再进行操作！'))

        mapping = grouped['woe'].apply(lambda x: round(x, self.ndigits))
        print(grouped)

        return grouped

    @staticmethod
    def is_integer(array):
        if array.dtype in (np.int8, np.int16, np.int32, np.int64):
            return True
        else:
            return False


def IV(feature, target, latex=True):
    woe = WOETransform()
    grouped = woe.transform(feature, target)

    if latex:
        print("\033[93m{}\033[0m".format('IV计算公式：IV=(正例占比 - 负例占比) * woe'))

    grouped['iv'] = (grouped['positive_rate'] - grouped['negative_rate']) * grouped['woe']

    # 计算特征的总IV值
    iv = grouped['iv'].sum()

    # 输出IV日志
    if iv < 0.02:
        print("\033[92mIV={:.2f}：{}\033[0m".format(iv, "几乎没有预测能力"))
    elif iv < 0.1:
        print("\033[92mIV={:.2f}：{}\033[0m".format(iv, "较弱的预测能力"))
    elif iv < 0.3:
        print("\033[92mIV={:.2f}：{}\033[0m".format(iv, "中等的预测能力"))
    elif iv < 0.5:
        print("\033[92mIV={:.2f}：{}\033[0m".format(iv, "较强的预测能力"))
    elif iv >= 0.5 and not np.isinf(iv):
        print("\033[92mIV={:.2f}：{}\033[0m".format(iv, "非常强的预测能力"))
    elif np.isinf(iv):
        print("\033[91mIV={}：{}\033[0m".format(iv, "数据存在问题，请检查是否存在缺失值或未进行分箱处理"))
    else:
        print("\033[91m{}\033[0m".format("无法计算IV"))

    return iv
