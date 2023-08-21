# -*- coding: utf-8 -*-
# Author: lkm
# date: 2023/8/20 1:12

import warnings

warnings.filterwarnings('ignore')

import numpy
import math
import numpy as np
import pandas as pd
from pandas.core.series import Series
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def TreeNumBinning(x, target, **kwargs):
    """连续型特征的决策树分箱

    Parameters
    ----------
    x: 是一个形状(n,1)的数组或数据帧。
    target: 是一个形状(n,)的数组或序列。
    kwargs: sklearn接口下的决策树所有参数。

    Examples
    --------
    >>> import pandas as pd
    >>> from binning.tree import TreeBinning
    >>> data = pd.read_csv('dataset/train.csv')[['emp_var_rate', 'subscribe']]
    >>> data['subscribe'] = data['subscribe'].apply(lambda x: 0 if x == 'no' else 1)
    >>> data['emp_var_rate'] = TreeBinning(x=data['emp_var_rate'], y=data['subscribe'], max_leaf_nodes=3, criterion='gini')

    return
    --------
    返回一维的数组。
    """
    x = np.array(x).reshape(-1, 1)
    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(x, target)

    # 获取决策树的结构信息
    tree_structure = clf.tree_
    # 取出根结点与内部结点的分裂阈值，左开右闭
    non_leaf_thresholds = tree_structure.threshold[tree_structure.children_left != tree_structure.children_right]
    print("\033[92m{}{}\033[0m".format("分箱结点为：", [round(i, 2) for i in non_leaf_thresholds]))

    # 映射到分箱箱号
    binning_res = clf.apply(x)
    return binning_res


def TreeCateBinning(x, target, **kwargs):
    """字符型特征的决策树分箱

    Parameters
    ----------
    x: 是一个形状(n,1)的数组或数据帧。
    target: 是一个形状(n,)的数组或序列。
    kwargs: sklearn接口下的决策树所有参数。

    Examples
    --------
    >>> import pandas as pd
    >>> from binning.tree import TreeCateBinning
    >>> data = pd.read_csv('train.csv')[['incident_severity', 'fraud']]
    >>> TreeCateBinning(data['incident_severity'], data['fraud'], max_leaf_nodes=3, criterion='gini')

    return
    --------
    返回一维的数组。
    """
    if type(x) == Series:
        x = x.values
    elif type(x) != numpy.ndarray:
        x = np.array(x)

    nunique = len(set(x))
    if nunique < 3:
        print("\033[93m{}\033[0m".format("Warning：枚举值少于3个，不建议进行分箱处理！"))
        binning_res = None
    else:
        # 训练编码模型
        encoder = LabelEncoder().fit(x)
        # 映射值
        encoded_data = encoder.transform(x)
        # 生成映射字典
        mapping_dict = dict(zip(encoded_data, x))

        clf = DecisionTreeClassifier(**kwargs).fit(encoded_data.reshape(-1, 1), target)
        # 获取决策树的结构信息
        tree_structure = clf.tree_
        # 取出根结点与内部结点的分裂阈值，左开右闭
        non_leaf_thresholds = tree_structure.threshold[tree_structure.children_left != tree_structure.children_right]
        non_leaf_thresholds = [math.floor(i) for i in set(non_leaf_thresholds)]
        non_leaf_thresholds.append(non_leaf_thresholds[-1])

        result = {}
        for index, value in enumerate(non_leaf_thresholds):
            if index == 0:  # 如果等于首位
                sublist = []
                for _index in range(value + 1):  # 遍历从0开始到首位的值
                    sublist.append(mapping_dict[_index])
                result[index] = sublist
            elif index == len(non_leaf_thresholds) - 1:  # 如果索引位等于最大位置
                sublist = []
                for _index in range(value + 1, max(mapping_dict) + 1):  # 最大位置的值开始到映射表最大值进行遍历
                    sublist.append(mapping_dict[_index])
                result[index] = sublist
            else:
                sublist = []
                for _index in range(non_leaf_thresholds[index - 1] + 1, value + 1):  # 取上一位的值加1到当前索引的值，进行遍历
                    sublist.append(mapping_dict[_index])
                result[index] = sublist
        print("\033[92m{}{}\033[0m".format("分箱结点为：", result))
        # 映射到分箱箱号
        binning_res = clf.apply(x)
    return binning_res

