# -*- coding: utf-8 -*-
# Author: lkm
# date: 2023/8/20 0:51
import numpy
import numpy as np
import pandas as pd
from toad import quality
from binning.tree import *


print("\033[91m{}\033[0m".format("This is red text"))  # 红色
print("\033[92m{}\033[0m".format("This is green text"))  # 绿色
print("\033[94m{}\033[0m".format("This is blue text"))  # 蓝色


if __name__ == '__main__':
    data = pd.read_csv('dataset/保险反欺诈预测_train.csv')[['incident_severity', 'fraud']]
    TreeCateBinning(data['incident_severity'], data['fraud'], max_leaf_nodes=3, criterion='gini')

