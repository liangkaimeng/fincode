# -*- coding: utf-8 -*-
# Author: lkm
# date: 2023/8/20 0:51
import numpy
import numpy as np
import pandas as pd
from toad import quality
from binning.tree import *
from sklearn.ensemble import RandomForestClassifier
from quality.information_value import IV
from quality.information_value import WOETransform
from sklearn.tree import DecisionTreeClassifier


print("\033[91m{}\033[0m".format("This is red text"))  # 红色
print("\033[92m{}\033[0m".format("This is green text"))  # 绿色
print("\033[94m{}\033[0m".format("This is blue text"))  # 蓝色


if __name__ == '__main__':
    X = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
    Y = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    iv = IV(X, Y)

