# -*- coding: utf-8 -*-
# Author: lkm
# date: 2023/8/23 23:35

from sympy import symbols, log, latex

# 创建WOE符号
WOE = symbols('WOE')

# 构建WOE公式
woe_expression = log(symbols('good_count') / symbols('bad_count')) - log(symbols('bad_count') / symbols('good_count'))

# 打印WOE公式的Latex形式
woe_latex = latex(woe_expression)
print("WOE Formula (Latex):", woe_latex)
