"""
Name: Brewery
Contributor: Gu Chengyang
Date: 2019/3/11
Objective: 一些基础参数和粗筛方法的集合
Run successfully on macOS 10.14.3, Python 3.7.2
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from mylib.measure import Measure

register_matplotlib_converters()  # 为使用时间作为横轴，需要加上这一行


class Brewery(object):
    @staticmethod
    def get_sharpe_ratio(df: pd.DataFrame, window_size) -> pd.DataFrame:
        return Measure.get_rolling_sharpe_ratio_alter(df, window_size=window_size)