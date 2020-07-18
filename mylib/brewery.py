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

    @staticmethod
    def get_calmar_ratio(df: pd.DataFrame, window_size) -> pd.DataFrame:
        return Measure.get_rolling_calmar_ratio(df, window_size=window_size)

    @staticmethod
    def get_information_ratio(df):
        pass

    @staticmethod
    def drop_high_volatility(df: pd.DataFrame, threshold=2) -> pd.DataFrame:
        """
        清除波动率过大的数据
        :param df:
        :param threshold: 波动率超过几个标准差
        :return: 原始数据除去了波动率过大的对应列
        Contributed by Gu Chengyang
        """
        return df[df.columns[df.min() < df.mean() - threshold * df.std()]]

    @classmethod
    def drop_low_sharpe_ratio(cls, df: pd.DataFrame, threshold=0, scale=0.5, window_size=3 * 21) -> pd.DataFrame:
        """
        清除 sharpe ratio 过低的数据
        首先计算 sharpe ratio，然后判断回报低于 threshold 的比例是否高于 scale，如果是，说明该策略表现较差
        :param df:
        :param threshold: 判断门槛
        :param scale: 低于门槛的比例
        :param window_size: 窗口大小
        :return: 原始数据除去 sharpe ratio 过低的对应列
        Contributed by Gu Chengyang
        """
        mask = (cls().get_sharpe_ratio(df, window_size).ge(threshold)).sum() > df.shape[0] * scale
        return df[df.columns[mask]]

    @classmethod
    def drop_low_calmar_ratio(cls, df: pd.DataFrame, threshold=0, scale=0.5, window_size=3 * 21) -> pd.DataFrame:
        """
        清除 calmar ratio 过低的数据
        首先计算 calmar ratio，然后判断回报低于 threshold 的比例是否高于 scale，如果是，说明该策略表现较差
        :param df:
        :param threshold: 判断门槛
        :param scale: 低于门槛的比例
        :param window_size: 窗口大小
        :return: 原始数据除去 calmar ratio 过低的对应列
        Contributed by Gu Chengyang
        """
        mask = (cls().get_calmar_ratio(df, window_size).ge(threshold)).sum() > df.shape[0] * scale
        return df[df.columns[mask]]

    @staticmethod
    def plot_trade_result(price: pd.DataFrame, **kwargs) -> None:
        """
        根据权重和实际价格绘制动态权益折线图
        :param price: 价格数据
        :param weight: 投资权重
        :return: 没有返回值
        Contributed by Gu Chengyang
        """
        # 计算份额价格
        if 'weight' in kwargs:
            weight = kwargs.get('weight')
        else:
            weight = pd.DataFrame().reindex_like(price).fillna(1. / price.shape[1])

        share_value = (price * weight).sum(axis=1)
        share_value = share_value[share_value != 0]

        mdd, mdd_start, mdd_end = Measure.get_mdd(share_value, where=True)

        # 绘制折线图
        fig, ax = plt.subplots()
        ax.plot(share_value)
        plt.axhline(y=share_value[mdd_start])
        plt.axhline(y=share_value[mdd_end])
        fig.autofmt_xdate()

        # 设置时间
        from datetime import datetime
        startpoint = datetime.fromtimestamp(datetime.timestamp(weight.first_valid_index()))
        endpoint = datetime.fromtimestamp(datetime.timestamp(weight.last_valid_index()))
        ax.set_xlim([startpoint, endpoint])

        # 显示图表
        plt.show()
