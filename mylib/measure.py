"""
Name: Measure
Contributor: Guo Yi, Gu Chengyang
Last update: 2019/4/24
Objective: 计算一系列的评价指标
Run successfully on macOS 10.14.3, Python 3.7.2
"""

import pandas as pd
import numpy as np
from functools import wraps
from pandas.plotting import register_matplotlib_converters
from warnings import warn
from multiprocessing import Pool, cpu_count

register_matplotlib_converters()  # 为使用时间作为横轴，需要加上这一行


# this wrapper checks the target string
def check_nan(f):
    @wraps(f)
    def wrapper(df, *args, **kwargs):
        if isinstance(df, pd.DataFrame):
            if df.isnull().values.any():
                warn('DataFrame contains NaN')
        elif isinstance(df, np.ndarray):
            if np.isnan(df).any():
                warn('ndarray contains NaN')
        return f(df, *args, **kwargs)
    return wrapper


class Measure(object):

    window_size_class = 3 * 252  # 默认的窗口大小 3 年

    # basics
    @staticmethod
    @check_nan
    def get_ret(df: pd.DataFrame, method='simple') -> pd.DataFrame:
        """
        计算逐日的回报率
        这是一个静态方法，意味着不需要实例化就可以调用
        :param df: 原始价格数据矩阵
        :param method: 计算方法，'simple' 表示计算简单增长率，否则计算对数增长率
        :return: 逐日回报率矩阵
        Contributed by Guo Yi, Gu Chengyang
        """
        if method == 'simple':
            ret = (df - df.shift(1)) / df.shift(1)
        else:
            ret = np.log(df / df.shift(1))
        # ret = ((1 + ret).cumprod() - 1)[1:]  # 第一行是 NaN，删掉
        return ret

    @staticmethod
    @check_nan
    def get_annualized_ret(df: pd.DataFrame):
        """
        计算年化收益率
        :param df: index 必须是日期
        :return:
        """
        return np.log(df.iloc[-1] / df.iloc[0]) / (df.index[-1] - df.index[0]).days * 365

    # extended
    @classmethod
    def get_max_ret_freq(cls, df: pd.DataFrame, method='simple'):
        """
        寻找整个时间内的最大回报率，以及对应的交易策略名称（列名）
        这是一个类方法，意味着不需要实例化就可以调用，并且可以调用类本身的方法
        :param df: 原始价格
        :param method: 计算方法，'simple' 表示计算简单增长率，否则计算对数增长率
        :return: 存在最大收益率的交易频率以及对应的列
        Contributed by Guo Yi, modified by Gu Chengyang
        """
        ret = cls().get_ret(df, method)  # 计算逐日回报率
        aggret = ((ret.add(1)).cumprod().sub(1))[1:]  # 计算总的回报率
        max_ret = max(aggret)  # 查找最大的回报率
        max_freq = ret[max_ret]  # 查找最大回报率对应的列名
        return max_ret, max_freq

    @staticmethod
    @check_nan
    def get_mdd(y, where=False):
        """
        计算最大回撤
        如需计算滚动最大回撤，请使用 Measure.get_rolling_mdd(df, window_size)
        :param y: 价格序列
        :param where: bool 是否报告
        :return: 最大回撤以及对应 index
        Contributed by Gu Chengyang
        """
        hist_max = np.maximum.accumulate(y)  # 计算窗口期内的累计最大值
        mdd = ((hist_max - y) / hist_max).max()
        if where:
            i = np.argmax((hist_max - y) / hist_max)  # 查找最大回撤的结束时间
            j = np.argmax(y[:i])  # 查找最大回撤的开始时间
            return mdd, i, j
        else:
            return mdd

    @classmethod
    def get_lost_period(cls, df: pd.DataFrame):
        """
        计算最长损失时间
        :param df: DataFrame格式，原始价格数据
        :return: 最长损失时间
        Contributed by Guo Yi
        """

        def get_single_period(y):
            """
            计算单列的最长损失时间
            :param y: 主函数df的一列
            :return: 该列的最长损失时间
            """
            # 盈利记为1，亏损记为-1
            win = np.where(y >= 0, 1, -1)
            trace = 0
            maximum = 0

            for i in range(len(win) - 1):
                if win[i] + win[i + 1] < 0:
                    trace = trace + 1
                else:
                    if trace > maximum:
                        maximum = trace + 1
                    trace = 0
            return maximum

        return cls().get_ret(df).apply(get_single_period, raw=False)

    @classmethod
    def get_half_dev(cls, df: pd.DataFrame):
        """
        计算股价低于均值时期的方差（人们更加担心价格下跌）
        :param df:原始价格数据
        :return:方差
        Contributed by Guo Yi
        """

        def get_single_dev(ret):
            mu = ret.mean()
            temp = ret[ret < mu]
            return sum((mu - temp) ** 2) / len(ret) * 0.5

        return cls().get_ret(df).apply(get_single_dev, raw=False)

    @classmethod
    def get_VaR(cls, df: pd.DataFrame, threshold=0.05, method='historical'):
        """
        计算在险价值
        :param df: 原始价格数据
        :param threshold: VaR置信度
        :param method:默认为'historical'表示历史数据，否则为正态分布法
        :return: 给定置信度下的VaR
        Contributed by Guo Yi
        """

        def get_single_VaR(ret):
            if method == 'historical':
                return ret.quatile(threshold)
            else:
                from scipy.stats import norm
                return norm.pdf(threshold, ret.mean(), ret.std())

        return cls().get_ret(df).apply(get_single_VaR, raw=False)

    @classmethod
    def get_ES(cls, df: pd.DataFrame, threshold=0.05):
        """
        计算期望损失
        :param df: 原始价格数据
        :param threshold: 置信度
        :return: 给定置信度下的期望损失
        Contributed by Guo Yi
        """

        def get_single_ES(ret):
            return ret[ret < ret.quantile(threshold)].mean()

        return cls().get_ret(df).apply(get_single_ES, raw=False)

    @classmethod
    def get_downside_deviation(cls, y, rf=0):
        return y[y < rf].std()

    @staticmethod
    @check_nan
    def get_sharpe_ratio_from_return(y, rf=0):
        """
        根据收益率矩阵计算 sharpe ratio
        :param y: 收益率矩阵
        :param rf:
        :return:
        """
        if rf == 0:
            return np.sqrt(252) * y.mean(axis=0) / y.std(axis=0)
        else:
            return np.sqrt(252) * y.mean(axis=0).sub(rf) / y.std(axis=0)

    @classmethod
    def get_sortino_ratio(cls, y, rf=0):
        return y.mean(axis=0).sub(rf) / cls.get_downside_deviation(y, rf)

    # rolling objects
    @classmethod
    def get_rolling_mdd(cls, df: pd.DataFrame, window_size=window_size_class) -> pd.DataFrame:
        """
        计算滚动最大回撤
        如需计算整个区间的最大回撤，请使用 Measure.get_mdd(y)
        :param df: 原始价格
        :param window_size: 窗口大小
        :return: 每天窗口内的最大回撤
        Contributed by Gu Chengyang
        """

        return df \
            .rolling(window=window_size, min_periods=window_size) \
            .apply(cls().get_rolling_mdd, raw=False) \
            .dropna(how='all')  # 先滚动，再分别计算最大回撤，最后删除最前面 3 年的数据

    @staticmethod
    def get_rolling_mean_ret(ret: pd.DataFrame, window_size=15):
        """
        计算滚动收益率
        :param ret: 资产的收益率序列
        :param window_size: 滚动计算的时间窗口
        :return: pandas DataFrame，滚动收益率
        Contributed by Guo Yi
        """
        return ret.rolling(window_size).mean()

    @staticmethod
    def get_rolling_std(ret: pd.DataFrame, window_size=15):
        """
        计算滚动波动率
        :param ret: 资产的收益率序列
        :param window_size: 滚动计算的时间窗口
        :return: pandas DataFrame，滚动收益率
        Contributed by Gu Chengyang
        """
        return ret.rolling(window_size, min_periods=window_size).std()

    @classmethod
    def get_rolling_downside_deviation(cls, ret: pd.DataFrame, window_size=15):
        """
        计算滚动下行波动率
        :param ret: 资产的收益率序列
        :param window_size: 滚动计算的时间窗口
        :return: pandas DataFrame，滚动下行收益率
        Contributed by Gu Chengyang
        """
        return ret.rolling(window_size, min_periods=window_size).apply(cls().get_downside_deviation)

    @classmethod
    def get_rolling_sortino_ratio(cls, ret: pd.DataFrame, window_size=15):
        """
        计算滚动下行波动率
        :param ret: 资产的收益率序列
        :param window_size: 滚动计算的时间窗口
        :return: pandas DataFrame，滚动下行收益率
        Contributed by Gu Chengyang
        """
        return ret.rolling(window_size, min_periods=window_size).apply(cls().get_sortino_ratio)

    @staticmethod
    def get_rolling_cov(ret: pd.DataFrame, window=15):
        """
        计算滚动协方差矩阵
        :param ret: 资产的收益率序列
        :param window: 滚动计算的时间窗口
        :return: numpy.array，滚动协方差矩阵
        Contributed by Guo Yi
        """
        noa = ret.shape[1]
        cov_ret = ret.rolling(window).cov().dropna()
        # 改变数据维度，以列表形式输出
        return cov_ret.values.reshape(cov_ret.shape[0] // noa, noa, noa)

    @classmethod
    def get_rolling_calmar_ratio(cls, df: pd.DataFrame, method='simple', window_size=window_size_class) -> pd.DataFrame:
        """
        计算滚动 calmar ratio
        :param df: 价格数据，若有多个维度，则按列计算
        :param method: 计算方法，'simple' 表示计算简单增长率，否则计算对数增长率
        :param window_size: 窗口大小
        :return: 每天窗口内的 calmar ratio
        Contributed by Gu Chengyang
        """

        ret = cls().get_ret(df, method)

        def my_gmean(y) -> float:
            """
            计算几何平均
            :param y: 每个滚动周期内的价格
            :return: 几何平均值
            """
            return np.sqrt(252) * ((y + 1).prod() ** (1 / window_size) - 1)  # 年化

        rolling_geo_mean_ret = ret.rolling(window=window_size).apply(my_gmean, raw=False)  # 滚动几何平均
        rolling_max_drawdown = cls.get_rolling_mdd(df, window_size)  # 最大回撤
        return (rolling_geo_mean_ret / rolling_max_drawdown).dropna(how='all')

    @classmethod
    def get_rolling_sharpe_ratio_alter(cls, df: pd.DataFrame, method='simple', rf=0,
                                       window_size=window_size_class) -> pd.DataFrame:
        """
        计算滚动 sharpe ratio
        这个方法比较慢，将在近期移除
        请使用 correlation_parallel
        :param df: 价格数据，若有多个维度，则按列计算
        :param method: 计算方法，'simple' 表示计算简单增长率，否则计算对数增长率
        :param rf: 基准回报率
        :param window_size: 滚动窗口大小
        :return: 滚动 sharpe ratio，每天、每种策略对应一个值
        Contributed by Gu Chengyang
        """
        return cls().get_ret(df, method) \
            .rolling(window=window_size, min_periods=window_size) \
            .apply(cls().get_sharpe_ratio_from_return, raw=False) \
            .dropna(how='all')  # 先滚动，再分别计算 sharpe ratio，最后删除最前面 3 年的数据

    @classmethod
    def get_rolling_sharpe_ratio_parallel(cls, df: pd.DataFrame, method='simple', rf=0,
                                          window_size=window_size_class) -> pd.DataFrame:
        """
        计算滚动 sharpe ratio
        使用了多线程技术，速度比 rolling 对象更快。
        注意，第 T 天的 sharpe ratio 使用了包含该天的信息，若要在 T+1 日进行预测，请使用截止到 T 日的结果
        :param df: 价格数据，若有多个维度，则按列计算。传入时请注意用两对中括号，表示是一个 DataFrame
        :param method: 计算方法，'simple' 表示计算简单增长率，否则计算对数增长率
        :param rf: 基准回报率
        :param window_size: 滚动窗口大小
        :return: 滚动 sharpe ratio，每天、每种策略对应一个值
        Contributed by Gu Chengyang
        """
        df = cls().get_ret(df, method)
        df_list = [df.iloc[i: i+window_size, :] for i in range(df.shape[0] - window_size + 1)]

        pool = Pool(processes=cpu_count()-1)
        results = pool.map(cls().get_sharpe_ratio_from_return, df_list)
        pool.close()
        pool.join()

        results = pd.concat(results, axis=1).T
        results.index = df.index[window_size - 1:]

        return results  # 先滚动，再分别计算 sharpe ratio，最后删除最前面 3 年的数据
