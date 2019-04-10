"""
Name: Back test engine
Contributor: Gu Chengyang
Date: 2019/4/10
Objective: 基于用户指定策略的回测
Run successfully on macOS 10.14.4, Python 3.7.2

Directions: 用户在调用的时候，
首先传入价格数据 price: pd.Dataframe，
然后重载 self.strategy() 函数，
最后执行 self.run()。
用户需要注意的是，self.strategy() 函数每天都会被重新调用，且不会保留该日之前存储的变量。
如您的策略需要涉及与该日之前的数据有关的结果，请将其设置为实例属性（即：使用 self.your_attribute）。
对于不熟悉语法的用户，凡名称前面有双下划线的属性和方法，均为私有方法。请调用不带下划线的属性和方法。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from warnings import warn
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum
from math import isnan

from mylib.measure import Measure

# 为使用时间作为横轴，该行代码在未来的 matplotlib 中将作为必须
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# 避免 chain index 提高性能，详见
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy


def security_check(f):
    """
    检查传入资产名称是否合法，以及价格是否为 NaN 的装饰器
    :param f:
    :return:
    """

    @wraps(f)
    def wrapper(self, security, *args, **kwargs):
        if isinstance(security, str):
            if security not in self.securities:
                raise TypeError(f'{security}: security not defined')
            if isnan(self.price.loc[self.today, security]):
                raise ValueError(f'price for security {security} on {self.today} is NaN')

        elif isinstance(security, list):
            pass

        else:
            raise TypeError('security unrecognized')

        f(self, security, *args, **kwargs)

    return wrapper


class FuturesMultiplier(object):
    # 期货乘数的枚举类型
    # http://www.qihuokaihu.net/html/201808-14/20180814134155.htm
    class Shanghai(Enum):
        pass

    class Zhenzhou(Enum):
        pass

    class Dalian(Enum):
        pass

    class CFFEX(Enum):
        IF300 = 300
        IF = 300
        IH50 = 300
        IH = 300
        IC500 = 200
        IC = 200
        ETF = 10000
        CFFEX = 10000

    class SHOil(Enum):
        pass


class BackTestFramework(object):
    # debug 模式，打印每笔交易明细
    print_debugging = False

    # 详细的 debug 模式，打印每个迭代到的日期，需要先启动 debug 模式
    print_debugging_detail = False

    def __init__(self, **kwargs):

        if 'price' not in kwargs:
            raise Exception('No price reference designated')
        else:
            self.price = kwargs.get('price')

    # securities
    @property
    def securities(self):
        return self._securities

    # stock_list
    @property
    def stock_list(self):
        return self._stock_list

    # futures_list
    @property
    def futures_list(self):
        return self._futures_list

    # benchmark
    @property
    def benchmark(self):
        return self._benchmark

    # factors
    @property
    def factors(self):
        return self._factors
    
    # start_date
    @property
    def start_date(self):
        return self._start_date

    # end_date
    @property
    def end_date(self):
        return self._end_date

    # 税费
    @property
    def commission(self):
        return self._commission

    # 滑点
    @property
    def slippage(self):
        return self._slippage

    # 初始现金
    @property
    def cash(self):
        return self._cash

    # 价格矩阵
    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise ValueError('price must be pandas DataFrame object')
        elif not isinstance(value.index, pd.DatetimeIndex):
            raise ValueError('index of price must be DatetimeIndex')
        else:
            self.__price = value

    # 日志，只读
    @property
    def log(self):
        return self._log

    # 每日持仓量，只读
    @property
    def position(self):
        return self._position

    # 当前持仓量，只读
    @property
    def current_position(self):
        return self._current_position

    # 每日价值，只读
    @property
    def value(self):
        return self._value

    # 因手续费支出的现金，只读
    @property
    def cash_spent_on_commission(self):
        return self._cash_spent_on_commission

    # 今日日期，只读
    @property
    def today(self):
        if self._today is None:
            warn('no today available, None is returned')
        return self._today

    # 昨日日期，只读
    @property
    def yesterday(self):
        if self._yesterday is None:
            warn('no yesterday available, None is returned')
        return self._yesterday

    def _logger(self, security, volume_change):
        """
        记录日志的函数
        :param security: 交易资产名类
        :param volume_change: 交易数量
        :return: no return
        """
        pass

    def _trade_classifier(self, f, security, *args, **kwargs):
        """
        区分传入的是单个资产名称还是一个 list，然后分别进行交易
        :param f: 交易调用的函数
        :param security: 传入资产
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def _order_single(self, security, volume_change):
        """
        单个资产下单函数，这个函数是主交易函数，所有的交易都需要它来完成，交易日志的记录也由它调用
        :param security: 交易资产名类，单个
        :param volume_change: 交易数量
        :return: no return
        """
        pass

    @security_check
    def buy(self, security: str or list, volume):
        """
        下单加仓函数
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param volume: 交易数量
        :return: no return
        """
        pass

    @security_check
    def sell(self, security: str or list, volume):
        """
        下单减仓函数
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param volume: 交易数量
        :return: no return
        """
        pass

    def _adjust_to_single(self, security, new_position):
        pass

    @security_check
    def adjust_to(self, security: str or list, new_position):
        """
        将某资产的仓位直接调整到所需的大小
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param new_position: 持仓目标数量
        :return:
        """
        pass

    def _adjust_to_portion_of_value_single(self, security, portion_of_value):
        pass

    @security_check
    def adjust_to_portion_of_value(self, security: str or list, portion_of_value):
        """
        将某资产的仓位直接调整到组合的价值比例
        例如，portion_of_value = 0.2 意味着把仓位调整到资产价值总的20%，注意受到剩余现金的约束
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param portion_of_value: 持仓目标与目前的比例
        :return:
        """
        pass

    def _adjust_to_by_portion_single(self, security, portion, otherwise_position):
        pass

    @security_check
    def adjust_to_by_portion(self, security: str or list, portion, otherwise_position=0):
        """
        将某资产的仓位按现在的比例调整到所需的大小
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param portion: 持仓目标与目前的比例
        :param otherwise_position: 如果现在持仓量为 0，可以同时调整到一个需要的仓位
        :return:
        """
        pass

    def _initialize(self):
        """
        初始化函数，根据用户指定的资产类型生成空的持仓量矩阵
        :return: no return
        """
        pass

    def _finalize(self):
        """
        最后处理函数，生成每日动态价值的矩阵
        :return:
        """
        pass

    @staticmethod
    def _date_range(start_date, end_date):
        """
        this generator allows iteration over days, skipping the first day
        :param start_date:
        :param end_date:
        :return:
        """
        for n in range(1, int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def strategy(self):
        """
        用户指定的策略函数，每天执行一次
        :return: no return
        """
        # this strategy should implement every day
        warn('Sample strategy function is called')
        for security in self.securities:
            try:
                self.buy(security, 1)
            except ValueError:
                # 没有现金了，或者碰到了 NaN 等，都不交易
                pass

    def run(self):
        """
        启动
        :return: no return
        """
        self._initialize()

        for single_date in self._date_range(self.start_date, self.end_date):
            if single_date in self.price.index:
                pass

        self._finalize()

    def summary(self, save_pic=False, pic_name='image.png'):
        """
        汇报资产组合的最大回撤、收益率、sharpe ratio，对价值作图等
        :return:
        """
        pass