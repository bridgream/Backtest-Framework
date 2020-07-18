"""
Name: Back test engine
Contributor: Gu Chengyang
Last update: 2019/4/24
Objective: 基于用户指定策略的回测
Run successfully on macOS 10.14.4, Python 3.7.3
Requirements: Python >= 3.6, Pandas >= 0.24.0

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
import matplotlib.dates
import seaborn as sns
from warnings import warn
from functools import wraps
from datetime import datetime
from math import isnan

from mylib.measure import Measure

# 为使用时间作为横轴，该行代码在未来的 matplotlib 中将作为必须
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def security_check_condition(enable_security_check):
    def security_check(f):
        """
        检查传入资产名称是否合法，以及价格是否为 NaN 的装饰器
        :param f:
        :return:
        """

        @wraps(f)
        def wrapper(self, security, *args, **kwargs):
            if enable_security_check:
                if isinstance(security, str):
                    if security not in self.securities:
                        raise TypeError(f'{security}: security not defined')
                    if isnan(self.price.loc[self.today, security]):
                        raise ValueError(f'price for security {security} on {self.today} is NaN')
                elif isinstance(security, list):
                    for s in security:
                        if not isinstance(s, str):
                            raise TypeError(f'security unrecognized: {s}')
                        if s not in self.securities:
                            raise TypeError(f'{s}: security not defined')
                        if isnan(self.price.loc[self.today, s]):
                            raise ValueError(f'price for security {s} on {self.today} is NaN')
                else:
                    raise TypeError('security unrecognized')
            return f(self, security, *args, **kwargs)

        return wrapper

    return security_check


class BackTestFramework(object):
    # debug 模式，打印每笔交易明细
    print_debugging = False
    # 详细的 debug 模式，打印每个迭代到的日期，需要先启动 debug 模式
    print_debugging_detail = False
    enable_security_check = True
    disable_short_sell = False
    _LIMIT_MOVE_THRESHOLD = 0.094

    multiplier_dict = {
        # http://www.qihuokaihu.net/html/201808-14/20180814134155.htm
        'stock': 100,

        'CU.SHF': 5,
        'RB.SHF': 10,
        'ZN.SHF': 5,
        'AL.SHF': 5,
        'AU.SHF': 1000,
        'WR.SHF': 10,
        'FU.SHF': 10,
        'RU.SHF': 10,
        'PB.SHF': 5,
        'AG.SHF': 15,
        'BU.SHF': 10,
        'HC.SHF': 10,
        'NI.SHF': 1,
        'SN.SHF': 1,

        'PTA.CZC': 5,
        'SR.CZC': 10,
        'CF.CZC': 5,
        'CY.CZC': 5,
        'WH.CZC': 20,
        'OI.CZC': 10,
        'RI.CZC': 20,
        'FG.CZC': 20,
        'RM.CZC': 10,
        'RS.CZC': 10,
        'JR.CZC': 20,
        'LR.CZC': 20,
        'SF.CZC': 5,
        'MA.CZC': 10,
        'AP.CZC': 10,
        'ZC.CZC': 100,

        'P.DCE': 10,
        'L.DCE': 5,
        'PVC.DCE': 5,
        'B.DCE': 10,
        'A.DCE': 10,
        'M.DCE': 10,
        'Y.DCE': 10,
        'C.DCE': 10,
        'J.DCE': 100,
        'JM.DCE': 60,
        'I.DCE': 100,
        'JD.DCE': 5,
        'BB.DCE': 500,
        'FB.DCE': 500,
        'PP.DCE': 5,
        'CS.DCE': 10,

        'IF.CFFEX': 300,
        'IH.CFFEX': 300,
        'IC.CFFEX': 200,
    }

    def __init__(self, **kwargs):

        if 'price' not in kwargs:
            raise Exception('No price reference designated')
        else:
            self.price = kwargs.get('price')

        self._start_date = None
        self._end_date = None

        # parameters
        if 'start_date' in kwargs:
            self.start_date = kwargs.get('start_date')
        else:
            try:
                self.start_date = self.price.index[0]
                warn(f'No start_date designated, using first date in price instead: {self.start_date}')
            except IndexError:
                warn('No start_date designated, default to None')

        if 'end_date' in kwargs:
            self.end_date = kwargs.get('end_date')
        else:
            try:
                self.end_date = self.price.index[-1]
                warn(f'No end_date designated, using last date in price instead: {self.end_date}')
            except IndexError:
                warn('No end_date designated, default to None')

        # 资产类型，基准，因子
        self.securities = kwargs.get('securities', self.price.columns.values.tolist())
        # by default, we take every security as stock
        self.stock_list = kwargs.get('stock_list', self.securities)
        self.futures_list = kwargs.get('futures_list', [])

        if 'benchmark' in kwargs:
            self.benchmark = kwargs.get('benchmark')
        else:
            warn('No benchmark designated, default to None')
            self._benchmark = None

        self.pct_change = kwargs.get('pct_change', None)

        self.factors = kwargs.get('factors', [])

        # 手续费
        self.commission = kwargs.get('commission', 0.)

        # 滑点
        self.slippage = kwargs.get('slippage', 0.01)

        # 初始现金
        self.cash = self._initial_cash = kwargs.get('cash', 1e8)

        # 下面为初始化部分
        # 日志
        self._log_dict = None
        self._log_index = None
        self._log = None

        # 持仓量
        self._position = None

        # 随时更新的持仓量，单位是手
        self._current_position = None

        # 价值
        self._value = None
        self._current_value = self._initial_cash
        self._cash_spent_on_commission = None

        # 滚动的日期
        self._today = None
        # self._today_index = None
        self._last_trading_day = None

        # 乘数
        self._multiplier = None
        self._pct_change = None

        # 参考数据
        self.info = None

    # 价格矩阵
    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise ValueError('price must be Pandas DataFrame object')
        elif not isinstance(value.index, pd.DatetimeIndex):
            raise ValueError('index of price must be DatetimeIndex')
        else:
            self._price = value

    # start_date
    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, value):
        if not isinstance(value, datetime):
            raise ValueError('start_date must be datetime')
        elif self._start_date is not None and self.price.index[0] >= value:
            raise ValueError('start_date out of range of price')
        self._start_date = value

    # end_date
    @property
    def end_date(self):
        return self._end_date

    @end_date.setter
    def end_date(self, value):
        if not isinstance(value, datetime):
            raise ValueError('end_date must be datetime')
        elif self._end_date is not None and self.price.index[-1] <= value:
            raise ValueError('end_date out of range of price')
        self._end_date = value

    # securities
    @property
    def securities(self):
        return self._securities

    @securities.setter
    def securities(self, value: list):
        if not isinstance(value, list):
            raise TypeError('securities must be list of strings')
        for s in value:
            if not isinstance(s, str):
                raise TypeError('securities must be list of strings')
            if s not in self.price.columns.values:
                raise TypeError(f'security {s} not in price matrix')
        self._securities = value

    # stock_list
    @property
    def stock_list(self):
        return self._stock_list

    @stock_list.setter
    def stock_list(self, value: list):
        if not isinstance(value, list):
            raise TypeError('stock_list must be list of strings')
        for s in value:
            if not isinstance(s, str):
                raise TypeError('stock_list must be list of strings')
            if s not in self.securities:
                raise TypeError(f'stock {s} not in securities')
        self._stock_list = value

    # futures_list
    @property
    def futures_list(self):
        return self._futures_list

    @futures_list.setter
    def futures_list(self, value: list):
        if not isinstance(value, list):
            raise TypeError('futures_list must be list of strings')
        for s in value:
            if not isinstance(s, str):
                raise TypeError('futures_list must be list of strings')
            if s not in self.securities:
                raise TypeError(f'futures {s} not in securities')
        self._futures_list = value

    # benchmark
    @property
    def benchmark(self):
        return self._benchmark

    @benchmark.setter
    def benchmark(self, value):
        if value is None:
            self._benchmark = None
        elif isinstance(value, str):
            if value in self.securities:
                self._benchmark = self.price[value]
            else:
                raise TypeError(f'benchmark \'{self.benchmark}\' not in securities')
        elif isinstance(value, pd.Series):
            self._benchmark = value.loc[self.start_date:self.end_date]
        else:
            warn('benchmark must be string or pd.Series, default to None')

    # 变动率矩阵
    @property
    def pct_change(self):
        return self._pct_change

    @pct_change.setter
    def pct_change(self, value: pd.DataFrame):
        if value is None:
            self._benchmark = None
        elif not isinstance(value, pd.DataFrame):
            raise ValueError('pct_change must be Pandas DataFrame object')
        elif not isinstance(value.index, pd.DatetimeIndex):
            raise ValueError('index of pct_change must be DatetimeIndex')
        else:
            self._pct_change = value

    # factors
    @property
    def factors(self):
        return self._factors

    @factors.setter
    def factors(self, value: list):
        if not isinstance(value, list):
            raise TypeError('factors must be list of strings')
        for s in value:
            if not isinstance(s, str):
                raise TypeError('factors must be list of strings')
        self._factors = value

    # 税费
    @property
    def commission(self):
        return self._commission

    @commission.setter
    def commission(self, value):
        if not 0 <= value < 1:
            raise ValueError('commission must between 0. - 1.')
        self._commission = value

    # 滑点
    @property
    def slippage(self):
        return self._slippage

    @slippage.setter
    def slippage(self, value):
        if not 0 <= value < 1:
            raise ValueError('slippage must between 0. - 1.')
        self._slippage = value

    # 初始现金
    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self, value):
        if value <= 0:
            raise ValueError(f'cash negative caught: {self.cash}')
        self._cash = value

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

    # 当前价值，只读
    @property
    def current_value(self):
        return self._current_value

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

    # 上个交易日，只读
    @property
    def last_trading_day(self):
        if self._last_trading_day is None:
            warn('no last_trading_day available, None is returned')
        return self._last_trading_day

    def _logger(self, security, volume_change):
        """
        记录日志的函数
        :param security: 交易资产名类
        :param volume_change: 交易数量
        :return: no return
        """
        multiplier = self.multiplier_dict[security] if security in self.futures_list else self.multiplier_dict['stock']
        current_price = self.price.loc[self.today, security]
        cost = round(volume_change * current_price * (1 + self.commission) * multiplier, 2)
        self._cash_spent_on_commission += round(volume_change * current_price * self.commission * multiplier, 2)
        self.cash -= cost
        self._current_position['cash'] = self.cash
        self.current_position[security] += volume_change
        self._current_value = np.nansum(self.current_position[self.securities] * self.price.loc[self.today] * self._multiplier) + self.cash
        if self.print_debugging:
            print(
                f'On {self.today.date()}, successfully ordered {volume_change * multiplier} of {security} at ¥{current_price:.2f}')
            print(f'Money spent: ¥{cost}, remaining cash: ¥{self.cash:.2f}')
            print(f'Total value: ¥{self._current_value:.2f}\n')

        self._log_dict[self._log_index] = ({
            'date': self.today,
            'security': security,
            'volume_change': volume_change * multiplier,
            'price': current_price,
            'current_position': self._current_position[security] * multiplier,
            'remaining_cash': self.cash,
            'current_value': self._current_value,
        })

        # 判断交易时是否可能涨、跌停
        if self.pct_change is not None:
            daily_pct_change = self._log_dict[self._log_index]['daily_pct_change'] = self.pct_change.loc[
                self.today, security]
            if isnan(daily_pct_change):
                self._log_dict[self._log_index]['possible_hit_limit'] = None
            elif daily_pct_change > self._LIMIT_MOVE_THRESHOLD:
                self._log_dict[self._log_index]['possible_hit_limit'] = 1
            elif daily_pct_change < -self._LIMIT_MOVE_THRESHOLD:
                self._log_dict[self._log_index]['possible_hit_limit'] = -1
            else:
                self._log_dict[self._log_index]['possible_hit_limit'] = 0

        self._log_index += 1

    @staticmethod
    def _trade_classifier(f, security, *args, **kwargs):
        """
        区分传入的是单个资产名称还是一个 list，然后分别进行交易
        :param f: 交易调用的函数
        :param security: 传入资产
        :param args:
        :param kwargs:
        :return:
        """
        if isinstance(security, str):
            # single security
            f(security, *args, **kwargs)
        else:
            # a list of securities
            for s in security:
                f(s, *args, **kwargs)

    def _order_single(self, security, volume_change):
        """
        单个资产下单函数，这个函数是主交易函数，所有的交易都需要它来完成，交易日志的记录也由它调用
        :param security: 交易资产名类，单个
        :param volume_change: 交易数量
        :return: no return
        """
        if self.pct_change is not None:
            daily_pct_change = self.pct_change.loc[self.today, security]
            if isnan(daily_pct_change):
                warn(f'pct_change is defined but NaN on {self.today} for {security}')
            elif abs(daily_pct_change) > self._LIMIT_MOVE_THRESHOLD:
                warn(f'Possible limit move detected on {self.today} for {security}. Please modify your strategy() function. ')

        volume_change = int(volume_change)
        if volume_change != 0:
            if security in self.stock_list:
                if self.print_debugging and self.print_debugging_detail:
                    print(f'Transaction for {security} in progress...')
                if self.disable_short_sell and self._current_position[security] + volume_change < 0:
                    warn(f'Overselling, will sell all {security}')
                    self.adjust_to(security, 0)
                else:
                    self._logger(security, volume_change)
            elif security in self.futures_list:
                if self.print_debugging and self.print_debugging_detail:
                    print(f'Transaction for {security} in progress...')
                self._logger(security, volume_change)

    @security_check_condition(enable_security_check)
    def buy(self, security: str or list, volume):
        """
        下单加仓函数
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param volume: 交易数量（手），例如股票一手为 100 股
        :return: no return
        """
        self._trade_classifier(self._order_single, security, volume)

    @security_check_condition(enable_security_check)
    def sell(self, security: str or list, volume):
        """
        下单减仓函数
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param volume: 交易数量（手），例如股票一手为 100 股
        :return: no return
        """
        self._trade_classifier(self._order_single, security, -volume)

    def _adjust_to_single(self, security, new_position):
        old_position = self._current_position[security]
        if old_position != new_position:
            # 如果前后仓位不同才需要调整
            volume = new_position - old_position
            self._order_single(security, volume)

    @security_check_condition(enable_security_check)
    def adjust_to(self, security: str or list, new_position):
        """
        将某资产的仓位直接调整到所需的大小
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param new_position: 持仓目标数量（手），例如股票一手为 100 股
        :return:
        """
        self._trade_classifier(self._adjust_to_single, security, new_position)

    def _adjust_to_value_single(self, security, new_value):
        multiplier = self.multiplier_dict[security] if security in self.futures_list else self.multiplier_dict['stock']
        self._adjust_to_single(security, new_value / self.price.loc[self.today, security] / multiplier)

    @security_check_condition(enable_security_check)
    def adjust_to_value(self, security: str or list, new_value):
        """
        将某资产的仓位直接调整到对应的价值
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param new_value: 新的目标价值
        :return:
        """
        self._trade_classifier(self._adjust_to_value_single, security, new_value)

    def _adjust_to_portion_of_value_single(self, security, portion_of_value):
        multiplier = self.multiplier_dict[security] if security in self.futures_list else self.multiplier_dict['stock']
        old_position = self._current_position[security]
        new_position = portion_of_value * self._current_value / self.price.loc[self.today, security] / multiplier
        if old_position != new_position:
            # 如果前后仓位不同才需要调整
            volume = new_position - old_position
            self._order_single(security, volume)

    @security_check_condition(enable_security_check)
    def adjust_to_portion_of_value(self, security: str or list, portion_of_value):
        """
        将某资产的仓位直接调整到组合的价值比例
        例如，portion_of_value = 0.2 意味着把仓位调整到资产价值总的20%，注意受到剩余现金的约束
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param portion_of_value: 持仓目标与目前的比例
        :return:
        """
        self._current_value = np.nansum(self.current_position[self.securities] * self.price.loc[self.today] * self._multiplier) + self.cash
        self._trade_classifier(self._adjust_to_portion_of_value_single, security, portion_of_value)

    def _adjust_to_by_portion_single(self, security, portion, otherwise_position):
        old_position = self._current_position[security]
        if old_position == 0:
            if otherwise_position != 0:
                # 如果原来的仓位是 0，则直接调整到指定的一个仓位
                self._order_single(security, otherwise_position)
        else:
            volume = old_position * (portion - 1)
            self._order_single(security, volume)

    @security_check_condition(enable_security_check)
    def adjust_to_by_portion(self, security: str or list, portion, otherwise_position=0):
        """
        将某资产的仓位按现在的比例调整到所需的大小
        :param security: 交易资产名类，可以是单个的名称（字符串），也可以是 list
        :param portion: 持仓目标与目前的比例
        :param otherwise_position: 如果现在持仓量为 0，可以同时调整到一个需要的仓位
        :return:
        """
        self._trade_classifier(self._adjust_to_by_portion_single, security, portion, otherwise_position)

    def _initialize(self):
        """
        初始化函数，根据用户指定的资产类型生成空的持仓量矩阵
        :return: no return
        """
        # initialize logger
        self._log = pd.DataFrame(columns=[
            'date', 'security', 'volume_change', 'price', 'current_position', 'remaining_cash'
        ])
        self._log_dict = dict()
        self._log_index = 0
        self.info = dict()

        # initialize date
        self._today = self.start_date
        self._last_trading_day = None

        # initialize position
        self._current_position = pd.Series(index=['date', *self.securities, 'cash'])
        self._current_position['date'] = self.start_date
        self._current_position['cash'] = self.cash
        self._current_position.fillna(value=0, inplace=True)

        # initialize cash and value
        self._current_value = self.cash = self._initial_cash
        self._cash_spent_on_commission = 0
        self._value = pd.DataFrame(columns=['date', *self.securities, 'turnover'])
        self._position = pd.DataFrame(columns=['date', *self.securities, 'cash'],
                                      index=self.price.loc[self.start_date: self.end_date].index)

        # initialize multiplier
        def find_multiplier(k):
            if k in self.stock_list:
                return self.multiplier_dict['stock']
            else:
                return self.multiplier_dict[k]

        self._multiplier = pd.Series({k: find_multiplier(k) for k in self.securities})

    def _finalize(self):
        """
        最后处理函数，生成每日动态价值的矩阵
        :return:
        """
        if len(self._log_dict) is 0:
            raise NotImplementedError('No effective trade implemented. ')
        self._log = pd.DataFrame.from_dict(self._log_dict, 'index')
        self._log_dict.clear()
        self._position.set_index('date', inplace=True)
        self._value = self._position.iloc[:, :-1] * self.price[self.price.index.isin(self._position.index)] * self._multiplier
        # 换手率
        self._value['turnover'] = abs(self._value - self._value.shift()).sum(axis=1) / abs(self._value.shift()).sum(axis=1)
        # 每天持仓的股票价值
        self._value['cash'] = self.position['cash']
        self._value['total_value'] = self._value[['cash', *self.securities]].sum(axis=1)
        # 将手数转换为数量
        self._position *= [*self._multiplier, 1]
        self.info['mdd'] = Measure.get_mdd(self.value['total_value'])
        self.info['annualized_ret'] = Measure.get_annualized_ret(self.value['total_value'])
        self.info['sharpe_ratio'] = Measure.get_sharpe_ratio_from_return(Measure.get_ret(self.value['total_value']))

    def strategy(self):
        """
        用户指定的策略函数，每天执行一次
        :return: no return
        """
        # this strategy should implement every day
        warn('Sample strategy function is called')
        # for security in self.securities:
        #     try:
        #         self.buy(security, 1)
        #     except ValueError:
        #         # 没有现金了，或者碰到了 NaN 等，都不交易
        #         pass

    def user_initialize(self):
        """
        用户指定的首日执行策略
        :return: no return
        """
        pass

    def user_finalize(self):
        """
        用户指定的最后一日执行策略
        :return: no return
        """
        pass

    def run(self):
        """
        启动
        :return: no return
        """
        print('Back test initiating...')
        self._initialize()
        self.user_initialize()

        print('Back test running...')
        for single_date in self.price.loc[self.start_date: self.end_date].index:
            if self.print_debugging and self.print_debugging_detail:
                print(single_date)
            self._last_trading_day = self._today
            self._today = single_date
            self._current_position['date'] = single_date
            self.strategy()
            self._position.loc[single_date, :] = self._current_position
        print(f'A total of {len(self._log_dict)} trades have been implemented. ')

        print('Back test finalizing...')
        self.user_finalize()
        self._finalize()
        print()

    def _get_annual_YTM(self):
        YTM_by_year = list(range(self.start_date.year, self.end_date.year + 1))
        YTM_by_year[0] = self.value[self.value.index.year == self.start_date.year].iloc[-1, :]['total_value'] / self._initial_cash - 1
        for i, rolling_year in enumerate(range(self.start_date.year + 1, self.end_date.year + 1), 1):
            YTM_by_year[i] = self.value[self.value.index.year == rolling_year].iloc[-1, :]['total_value'] / self.value[self.value.index.year == rolling_year - 1].iloc[-1, :]['total_value'] - 1
        return YTM_by_year

    def summary(self, save_pic=False, pic_name='image.png'):
        """
        汇报资产组合的最大回撤、收益率、sharpe ratio，对价值作图等
        :return:
        """
        print('Plotting graph and generating statistics... ')
        df = self.value['total_value']
        if self.benchmark is not None:
            benchmark = self.benchmark * self._initial_cash / self.benchmark[0]
            benchmark.name = 'benchmark'

            ax = sns.lineplot(data=[df, benchmark])

        else:
            ax = sns.lineplot(data=[df])

        # format the ticks and grid
        ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
        ax.set_xlim(self.start_date, self.end_date)
        plt.title('Dynamic value of portfolio')
        plt.gcf().autofmt_xdate()
        ax.grid(True)
        if save_pic:
            plt.savefig(pic_name)
        else:
            plt.show()
        plt.close()

        # 打印总的 MDD、YTM、SR
        print(f'MDD: {self.info["mdd"] * 100:.1f}%')
        print(f'YTM: {self.info["annualized_ret"] * 100:.2f}%')
        print(f'SR: {self.info["sharpe_ratio"]:.2f}')
        print()

        # 打印逐年的YTM
        YTM_by_year = self._get_annual_YTM()
        print('YTM by year: ')
        print('Year   YTM')
        for i, rolling_year in enumerate(range(self.start_date.year, self.end_date.year + 1)):
            print(f'{rolling_year} {YTM_by_year[i] * 100:7.2f}%')
        print()

    def plot_rolling_sr(self, window_size=93):
        """
        作出滚动 Sharpe Ratio 曲线
        :param window_size:
        :return:
        """
        print('Plotting rolling sharpe ratio... ')

        rolling_sr = Measure.get_rolling_sharpe_ratio_parallel(self.value[['total_value']], window_size=window_size)
        sns.lineplot(data=rolling_sr)
        plt.title('Rolling Sharpe ratio')
        plt.axhline(y=1)
        plt.show()
        plt.close()

    def log_to_excel(self, file_name='result.xlsx'):
        with pd.ExcelWriter(file_name) as writer:
            self.log.to_excel(writer, sheet_name='trade_log')
            self.position.to_excel(writer, sheet_name='position')
            self.value.to_excel(writer, sheet_name='value')
            self.price.to_excel(writer, sheet_name='hist_price')

    @staticmethod
    def _winning_gatherer(stock_series: pd.DataFrame):
        """
        根据交易日志里某个资产的交易情况，输出胜率等
        :param stock_series:
        :return:
        """
        current_holding = False
        start_date_this_round = None
        value_this_round = 0
        history = pd.DataFrame(columns=['start_date', 'end_date', 'value_change', 'win'])
        for _, this_row in stock_series.iterrows():
            value_this_round += this_row['volume_change'] * this_row['price']
            if current_holding:
                if this_row['current_position'] == 0:
                    # 之前有仓位，但是现在平仓了
                    current_holding = False
                    if value_this_round > 0:
                        win_this_round = True
                    else:
                        win_this_round = False
                    history = history.append({
                        'start_date': start_date_this_round,
                        'end_date': this_row['date'],
                        'value_change': value_this_round,
                        'win': win_this_round
                    }, ignore_index=True)
                    value_this_round = 0
            else:
                # 之前没仓位，现在肯定是建仓
                current_holding = True
                start_date_this_round = this_row['date']
        return history

    def winning_analyzer(self):
        """
        根据每个完整的建仓、平仓过程，计算收益情况和胜率
        :return:
        """
        print('Generating winning analysis report... ')
        return self.log.groupby('security').apply(self._winning_gatherer).droplevel(1).sort_values(
            by='start_date').reset_index()

    def calculate_winning_percentage_by_security(self):
        """
        根据交易日志分资产计算胜率
        :return:
        """
        print('Generating winning percentages by security... ')

        def _single_winning_percentage(stock_series):
            winning_gather = self._winning_gatherer(stock_series)
            win_state = np.where(winning_gather['win'], 1, 0)
            win_count = sum(win_state)
            total_count = len(win_state)
            if total_count != 0:
                # 有可能发生建了仓，但是还没有清仓的可能
                winning_pct = sum(win_state) / len(win_state)
            else:
                winning_pct = None
            return pd.Series({
                'win_count': win_count,
                'total_count': total_count,
                'winning_pct': winning_pct,
            })

        return self.log.groupby('security').apply(_single_winning_percentage)

    def calculate_winning_percentage_by_year(self):
        """
        根据交易日志分年度计算胜率
        :return:
        """
        # 计算逐年的YTM
        print('Generating winning percentages by year... ')
        YTM_by_year = self._get_annual_YTM()
        ret = pd.DataFrame(columns=['year', 'win_count', 'total_count', 'winning_pct'])
        history = self.winning_analyzer()
        for i, rolling_year in enumerate(range(self.start_date.year, self.end_date.year + 1)):
            this_year_trading_log = history[history['end_date'].dt.year == rolling_year]
            win_state = np.where(this_year_trading_log['win'], 1, 0)
            win_count = sum(win_state)
            total_count = len(win_state)
            if total_count != 0:
                # 有可能发生建了仓，但是还没有清仓的可能
                winning_pct = sum(win_state) / len(win_state)
            else:
                winning_pct = None
            ret = ret.append({
                'year': rolling_year,
                'win_count': win_count,
                'total_count': total_count,
                'winning_pct': winning_pct,
                'YTM': YTM_by_year[i],
            }, ignore_index=True)

        return ret.astype({'year': int, 'win_count': int, 'total_count': int, "winning_pct": float, "YTM": float})
