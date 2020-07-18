2019/5/15 更新

optimization.py

1. 加入`Optimization.grid_search_target_calmar()`，这是一个基于回测数据调整仓位、以最大化 Calmar 比例的函数。



2019/5/6 更新

optimization.py

1. 加入`Optimization.grid_search()`。



2019/4/24 更新

backtest.py

1. 增加了``BackTestFramework.pct_change``属性，通过传入日涨跌幅矩阵，程序将自动检查涨跌幅，同时在输出日志时同时输出当日涨跌幅度。
2. 增加了一些安全性检查。

measure.py

1. 增加了``Measure.get_downside_deviation()``便于计算下行波动率。
2. 增加了``Measure.get_rolling_downside_deviation()``便于计算滚动的下行波动率。



2019/4/22 更新

backtest.py

1. 增加了``BackTestFramework.winning_analyzer()``功能，计算建仓到平仓过程中的价值变化和获胜情况。
2. 增加了``BackTestFramework.calculate_winning_percentage_by_security()``功能，分资产计算胜率。
3. 增加了``BackTestFramework.calculate_winning_percentage_by_year()``功能，分年度计算胜率。
4. 增加了``BackTestFramework.plot_rolling_sr(window_size)``功能以打印滚动Sharpe ratio的图像。
5. ``BackTestFramework.summary()``现在可以打印分年度的收益率了。
6. 消歧义：``BackTestFramework.yesterday``现在已更名为``BackTestFramework.last_trading_day``。
7. 增加了执行过程的提示，并打印在命令行里。
8. 一些细节修改。



2019/4/15 更新

backtest.py

1. 修复了一些内部错误，通过移除Pandas.DataFrame.append()，速度提升约15倍。
2. 允许用户关闭参数检验了。
3. 增加了``BackTestFramework.adjust_to_value()``功能，将仓位调整到对应的指定价值。
4. 增加了``BackTestFramework.log_to_excel()``功能，将交易日志导出到Excel文档中。



2019/4/10 更新

backtest.py（还存在 bug，待下次修复）

1. 逐步加入对期权的支持。现在所有交易单位为"手"，例如对于股票，默认一手为 100 股。
2. 给 benchmark 作图时，基准价值与初始现金成比例。
3. 手续费默认修改为 0。
4. 其他内部逻辑优化。现在速度又可以提升数十倍了。



2019/4/8 更新

backtest.py

1. 允许一次性交易多个标的，通过传入一组资产的名称构成的 **list**，不再接受 **tuple**。
2. 增加了``BackTestFramework.__finalize()``函数。
3. 修正了动态价值计算时错误调用乘法的问题。
4. 其他内部参数传递和逻辑优化。



2019/4/3 更新：

backtest.py

1. 允许一次性交易多个标的，通过传入一组资产的名称构成的 **tuple**。
2. 增加了``BackTestFramework.__trade_classifier()``来区分传入的是单个的名称还是 tuple，大幅减少代码量。
3. 增加了``BackTestFramework.adjust_to_portion_of_value()``，允许将交易的目标仓位定在目前资产组合价值的某个比例上。
4. 修正了``security_check``装饰器的 bug。
5. 修正了``BackTestFramework.__order_single()``函数的逻辑问题，现在不会触发卖空了。
6. 预留了期货乘数的接口``FuturesMultiplier``和期货交易接口``futures_list``。
7. 参数传入部分简化。
8. 调试输出部分使用 f-string 简化。
9. 错误类型调整。
10. 一些其他的逻辑调整。



2019/4/1 更新：

backtest.py

1. 允许一次性交易多个标的，通过传入一组资产的名称。



2019/3/25 更新：

backtest.py

1. 加入了"上个交易日"。

2. 从价格矩阵中读取开始和结束日期。
3. summary函数改进。
4. 其他bug修正。

measure.py

1. 使用多线程技术，快速计算滚动 Sharpe ratio，可比以往版本快 100 多倍。
2. 计算单个 Sharpe ratio 的函数名称作了修改。
3. 其他bug修正。