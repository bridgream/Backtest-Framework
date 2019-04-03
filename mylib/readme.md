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