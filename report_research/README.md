# 研报因子与新增交易状态实验

最终报告：`output/pdf/report_code2_constraints.pdf`。原 `code2.py`、原报告及前两轮结果均保留。

本轮动态中证500完整研究保存于 `results_code2_constraints/`，全A价格相关家族补充保存于 `results_research_all_a_constraints/`。覆盖2016—2018年，模型共同样本外区间为2017年2月至2018年12月。

主要改变是接入历史ST、停复牌时段、实际涨跌停价和官方复权因子；同时按本地研报修正SmartMoney、CPV、CDPDP′、理想振幅、理想反转、CDPP，保留旧构造配对对照。预测与持有期限分别对应5、10、20交易日，自然月末研究另行处理。

多空主表每腿100%净资产、总名义敞口200%，假定可借券及保证金融资，每次实际买卖收取0.0015费用。每腿50%及执行约束/费用敏感性均另列。结果不能视为无融资、无借券限制的实盘收益。

## 复现

在当前作业根目录、已安装所需依赖的Python环境运行：

```sh
python code2_new.py --with-xgboost
python audit/full_a_price_factors.py
python audit/constraints_sensitivity.py
python -m unittest discover -s tests -p 'test_code2*.py' -v
python report_research/build_report.py
```

报告采用XeLaTeX，`report_ready.tex`已经展开表格；从`report_research/`编译两次。报告的表格和图像由真实账本生成，不手填收益率。

压缩包含PDF、LaTeX源文件、图表、汇总CSV、核心代码、测试、运行日志和数据覆盖诊断。`source/`保持核心模块、`audit/`和`tests/`结构，可复制回作业根目录使用。完整原始行情和体积较大的逐笔成交、Parquet文件保留在工作区，不重复装入报告压缩包。

## 结果与验证

- `tables/summary.csv`：完整单因子及旧构造研究。
- `tables/oos_summary.csv`：共同样本单因子、ICIR-12及XGBoost。
- `tables/paired_summary.csv`：同样本、同交易规则的新旧公式配对。
- `tables/sensitivity.csv`：56个执行、费用及敞口组合。
- `tables/all_a_summary.csv`：全A价格相关家族对照，非全A完整六因子模型。
- `provenance/`：配置、数据覆盖、特殊限价冲突、证券代码贯通信息。
- `logs/`与`validation.json`：54项检查、420条费后组合账本和5,101,448笔成交的核对记录，以及训练标签和ICIR时间检查。

仅对实际开盘时段下单。等于涨停不买、等于跌停不卖；买不进留现金，卖不出保留持仓并重试。开盘价超出所给限价、无效限价和缺界限时禁止可疑成交，不事后删除该股票。上市年龄仍以首次正报价近似，缺少完整Barra暴露、大小单方向、借券供应及订单队列，报告已保留这些边界。
