# 因子定义再次复核版

最终报告：output/pdf/report_code2_definitions.pdf。此前report_code2_constraints.pdf及其运行目录冻结保留。核心入口仍为code2_new.py，默认输出已改为results_code2_definitions/。

本轮主要发现是理想振幅遗漏原始研报的有效日细则：20日内须剔除停牌及一字涨跌停，有效日不足10天才置空，按有效日中最高/最低价格各25%计算振幅差。此前要求20日全有效，且未在历史窗口剔除一字板。本轮按20200516原报告第5页修正，并按20210924第8页保持25%切割比例；没有按测试收益改方向或调参数。

聪明钱和CDPDP′主要公式核对一致；偏度、尾盘收益是原作业扩展，所给四篇研报没有对应定义，不能称为研报因子。数学退化时置0、分钟首笔收益、25%舍入、价格并列处理等仍是实现约定。详细说明见audit/因子定义复核.md。

## 结果目录

- results_code2_definitions/：完整中证500单因子、配对、共同样本XGBoost/ICIR、每日账本和费用/敞口敏感性。XGBoost共69个模型已重训。
- results_code2_definition_audit/：中证500、全A的理想振幅定义/股票池/期限对照，共28个案例，分别比较旧严格20日、有效日放宽但不排一字板、原始研报有效日规则，另有共同样本自然月配对。
- results_research_all_a_constraints/：此前的全A价格相关家族补充；其公式及数据不受本轮理想振幅修订影响。它不是全A完整六因子模型。

主要表格和图像在本报告目录tables/、figures/。validation.json保留所有账本核对数量和来源摘要，provenance/保留定义审计、独立数值核对和数据覆盖结果。58项测试通过。原始研报PDF、来源地址和校验值在audit/source_reports/。

## 复现

在作业根目录运行：

    python code2_new.py --with-xgboost
    python audit/recheck_factor_definitions.py
    python audit/constraints_sensitivity.py
    python -m unittest discover -s tests -p 'test_code2*.py' -v
    python report_definitions/build_report.py

独立核对脚本读取上轮冻结日行情，保证前后使用相同的交易约束；原始分钟算术则独立读取3只固定股票的10天数据核对。若在干净目录使用，须先保留/恢复此前results_code2_constraints/与results_research_all_a_constraints/的冻结日行情及汇总，或显式生成相同口径的基准输入。

新完整主实验从未改变的分钟原始指标缓存重算滚动窗口、截面因子、交易及模型；未直接复用旧预处理因子或旧预测。

报告用XeLaTeX对report_ready.tex编译两次。压缩包附核心代码、测试、汇总、图表、日志和补查研报的来源记录，完整原始行情及体积较大的逐笔成交Parquet仍在工作区。

多空主表每腿100%净资产、合计200%名义敞口，假定可借券及保证金融资，每次实际买卖费率0.0015；每腿50%对照另列。开盘成交仍使用实际历史交易状态，无法买入留现金、无法退出保留并重试。
