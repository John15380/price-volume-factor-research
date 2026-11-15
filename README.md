# 量价多因子选股实验

当前入口为 **code2_new.py**；原始`code.py`、`code2.py`保持原样。

## 文件位置

| 目录/文件 | 用途 |
| --- | --- |
| output/pdf/report.pdf | 当前唯一交付报告 |
| output/submission.zip | 报告、代码、配置、汇总和调参记录 |
| report/ | 当前报告的LaTeX、图像、表格及生成程序；沿用原报告字体和模板组件 |
| results/ | 当前唯一实验目录；含九因子、十组账本、XGBoost与全部调参记录 |
| results/input/ | 已验证的九因子及实际交易状态输入、缓存来源和校验值 |
| data/ | 原始数据，未移动或删除 |
| archive/ | 此前报告、实验及原作业辅助产物的可逆归档 |
| audit/ | 当前定义、版本和归档说明 |
| tests/ | 账本、因子、训练时间边界及筛选规则测试 |

## 当前实验规则

候选因子共9项：SmartMoney、CPV、CDPDP′、IdealAmplitude、IdealReversal、CDPP、CDPDP-4、LowAmpMomentum、Skewness。尾盘收益及简化构造对照不进入当前实验、模型或报告。

主样本为2016—2018年动态中证500。十组等权，G10统一为看好组；多头结果来自独立G10账本，多空来自G10多头/G1空头的独立账本。多头目标100%，研究多空每腿100%、总名义目标200%，假定可借券和相应融资。交易按历史ST、09:30停复牌及实际涨跌停价限制，未成交资金留现金、受阻退出保留并重试。

**手续费继续为双边千三：每次买、卖各0.0015，完整往返合计0.003。** 两条腿均按实际成交额扣费，初始、调仓、漂移再平衡和退出均计费；未另估借券及市场冲击成本。

XGBoost分别预测5/10/20交易日连续开盘收益的截面去均值。模型每月用过去12个月、标签已实现的样本训练；因子去重和择优仅在训练期进行。按训练期日截面Spearman相关及RankIC稳定性选择代表，每个时间验证折均重新筛选。季度搜索96组参数，前16组晋级3折连续时间验证，早停使用历史验证RMSE；按三折RankIC稳定性选择模型，树数取获胜候选三折最优树数中位数。相关阈值、因子预算和树模型参数共同搜索。

共24次季度搜索、3456次验证拟合、69个月度模型。测试月份收益不参与参数、因子或早停选择。稀缺历史因子的缺失由XGBoost原生处理，预测入口不使用未来是否有收益标签筛股票。

## 运行

```sh
conda activate quant
python code2_new.py --with-xgboost
python -m unittest discover -s tests -p 'test_code2*.py' -v
python report/build_report.py
```

也可执行`make all`依次运行实验、核验、编译报告并打包；`make report`只更新报告，`make package`只打包。

初次从原始数据计算可能较慢；现有输入已对九个因子全部股票日重新执行截面处理并校验。使用缓存时校验因子定义、辅助交易数据签名及缓存文件哈希，不复用模型预测。输入变化时需将`results/input/`移入归档后重建。

模型重跑可使用`python code2_new.py --models-only`。报告在`report/`中用XeLaTeX对`report_ready.tex`编译两次，构建目录放在`tmp/pdfs/current/build/`；最终PDF复制至`output/pdf/report.pdf`。

## 结果检索

- `results/factor_summary.csv`：九个因子、四种调仓期限、十组和多空的当前绩效。
- `results/oos_summary.csv`：共同预测日历下的单因子及模型样本外绩效。
- `results/long_short_summary.csv`：方便查看G10多头和理论多空的汇总。
- `results/feature_selection_summary.csv`：69个模型的因子评分、去重理由及入选名单。
- `results/best_parameters.csv`：24次季度搜索最终参数；每次完整候选、验证折和筛选过程在`results/models/`。
- `results/audit/validation.json`：现金、成本、持仓、净值、绩效及训练时间检查。

大体积行情、逐笔成交、缓存和权重文件保留在工作区，不装入Git和报告压缩包。代码、配置、汇总、搜索记录和报告均纳入版本管理。

## 本地Git版本

本目录为独立仓库，未设置远程推送。根据保存的源码和报告快照补建历史版本，提交时间为实际建库时间：

| 版本 | 提交 | 内容 |
| --- | --- | --- |
| v1 | 0898d32 | 原始code.py |
| v2 | ffea9f9 | 原始code2.py和报告 |
| v3 | fc27027 | 第一轮修复账本及XGBoost |
| v4 | 28a827a | 研报构造及历史交易状态 |
| v5 | 20a0cf5 | 上一次对话的完整产物 |
| v6 | 当前HEAD | 本轮九因子、调参、十组及目录整理 |

本轮只作一次最终提交，v5是其直接父提交，即`HEAD~1`。

```sh
git log --oneline
git show HEAD~1:code2_new.py
git show 0898d32:code.py
git show ffea9f9:code2.py
```

旧目录的迁移清单和关键文件哈希保存在`audit/archive_manifest.json`；历史完整产物仍在`archive/`中。
