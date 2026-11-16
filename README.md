# 量价多因子实验：2016—2017训练，2018回测

当前只有一个代码文件 **code2_new.py**；最终PDF直接输出在根目录 **report.pdf**，沿用原报告字体。Table 2及主要数表使用原模板正文五号字。

| 路径 | 当前内容 |
| --- | --- |
| code2_new.py | 因子、交易状态、模型、回测、基准、71项测试和报告生成 |
| report.pdf、report/ | 最终报告及其LaTeX、图表和模板 |
| results/CSI300/ | 历史沪深300输入、固定模型、训练期验证及2018测试 |
| results/CSI500/ | 历史中证500同口径实验 |
| results/AllA/ | 本地覆盖全部A股同口径实验 |
| results/ | 跨池比较、选型、十分组、多空评价和核验汇总 |
| data/、otherdata/ | 原始数据 |
| 价量多因子模型/、作业二.docx | 原研报与作业要求 |

历史code.py、code2.py、旧报告及辅助产物只保存在本地Git。当前不保留tmp、output、audit、archive等目录。

## 运行

```sh
conda activate quant
python code2_new.py --compare-universes
python code2_new.py --report
```

`make all`执行三池实验、比较和报告；`make test`运行71项内置测试；`make report`核验当前结果并编译根目录report.pdf。报告需要XeLaTeX，编译中间文件放系统临时目录并自动清理。

单独重跑某一池：

```sh
python code2_new.py --universe 300 --with-xgboost
python code2_new.py --universe 500 --with-xgboost
python code2_new.py --universe all --with-xgboost
python code2_new.py --self-test
```

三池比较可复用相同固定协议下已完整完成的实验；单池命令重新运行其模型和回测。`--universe`默认500，`--output-dir`指定实验结果根目录，报告命令读取默认的`results/`。`--prepare-only`只准备或校验输入。原始输入或因子定义变更后应清除对应input目录再重建；修改模型逻辑后应重跑单池命令。个股滚动分量可复用，各池截面处理分别重算。

## 固定训练及选型

- 最终训练信号位于2016—2017，标签退出日严格早于2018-01-01；训练标签的行情对象本身不包含2018。
- 三折内层时间验证分别为2017年Q1、Q2、Q3；折内训练只使用季度开始前已经实现的标签。
- 每池、每个5/10/20交易日期限搜索96候选，前16组晋级三折；早停使用训练期验证RMSE，参数按验证RankIC稳定性选择。每折单独计算相关性去重与因子质量。
- 2017年前三季度的时间验证预测用于选股票池和期限：每个因子及XGBoost在9个候选中，按验证费后多空年化择优。此步骤不读取2018收益。
- 最后用2016—2017完整已实现样本重新筛选因子并拟合一次，2018全年权重和参数冻结。2018的因子及动态成分仍按当日信息更新。
- 三池共9次搜索、1296次验证拟合；另拟合27个训练期验证模型和9个最终固定模型。2018重训次数为0。

训练期验证参与联合选型，只用于选择。当前是既有样本上的固定划分回测，不将历史验证数字视为新增时期的前瞻证据。

## 因子与交易

候选共9项：SmartMoney、CPV、CDPDP′、IdealAmplitude、IdealReversal、CDPP、CDPDP-4、LowAmpMomentum、Skewness。尾部收益不进入研究。低振幅长端动量高值做多，其余低值做多；模型高分做多，方向不依2018结果翻转。

十组等权，G10为看好组，每个多头组目标100%；理论多空多、空各100%，总名义仓位200%。T日收盘信号，T+1开盘执行。2017最后交易日生成首次测试信号，2018首个交易日开盘建仓。使用历史ST、09:30停复牌及真实涨跌停价；受阻买入留现金，退出受阻保留重试，末日尝试退出。假定借券及融资可得，未另估借券和冲击成本。

**手续费为买卖各0.0015，完整往返0.003。** 年化按250交易日几何复利，夏普按算术日收益计算，只保留当前口径。

基准是同池全体合格股票等权，不按因子缺失或模型预测过滤，使用相同日历、开盘成交限制、现金处理及费用；它不是官方市值加权指数。另以相同信号及限制、费用率为0重跑基准。预测覆盖差异也会影响相对收益。

## 结果文件

各池分别有`validation/`、`test/`账本及对应summary.csv，`models/`保存固定权重、参数、训练边界、入选因子和验证/测试预测，`benchmarks/`保存同期等权基准。

- `results/universe_comparison.csv`：90个候选的验证多空年化及2018各项结果，报告主评多空。
- `results/validation_candidates.csv`：选池、选期限的全部训练期验证候选。
- `results/selected_cases.csv`：仅按验证多空年化选出的十项策略，含2018多空累计、年化、夏普及回撤。
- `results/decile_returns.csv`：训练期验证与2018各900条十分组绩效，含累计、年化和夏普。
- `results/benchmark_summary.csv`、`benchmark_month_returns.csv`：等权累计及月度收益。
- `results/best_parameters.csv`、`feature_selection_summary.csv`、`training_summary.csv`：9个最终模型汇总。
- `results/validation.json`：现金账本、费用、标签日期、模型冻结及选型范围核验。

报告主评多空收益。每项策略同一张图的上半部叠加G1—G10、多空和股票池等权净值，下半部保留十分组费后累计收益柱状图；全部图注明股票池与期限。等权只作市场背景，两种组合的名义敞口不同。三池其他测试结果完整保留，但不根据2018最高收益替换选图组合。

## 合成因子训练与lag

训练标签是每只股票未来N交易日开盘收益减同日可用标签的截面均值；不是组合多空收益或RankIC数值。训练损失保持`reg:squarederror`，验证RMSE用于早停，RankIC稳定性选择超参数；股票池和调仓期限按训练期验证的费后多空年化选择。2018不参与任何上述选型，9个模型权重及预测本轮均复用。

- 特征附加lag为0：T收盘后已知的因子，未额外整体shift。
- 执行lag为1个交易日：T收盘信号于T+1开盘执行。
- 标签跨度N为5/10/20：从T+1开盘持有至T+N+1开盘，退出偏移分别为6/11/21。
- 因子10/20/160日计算窗口是回看窗口，不是执行lag；所有偏移按交易日历。

实际训练末日和上述lag值见`results/training_lag.csv`。所选十项策略的100条柱状图数值见`results/selected_decile_returns.csv`。十分组累计收益之差不等于独立多空账本收益。

## Git历史

本目录是独立本地仓库，无远程推送。旧文件从历史读取：

```sh
git log --oneline
git show 0898d32:code.py
git show ffea9f9:code2.py
git show 58a98cc:code2_new.py
```

v1 `0898d32`为原code.py，v2 `ffea9f9`为原code2.py；v3 `fc27027`、v4 `28a827a`、v5 `20a0cf5`、v6 `8475f6c`保留此前各轮。`4228193`补存原始笔记本及报告素材，v7 `58a98cc`为单文件整理。v8 `97a5b0a`记录固定训练及2018回测；最新提交记录本轮多空评价、组合净值、十分组柱状图和lag说明。
