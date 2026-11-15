# 量价多因子选股实验

当前只保留一个代码文件 **code2_new.py**，包含因子构造、历史交易状态、回测账本、XGBoost筛选调参、自动测试和报告生成。最终报告直接位于根目录 **report.pdf**。

| 文件或目录 | 内容 |
| --- | --- |
| code2_new.py | 唯一Python代码和运行入口 |
| report.pdf | 当前最终报告，沿用原报告字体 |
| report/ | 当前报告的LaTeX源文件、模板、图像和表格 |
| results/ | 当前实验结果、配置、调参记录、成交与净值账本 |
| data/、otherdata/ | 原始数据 |
| 价量多因子模型/、作业二.docx | 原研报与作业要求 |
| requirements.txt、Makefile | 依赖和运行快捷命令 |

历史代码、报告和辅助文件只保存在Git中，当前目录不保留code.py、code2.py、独立的旧模块，以及tmp、output、audit、archive目录。可重算的历史缓存和废弃中间结果已清理。

## 运行

```sh
conda activate quant
python code2_new.py --self-test
python code2_new.py --with-xgboost
python code2_new.py --report
```

`make all`执行实验和报告生成；`make test`运行内置65项测试；`make report`核验当前账本并生成根目录report.pdf。报告编译需要XeLaTeX；编译中间文件放在系统临时目录，成功或失败后均自动清理。

仅重跑模型使用`python code2_new.py --models-only`。仅验证或准备因子缓存使用`python code2_new.py --prepare-only`。缓存按因子定义、辅助行情签名与文件哈希校验；原始数据或定义变化后应移除`results/input/`再重建。报告命令使用默认`results/`中的当前结果。

## 实验规则

九个候选因子为SmartMoney、CPV、CDPDP′、IdealAmplitude、IdealReversal、CDPP、CDPDP-4、LowAmpMomentum、Skewness。尾部收益因子不进入当前实验和模型。

主样本为2016—2018年动态中证500。十组等权，G10为看好组；多头目标100%，理论多空每腿目标100%、总名义仓位200%。T日收盘信号，T+1开盘按历史ST、09:30停复牌及实际涨跌停价执行。受阻买入资金留现金，受阻退出头寸保留并重试。假定可借券及相应融资，未另估借券和市场冲击成本。

**手续费为双边千三：每次买、卖各0.0015，完整往返0.003。** 两条腿均按实际成交额收费。年化收益按净值几何年化，夏普按日收益计算，仅展示当前口径。

XGBoost分别预测5/10/20交易日连续开盘收益的截面去均值。模型每月使用过去12个月、标签已实现的样本。相关性去重和择优仅在训练期执行，每个内层验证折重新筛选。季度搜索96组候选，前16组晋级3折时间验证；使用历史验证RMSE早停，按验证RankIC稳定性择优。相关阈值、因子预算与树模型参数共同搜索。

当前结果包含24次季度搜索、3456次验证拟合、69个月度模型。此次单文件整理保持因子、交易和模型计算逻辑不变，沿用已完成的实验结果，重新执行测试、输入校验和726条组合账本核验。

## 结果检索

- `results/factor_summary.csv`：九因子、四种调仓期限、十组及多空绩效。
- `results/oos_summary.csv`：共同预测日历下的单因子和模型绩效。
- `results/long_short_summary.csv`：G10多头及理论多空汇总。
- `results/feature_selection_summary.csv`：69个模型的评分、去重理由与入选名单。
- `results/best_parameters.csv`：季度最优参数；完整候选及验证折记录在`results/models/`。
- `results/run_config.json`：配置、源码及输入校验值、单文件整理来源。
- `results/validation.json`：账本、费用、训练时间及相关性核验。

大体积行情、缓存、逐笔账本和模型权重保留在当前数据或结果目录；代码、配置、汇总、搜索记录和报告纳入Git。

## Git历史

本目录是独立本地Git仓库，无远程推送。旧文件从历史提交读取，无需保留当前副本：

```sh
git log --oneline
git show 0898d32:code.py
git show ffea9f9:code2.py
git show 20a0cf5:code2_new.py
git show 8475f6c:output/pdf/report.pdf > /tmp/report-v6.pdf
```

v1 `0898d32`保存原code.py；v2 `ffea9f9`保存原code2.py及原报告；v3 `fc27027`、v4 `28a827a`、v5 `20a0cf5`、v6 `8475f6c`保存此前各轮版本。`4228193`补存原始笔记本和报告素材；之后的最新提交为本轮单文件整理。
