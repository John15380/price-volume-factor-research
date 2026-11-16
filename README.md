# 八因子与XGBoost：两指数、2016—2017训练、2018回测

当前唯一源码为 `code2_new.py`，最终报告为根目录 `report.pdf`。股票池仅保留历史沪深300和中证500，期限为5、10、20交易日；八个因子及XGBoost各六组，共54组结果。字体沿用原模板，主要表格为10.5pt。

## 实验口径

候选为SmartMoney、CPV、CDPDP′、IdealAmplitude、IdealReversal、CDPDP-4、LowAmpMomentum、Skewness。CDPP不进入输出或模型。五分组等权，Q1看好、Q5看差，预先固定做多Q1、做空Q5，保留真实盈亏，不取绝对值。低振幅动量和模型高值做多，其余单因子低值做多。

T日收盘形成信号，T+1开盘执行。每腿目标为组合净资产的100%，收益以组合净资产计算。多头保留ST、停复牌和涨跌停检查；空头按既定理论无约束假设执行，但缺少有效开盘报价时不虚构成交。买卖各按实际成交金额收1.5‰，完整往返3‰，借券费与融资费为0。

2016—2017训练，2018全年冻结模型。2017前三季度分别在季度开始前进行嵌套选参和因子去重，标签实现时间不得超过参数可知日。两池三期限保留六个最终模型、18个外层模型，对应24次搜索、3456次内层拟合；每次96组候选、16组晋级三折。目标为个股N日开盘收益减同日截面均值，以平方误差训练，日RankIC稳定性选参。

本轮沿用此前冻结模型，核对学习AST、输入与模型哈希，重新预测与已保存值一致；没有因缩小展示范围而重选参数。分钟因子及其定义保持原口径。历史核查说明和旧范围结果保存在Git历史中。

## 报告与选图

每项策略列出两池三期限的六组结果，加粗2018费后多空累计收益最高的一组。所有九张收益图统一选取九项策略平均费后累计收益最高的配置；这是2018结果的事后展示，不用于训练或调参。

主结果只展示IC、ICIR、RankIC、RankICIR、IC观测数，以及2018累计多空收益、年化波动、夏普、最大回撤、日均换手。报告及主指标CSV删除2017验证年化和2018年化收益列。底层组合摘要保留绩效复算所需原始字段，训练验证仍正常执行。

每张图上半部叠加Q1—Q5、多空及同池等权背景净值，下半部为五分组累计收益柱状图。所有图标明股票池和期限。

## 运行

```sh
conda activate quant
python code2_new.py --compare-universes --reuse-frozen-models
python code2_new.py --report
python code2_new.py --self-test
```

单池可用 `--universe 300` 或 `--universe 500`。去掉 `--reuse-frozen-models` 会重新训练；该选项仅允许学习代码与v13基准一致时使用。`make all`执行两池实验与报告。PDF需要XeLaTeX，中间文件仅位于系统临时目录。

## 文件

| 路径 | 内容 |
|---|---|
| `results/factor_metrics.csv` | 全部54组主指标，BestIn2018标记加粗配置 |
| `results/universe_comparison.csv` | 两池六组配置的完整结果明细 |
| `results/selected_cases.csv` | 各策略自身累计收益最佳配置 |
| `results/plot_configurations.csv` | 六组统一选图配置的平均累计收益排名 |
| `results/plot_cases.csv` | 统一画图配置下九项策略 |
| `results/selected_quintile_returns.csv` | 九张图的45个柱值 |
| `results/CSI300/`、`results/CSI500/` | 输入、冻结模型、成交、账本、IC及内部验证 |
| `results/validation.json` | 672条账本序列、模型时序和报告核验记录 |

原始数据、研报与作业要求保留。当前目录不建立tmp、output、audit或archive文件夹；历史代码和旧结果仅保留在本地Git。

## 版本

v1为原code.py，v2为原code2.py；v3—v16记录此前模型及核查过程。v17将当前结果范围缩减为沪深300和中证500，删除报告及主指标的两列收益年化结果，保留54组指标并重新选图。
