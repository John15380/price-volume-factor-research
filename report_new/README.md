# 修订报告及XGBoost结果

报告保持原作业“摘要、课题说明、数据处理、因子公式、有效性分析、因子思考、结论、代码输出”的主体结构，另附全时期统计表和共同样本全部周/月频图。正文共23页、24幅图，PDF内嵌两个完整Python源码附件。

主报告比较采用2017-02-03至2018-12-28的共同信号区间，468个日收益；完整2016至2018年单因子结果在附录单列。六个单因子与XGBoost使用完全相同的股票日样本、日历及费用规则。三种期限各23个月度模型，总计69个模型；不使用测试期调参或早停。

共同样本理论多空结果：

| XGBoost期限 | 年化收益 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: |
| 5日 | 4.49% | 1.11 | -7.25% |
| 10日 | 2.08% | 0.55 | -4.20% |
| 20日 | 2.94% | 0.80 | -3.59% |

这些是50%多头、50%空头目标下的费后理论结果，并非保证可实现的实盘收益。CPV在相同样本三个期限下的理论多空年化均高于XGBoost。缺失行情、受阻头寸、成交及借券假设在报告中披露，未为得到更高收益而删除它们。

## 文件

- `report_code2_new.pdf`：打包版报告；工作区正式PDF位于 `output/pdf/report_code2_new.pdf`。
- `report_new.tex`：章节编辑模板；`report_ready.tex`为已展开表格、可直接编译的版本。
- `values.tex`、`tables/`、`figures/`：由真实结果自动生成的数值、表格与图片。
- `source/`：两个策略源码、说明及26项测试的源文件。
- `validation.json`：234条组合账本、1,898,300笔成交的现金、费用、净值及绩效对账记录。
- `execution_log.txt`、`tests_output.txt`：实际运行记录与测试结果。

## 在原工作区重跑

```sh
/opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python code2_new.py \
  --with-xgboost --output-dir results_code2_new_report
/opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python report_new/build_report.py
```

`results_code2_new_report/`保存全部39个案例（18个完整时期单因子、21个共同样本对照）、预测、模型、日账本与成交记录。打包报告不包含课程原始分钟数据；若独立重跑，把`source/`内两个主程序放入包含`data/`的项目根目录后运行。

## 仅重新编译PDF

无需原始行情，进入本目录执行两次：

```sh
xelatex -interaction=nonstopmode -halt-on-error \
  -jobname=report_code2_new report_ready.tex
```

需要XeLaTeX、ctex、attachfile2等TeX Live组件。中文优先使用当前Mac的Noto字体；其他环境会回退到TeX Live的Fandol字体。

表格数字应修改数据后重新生成，不宜直接手工更改。要编辑正文，修改`report_new.tex`后重新执行`build_report.py`，再编译`report_ready.tex`。
