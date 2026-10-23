# HyperBasket Casework 2 — 新版 Task 1（a–f）

本目录使用 2026-10-12 新版包重新完成 Task 1，旧目录 hyperbasket_task1a 保留。
先读 outputs/Task1_Report_EN.pdf（英文报告），再看 outputs/demand_estimates.csv（平台上传文件）。

## 新旧数据差异

- 新版共 132,389 单、502,730 条商品明细、529,134 件；日均 362.71 单。
- 旧版每日需求导出共 1,523,721 件，新版为其 34.726%。订单被抽稀，不能只把旧预测标准差和分位数同比缩小。
- 新版修复了异常的 23:59:58 时间戳，placement_minute 为从第 0 天起的累计分钟；分析时使用 placement_timestamp 的小时。
- order_id 已重编，因此不能用新版订单 ID 连接旧版明细。
- 9 个数据文件的 SHA-256 全部匹配包内校验表；所有订单的篮子件数均与明细求和一致。

## 各小问结果与方法

### 1(a) 每个 SKU 的时变日需求分布

从 Orders 与 Order_Lines 连接汇总，不依赖旧 SQL 导出。构建 2,000 SKU × 365 天面板；在验证文件完整、主键及连接正确后，无销售记录的 SKU-date 记为零。

比较三个候选：A 最近 8 周同星期均值；B Poisson GLM 的星期效应 + 对数线性趋势；C 在 B 上加年度 sin/cos。先用 253 天训练、56 天验证；按 Poisson deviance 选择结构，1% 范围内选简单模型；锁定后，用前 309 天重训并在最后 56 天独立测试。

新版选择 B。验证 deviance A/B/C = 2.0857/0.7691/1.0000；测试 B 的 WAPE 61.65%、MAE 0.4306 件/SKU-day。季节项已测试，但本次没有提高验证表现，不能为了“包含季节性”强行采用效果更差的 C；这也不能证明真实需求没有季节性。只有一年的数据，趋势与年度季节效应难以区分。

用训练残差估计 NB2 离散度 alpha，方差 = mu + alpha × mu²；alpha 近零采用 Poisson。输出每日均值、标准差及整数 p50/p90/p99。测试覆盖率为 81.95%/95.08%/99.23%；零需求较多使离散分位数覆盖率常高于名义值。这不是均值置信区间，也不等于库存满足率。14,000 次拟合状态均为 ok。

最终用全年重拟合 B，生成 2027-09-07 至 2027-12-27 的 112 天预测。平台要求文件为 224,000 行，列严格为 sku_id,date,mean_units,sd_units。逐 SKU 系数、残差 lag-1 相关及测试指标一并保留。

局限：未纳入参数估计不确定性；未建立跨 SKU、跨日期联合需求模型；56 天测试短于 112 天外推。Task 2 不能把每日 p90/p99 直接相加，需按实际补货日历建模累计需求。

### 1(b) 星期与日内高峰

星期均值使用各星期的实际天数（周一 53 天，其余 52 天）；所有日期 × 营业小时补全后统计。

- 平均最忙星期：周六，473.62 单/日。
- 平均最忙时段：周六 17:00–18:00，75.31 单/小时。
- 历史最大单日：2026-12-26，600 单。
- 历史最大单小时：2026-12-19 17:00–18:00，108 单。

不能用全年平均 45.34 单/营业小时直接代替高峰容量要求。

### 1(c) 篮子大小

每单平均 3.997 件；p50/p90/p99 为 3/7/12 件。每单平均 3.797 种不同 SKU。分别提供件数与商品种数的完整经验概率分布，便于后续区分按件与按行发生的拣选工作。

### 1(d) 共购关系

以订单中是否出现 SKU 为二元变量，计算 support、两个方向的 confidence 和 lift。单侧超几何检验 + BH 多重检验校正，覆盖全部 1,999,000 个可能商品对；无共现的商品对 p=1。筛选 q≤0.05 且 lift>1 的 29,025 对。

最高频显著对 P0001/P0002 共现 11,898 单，占全部订单 8.99%，但 lift 仅 1.030。说明高频、统计显著与强关联不是一回事。完整显著对、所有观察到的商品对及共购次数前 100 对均已输出。

这是未控制篮子大小、时间因素等的边际关联筛选，不能解释为因果关系；商品对之间存在依赖，BH 名义 FDR 的保证依赖统计假设。布局建议需结合温区、访问频率和路径验证，而不能直接按显著性把商品放在一起。

### 1(e) 温区组合

54.36% 篮子跨两个及以上温区；9.72% 涉及全部三温区。全部组合：Ambient 43.63%；Ambient+Chilled 30.95%；Ambient+Frozen 13.09%；三温区 9.72%；Chilled 1.42%；Chilled+Frozen 0.60%；Frozen 0.59%。

跨区拣选是常态。比例不等于 staging 容量，后续还需结合到达峰值和驻留时间。

### 1(f) OTS：区分 OTS 场景与 OTD promise

PDF 第 6 页规则及参数 CSV 支持：OTS 可用时间 = OTD promise − estimated delivery minutes。在 20/30/45 分钟 OTD 下，平均可用 OTS 为 8.675/18.675/33.675 分钟；标准差均为 2.584 分钟，均无负值。提供逐单 deadline 和 courier arrival，且没有重复扣减 courier offset。

Case narrative 将 Fast/Medium/Slow 明确定义为 QF 内部所需的 OTS 范围：4–5、5–7、8–10 分钟。它们不是 OTD promise。20/30/45 分钟是另一组 delivery promise；依据规则 5，可用 OTS budget = OTD promise − estimated delivery minutes。

因此两组数字不冲突：前者是设施需要达到的 OTS 性能场景，后者是客户交付承诺。代码分别输出 ots_scenario_requirements.csv 与 ots_budget_summary.csv，不再把 4–5/5–7/8–10 当成 OTD promise 做负预算测试。

## 复现

使用 Python 3，先安装 requirements.txt 中依赖。在本目录运行：

```bash
python analyze_task1.py --source '/path/to/Casework 2_HyperBasket Quick Fulfiller Design_v2'
python forecast_task1a.py --input outputs/daily_demand.csv --output outputs --horizon 112
python report_task1.py
python package_results.py
```

本次统计分析使用系统 Python 3.14（已有 scipy、statsmodels）；报告生成使用 Codex bundled Python（已有 reportlab）。同一个安装齐依赖的 Python 环境即可完成全部步骤。环境版本记录在 outputs/run_manifest.json。

## 输出定位

- outputs/Task1_Report_EN.pdf / .md：英文分析报告，可编辑文本另存。
- outputs/demand_estimates.csv：Task 1 平台上传文件。
- outputs/daily_demand.csv：由新版原始订单生成的 730,000 行历史 SKU-day 表。
- outputs/future_daily_forecast.csv：含离散度和 p50/p90/p99 的完整预测。
- outputs/model_comparison.csv、test_by_sku.csv、test_predictions.csv：验证和独立测试证据。
- outputs/model_parameters_and_diagnostics.csv：每次拟合的参数和诊断。
- outputs/weekday_profile.csv、weekday_hour_profile.csv、hourly_operations.csv：高峰分析。
- outputs/basket_size_units_distribution.csv、distinct_skus_distribution.csv：篮子分布。
- outputs/significant_sku_pairs.csv、observed_sku_pairs.csv、top_100_copick_pairs.csv：共购分析。
- outputs/zone_combinations.csv：七类温区组合。
- outputs/ots_budget_summary.csv、order_staging_deadlines.csv、ots_scenario_requirements.csv：OTS 规则计算与冲突敏感性。
- outputs/data_audit.json、run_manifest.json、upload_validation.json：输入、拟合和平台格式校验。

代码与计算由 AI 辅助完成；团队应自行审核、解释模型与局限，并据实际参与情况填写课程要求的 AI 使用声明。本目录没有代写“团队已经独立审核”的事实声明。
