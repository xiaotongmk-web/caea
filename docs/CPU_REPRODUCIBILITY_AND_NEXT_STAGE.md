# CAEA CPU 复现与下一阶段实现

当前公开论文在 [paper/CAEA.pdf](../paper/CAEA.pdf)。此分支额外提供一个隔离的 CPU 验证入口和数据协议生成器。

`tools/check_reproducibility.py` 将 code/results 复制到临时目录，运行已归档的 CPU 检查，并保存日志、输入哈希、依赖版本和退出码。工具检查源目录在运行前后的哈希。输出目录须位于仓库外且尚不存在。

```bash
python tools/check_reproducibility.py --out /path/to/new-caea-replay
python -m unittest discover -s tests -v
```

回放使用已归档结果。已记录的请求、模型参数和模拟之间的区别见仓库首页。新实验优先级见[后续验证计划](../RESEARCH_PLAN_zh.md)。

## 下一阶段已实现的数据协议

`protocols/build_evidence_grid.py` 使用标准库，不加载模型，默认生成 60 开发源情境与 300 确认源情境。
每个源情境 2 证据水平 × 2 保义模板 × 3 立场 = 12 请求，总计 4320 请求。
证据翻转交换 P(E|H) 与 P(E|¬H)，保持 prior 与事件 H 含义；Bayes 公式给出不同目标。
同一证据的 stance/改写不改目标；无立场是额外基线。
原/反事实似然交换视作同一源，避免其跨开发/确认集重复。
输出独立 requests.jsonl 与 labels.scoring-only.jsonl，后者不得输入模型；protocol.json 保存种子和哈希。
重复运行到已有目录会失败。生成是设计提案，不等于外部预注册或已运行实验。

```bash
python protocols/build_evidence_grid.py --out /root/autodl-tmp/caea-grid-唯一ID
python -m unittest discover -s tests -v
```

这两种模板同时出现在两个 split，仅检验新证据三元组下的模板鲁棒性，**不是模板族外迁移**。
还需与服务器训练/历史测试证据做重合检查；没有拿到服务器数据前，不能宣称独立确认集未受污染。
协议一旦被用来选择方法，其确认集应降级为探索数据。

## 接入现有服务器实验的具体约定

1. 先登记真实模型/adapter 目录、三个实际种子、训练日志、训练代码版本、数据哈希、chat template 和依赖版本。
   未知来源标 unknown，不使用本 PR SHA 追认旧训练；原文件原位保留。
2. 推理适配器接收 requests 中 prompt/id，显式 local_files_only=True，固定生成长度/解码，
   顺序加载已有 base、Stance-DPO、Evidence-DPO，禁止自动回退下载。先 CPU 数据检查，再另行 GPU 推理小试；不重训。
3. 每请求保存 id、method、seed、raw_response、parsed_probability、status、model/adapter 标识、
   prompt/template 哈希、token 数、秒数与峰值显存。解析失败/拒答仍保留，不能过滤后才评分。
4. 评估器只按 id 联接 scoring-only 标签，拒绝缺失/重复/额外 id。组间须同源覆盖。
   固定证据比较 high-low，固定立场比较 counterfactual-original；分别报告概率平方误差、
   立场差、翻转方向和更新误差。自报概率的平方误差不能混称离散候选报告分布的期望风险。
5. 加入 evidence-first prompting、开发集简单收缩、公共证据求解器；若纳入 CAEA，必须加入同审计器 audit-only。
   按实际种子分别报告，再按源情境做配对 cluster bootstrap；所有干预与方法一起重采样。
   区间只条件于现有三种子，另外报告种子变异；冻结主要比较、多重校正与更新非劣界值。

本 PR 不猜测尚未同步的 adapter 格式或 TRL 训练入口，故不提供伪可运行的模型执行命令。
完成资产核对后，按上述约定实现适配器/评分器；新训练须由评估结果与消融问题驱动。
