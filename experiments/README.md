# 统一科研实验入口

Python >=3.8，仅标准库；沿用现有环境，不执行 pip、不导入 torch、不连接网络。
该层负责资产来源与离线评估；现有训练代码仍保留在服务器，尚未接入。

结构：`experiments/` 放管理与评估入口；`docs/` 放审查和协议；`tests/` 放管理层测试。
旧 `code/`、`gpu/`、`results/` 不迁移。服务器建议新增 `/root/autodl-tmp/caea-runs/<唯一实验ID>/`，
其下放 `assets.json`、`protocol.json`、`environment.txt`、`predictions.jsonl`、`metrics.json`、`logs/`。
checkpoint 引用已有目录，不复制大权重。不要向公共仓库上传敏感日志、原始私有数据或权重。

## 先登记现有资产

把 `assets.example.json` 复制成自己的配置，用真实已有路径替换占位符。
每个种子的 adapter、日志、训练配置、数据清单、已有翻转结果分别登记，添加 `seed` 等来源字段。
路径可以是 root 下的绝对或相对路径；root 外和指向 root 外的软链接会被拒绝。
工具仅哈希显式指定且 <=64 MiB 的文件，目录只登记引用，不遍历权重。

```bash
cd /root/autodl-tmp/你的仓库目录
python -m experiments.manage inventory /root/autodl-tmp/你的资产配置.json --out /root/autodl-tmp/caea-runs/你的唯一ID/assets.json
```

路径占位符必须替换。输出用独占创建，已有文件会报错；换新的实验 ID，不删除旧文件。
它不会检查 GPU 兼容性。保留服务器现有环境，单独保存 `python --version`、`python -m pip freeze`、
`nvidia-smi` 输出，并核对当前 torch 的真实 BF16 小试，切勿直接运行旧安装脚本。

## 统一逐样本预测格式

一行一个源情境的一对干预，不把两条响应当成两个独立样本：

```json
{"source_id":"heldout-001","method":"evidence_dpo","seed":"实际种子值","condition":"flip","target_a":0.2,"target_b":0.8,"p_a":0.3,"p_b":0.7,"status_a":"ok","status_b":"ok"}
```

condition 为 `stance`（固定证据、相反立场）、`paraphrase`（证据语义不变）或 `flip`（证据改变目标）。
目标是 H 的概率，两个干预中 H 的含义必须一致。目标仅用于评分，不向模型泄漏。
失败状态为 `parse_failure` / `refusal`，对应 p 必须为 null；失败的平方误差按预设最大值 1 计分。
在协议中冻结概率提取规则：选项归一化概率与自由文本自报概率不能混在同一比较中。
现有不同格式需要显式转换并保留原文件与转换代码，不自动猜测字段。

```bash
python -m experiments.manage evaluate /root/autodl-tmp/caea-runs/你的唯一ID/predictions.jsonl --out /root/autodl-tmp/caea-runs/你的唯一ID/metrics.json
python -m unittest discover -s tests -v
```

输出按方法、训练种子、条件分别汇总：失败数、含失败惩罚的目标概率平方误差、有效配对绝对变化、翻转方向正确率与更新误差。
概率目标的平方误差不是 Bernoulli 结果的原始 Brier score；不给无标签文本附会正确率。
这是描述统计，未实现置信区间或显著性检验；方法比较前须检查相同源 ID 覆盖，不能直接比较不匹配的组均值。
下一阶段方案见 [实验协议](../docs/NEXT_EXPERIMENTS_zh.md)。
